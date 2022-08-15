"""
    File name: MPFProcessHandler.py
    Author: Matthew Allen

    Description:
        This is the container for an MPFProcess object. It handles spawning the process and interfacing with it
        from the main process.
"""
import logging
import multiprocessing as mp
import time
import traceback
from queue import Empty
import psutil

from distrib_rl.MPFramework import MPFDataPacket, MPFTaskChecker

try:
    #We want true asynchronicity with as little task switching as possible so we want to spawn processes with independent
    # Python interpreters. However, the start method may have already been set before we run this code so we'll catch
    # that exception here and ignore it.
    mp.set_start_method('spawn')

except RuntimeError:
    pass


class MPFProcessHandler(object):
    def __init__(self, input_queue=None, input_queue_max_size=1000, output_queue_max_size=1000):
        self._output_queue = mp.Queue(maxsize = output_queue_max_size)
        self._process = None
        self._MPFLog = logging.getLogger("MPFLogger")
        self._terminating = False

        if input_queue is None:
            self._input_queue = mp.Queue(maxsize = input_queue_max_size)
        else:
            self._input_queue = input_queue

    def setup_process(self, process, shared_memory = None, cpu_num = None):
        """
        This function spawns a process on a desired CPU core if one is provided. It expects an instance of
        MPFProcess to be passed as the process argument.

        :param process: An instance of an MPFProcess object.
        :param cpu_num: The CPU core to spawn the process on. Only available on Linux.
        :return: None.
        """
        self._MPFLog.debug("Setting up a new MPFProcess...")
        self._process = process

        #Setup process i/o and start it.
        process._inp = self._input_queue
        process._out = self._output_queue
        process.set_shared_memory(shared_memory)
        process.start()

        #If a specific cpu core is requested, move the process to that core.
        if cpu_num is not None:
            self._MPFLog.debug("Moving MPFProcess {} to CPU core {}.".format(process.name, cpu_num))
            if type(cpu_num) not in (list, tuple):
                cpu_num = [cpu_num]
            psutil.Process(process.pid).cpu_affinity(cpu_num)

        self._MPFLog.debug("MPFProcess {} has started!".format(process.name))

    def put(self, header, data, delay=None, block=False, timeout=None):
        """
        Function to send data to the process contained by this object.
        :param header: Header for data packet.
        :param data: Data to send.
        :param delay: Optionally delay for some period after putting data on the queue.
        :param block: mp.Queue block argument.
        :param timeout: mp.Queue timeout argument.
        :return: None.
        """

        if self._check_status():
            return

        if not self._input_queue.full():
            #Construct a data packet and put it on the process input queue.
            task = MPFDataPacket(header, data)
            self._input_queue.put(task, block=block, timeout=timeout)
            del task
        else:
            self._MPFLog.debug("MPFProcess {} input queue is full!".format(self._process.name))

        if delay is not None:
            time.sleep(delay)

        del header
        del data

    def get(self, block=False, timeout=None):
        """
        Function to get one item from our process' output queue.
        :return: Item from our process if there was one, None otherwise.
        """

        if self._check_status():
            return None

        #First, check if the queue is empty and return if it is.
        if self._output_queue.empty():
            return None

        if not self._output_queue.empty():
            result = self._output_queue.get(block=block, timeout=timeout)
            return result()

        return None

    def get_all(self, block=False, timeout=None, failure_sleep_time=0.1, cleaning_up=False):
        """
        Function to get every item currently available on the output queue from our process. The implementation of
        this function looks a bit odd, but it has been my experience that simply checking if a queue is empty almost never
        results in an accurate measurement of how many items are actually available on the queue. This function
        tries its best to guarantee that all items will be taken off the queue when it is called.
        `
        :param block: mp.Queue block argument.
        :param timeout: mp.Queue timeout argument.
        :param cleaning_up: boolean to indicate that cleanup is happening. Do not modify.
        :return: List of items obtained from our process.
        """

        if self._check_status() and not cleaning_up:
            return None

        #First, check if the queue is empty and return if it is.
        if self._output_queue.empty():
            return None

        results = []
        try:
            while not self._output_queue.empty():
                try:
                    result = self._output_queue.get(block=block, timeout=timeout)

                    header, data = result()
                    results.append((header, data))
                    result.cleanup()

                    del result
                    failCount = 0
                except Empty:
                    return None
        except Exception:
            error = traceback.format_exc()
            self._MPFLog.critical("GET_ALL ERROR!\n{}".format(error))
        finally:
            return results

    def is_alive(self):
        """
        Function to check the status of our process.
        :return: True if process exists and is alive, False otherwise.
        """
        if self._process is None:
            return False

        return self._process.is_alive()

    def close(self):
        """
        Function to close and join our process. This should always be called when a process is no longer in use.
        :return: None.
        """
        if self._terminating:
            return

        self._terminating = True

        #Put an exit command on the input queue to our process.

        self._MPFLog.debug("Sending terminate command to process {}.".format(self._process.name))
        task = MPFDataPacket(MPFTaskChecker.EXIT_KEYWORDS[0], self._process.name)
        self._input_queue.put(task)

        #Get any residual items from the output queue and delete them.
        self._MPFLog.debug("Beginning residual output collection...")
        print("Calling get_all from shutdown...")
        residual_output = self.get_all(cleaning_up=True)
        if residual_output is not None:
            self._MPFLog.debug("Removed {} residual outputs from queue.".format(len(residual_output)))
            del residual_output

        #Note here that we do not join either the input or output queues.
        #A process handler should only have a single process, so if the user has passed
        #a joinable queue to this handler's process, they are responsible for closing it.

        self._MPFLog.debug("Beginning process termination...")
        self._join_process()
        self._MPFLog.debug("Successfully terminated MPFProcess {}!".format(self._process.name))

        # Get residual output again in case something happened between the first call to get_all() and _join_process()
        # We do this twice rather than deleting the above block because PyTorch Tensors don't release their reference
        # if .join() is called on a process before a queue containing a tensor is emptied, so we have to remove everything
        # from the queue, then call join, then remove any extraneous data from the queue again here.
        residual_output = self.get_all(cleaning_up=True)
        if residual_output is not None:
            self._MPFLog.debug("Removed {} residual outputs from queue.".format(len(residual_output)))
            del residual_output

        self._MPFLog.debug("All MPFProcess objects have been terminated!")

    def join(self):
        self.close()

    def stop(self):
        self.close()

    def _check_status(self):
        """
        Private function to check if our process is still running. If it is not, cleanup the resources.
        :return:
        """
        if self._terminating:
            return True

        if not self.is_alive():
            self._MPFLog.critical("Detected failure in MPFProcess {}! Terminating...".format(self._process.name))
            self.close()
            return True
        return False

    def _join_process(self):
        self._process.join(timeout=5)
        while self.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
