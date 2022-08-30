"""
    File name: process_handler.py
    Author: Matthew Allen

    Description:
        This is the container for an Process object. It handles spawning the process and interfacing with it
        from the main process.
"""
import logging
import multiprocessing as mp
import time
import traceback
from queue import Empty
import psutil

from distrib_rl.mpframework import DataPacket, TaskChecker

try:
    # We want true asynchronicity with as little task switching as possible so we want to spawn processes with independent
    # Python interpreters. However, the start method may have already been set before we run this code so we'll catch
    # that exception here and ignore it.
    mp.set_start_method("spawn")

except RuntimeError:
    pass


class ProcessHandler(object):
    def __init__(
        self, input_queue=None, input_queue_max_size=1000, output_queue_max_size=1000
    ):
        self._output_queue = mp.Queue(maxsize=output_queue_max_size)
        self._process = None
        self._logger = logging.getLogger("MPFLogger")
        self._terminating = False

        if input_queue is None:
            self._input_queue = mp.Queue(maxsize=input_queue_max_size)
        else:
            self._input_queue = input_queue

    def setup_process(self, process, shared_memory=None, cpu_num=None):
        """
        This function spawns a process on a desired CPU core if one is provided. It expects an instance of
        Process to be passed as the process argument.

        :param process: An instance of an Process object.
        :param cpu_num: The CPU core to spawn the process on. Only available on Linux.
        :return: None.
        """
        self._logger.debug("Setting up a new Process...")
        self._process = process

        # Setup process i/o and start it.
        process._inp = self._input_queue
        process._out = self._output_queue
        process.set_shared_memory(shared_memory)
        process.start()

        # If a specific cpu core is requested, move the process to that core.
        if cpu_num is not None:
            self._logger.debug(
                "Moving Process {} to CPU core {}.".format(process.name, cpu_num)
            )
            if type(cpu_num) not in (list, tuple):
                cpu_num = [cpu_num]
            psutil.Process(process.pid).cpu_affinity(cpu_num)

        self._logger.debug("Process {} has started!".format(process.name))

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
            # Construct a data packet and put it on the process input queue.
            task = DataPacket(header, data)
            self._input_queue.put(task, block=block, timeout=timeout)
            del task
        else:
            self._logger.debug(
                "Process {} input queue is full!".format(self._process.name)
            )

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

        # First, check if the queue is empty and return if it is.
        if self._output_queue.empty():
            return None

        if not self._output_queue.empty():
            result = self._output_queue.get(block=block, timeout=timeout)
            return result()

        return None

    def get_all(
        self, block=False, timeout=None, failure_sleep_time=0.1, cleaning_up=False
    ):
        """
        Function to get every item currently available on the output queue from our process. The implementation of
        this function looks a bit odd, but it has been my experience that simply checking if a queue is empty almost never
        results in an accurate measurement of how many items are actually available on the queue. This function
        tries its best to guarantee that all items will be taken off the queue when it is called.
        `
        :param block: mp.Queue block argument.
        :param timeout: mp.Queue timeout argument.
        :param failure_sleep_time: time between sleep cycles when a failure occurs. Leave alone unless necessary.
        :param cleaning_up: boolean to indicate that cleanup is happening. Do not modify.
        :return: List of items obtained from our process.
        """

        if self._check_status() and not cleaning_up:
            return None

        # First, check if the queue is empty and return if it is.
        if self._output_queue.empty():
            return None

        failCount = 0

        results = []
        try:
            # Here we take items off the queue for as long as the qsize function says we can.
            # TODO: fixme
            # while self._output_queue.qsize() > 0:
            while not self._output_queue.empty():
                try:
                    result = self._output_queue.get(block=block, timeout=timeout)
                    # print("GOT RESULT",self._output_queue.qsize(), self._output_queue.empty())

                    header, data = result()
                    results.append((header, data))
                    result.cleanup()

                    del result
                    failCount = 0
                except Empty:
                    failCount += 1
                    if failure_sleep_time is not None and failure_sleep_time > 0:
                        time.sleep(failure_sleep_time)

                    # It appears to be the case that the empty flag in the queue object
                    # is not related to the qsize() function, so an empty queue exception can
                    # be thrown even when the queue is not actually empty.

                    # This code can infinitely loop for some reason. qsize() can return a valid integer,
                    # while empty() can return true for an unlimited period of time, causing get to fail indefinitely.
                    # Whatever data remains in the queue simply cannot be
                    # retrieved at this point, and it is left in memory until the Python interpreter is closed.
                    if failCount >= 10:
                        if (
                            # TODO: fixme
                            # self._output_queue.qsize() == 0
                            # and self._output_queue.empty()
                            self._output_queue.empty()
                        ):
                            break

                        self._logger.critical(
                            "GET_ALL FAILURE LIMIT REACHED ERROR!\n"
                            "FAILURE COUNT: {}\n"
                            # TODO: fixme
                            # "REMAINING QSIZE: {}\n"
                            "QUEUE EMPTY STATUS: {}".format(
                                failCount,
                                # TODO: fixme
                                # self._output_queue.qsize(),
                                self._output_queue.empty(),
                            )
                        )
                        break
                    else:
                        continue
        except Exception:
            error = traceback.format_exc()
            self._logger.critical("GET_ALL ERROR!\n{}".format(error))
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

        # Put an exit command on the input queue to our process.

        self._logger.debug(
            "Sending terminate command to process {}.".format(self._process.name)
        )
        task = DataPacket(TaskChecker.EXIT_KEYWORDS[0], self._process.name)
        self._input_queue.put(task)

        # Get any residual items from the output queue and delete them.
        self._logger.debug("Beginning residual output collection...")
        print("Calling get_all from shutdown...")
        residual_output = self.get_all(cleaning_up=True)
        if residual_output is not None:
            self._logger.debug(
                "Removed {} residual outputs from queue.".format(len(residual_output))
            )
            del residual_output

        # Note here that we do not join either the input or output queues.
        # A process handler should only have a single process, so if the user has passed
        # a joinable queue to this handler's process, they are responsible for closing it.

        self._logger.debug("Beginning process termination...")
        self._join_process()
        self._logger.debug(
            "Successfully terminated Process {}!".format(self._process.name)
        )

        # Get residual output again in case something happened between the first call to get_all() and _join_process()
        # We do this twice rather than deleting the above block because PyTorch Tensors don't release their reference
        # if .join() is called on a process before a queue containing a tensor is emptied, so we have to remove everything
        # from the queue, then call join, then remove any extraneous data from the queue again here.
        residual_output = self.get_all(cleaning_up=True)
        if residual_output is not None:
            self._logger.debug(
                "Removed {} residual outputs from queue.".format(len(residual_output))
            )
            del residual_output

        self._logger.debug("All Process objects have been terminated!")

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
            self._logger.critical(
                "Detected failure in Process {}! Terminating...".format(
                    self._process.name
                )
            )
            self.close()
            return True
        return False

    def _join_process(self):
        self._process.join(timeout=5)
        while self.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
