"""
    File name: MPFProcess.py
    Author: Matthew Allen

    Description:
        This is the parent class for a process object. It executes the process loop and handles i/o with the main process.
        Note that process termination happens regardless of what any child objects are in the middle of, so it is
        important to implement the cleanup function properly as it is the only notification of termination that child
        objects will get.
"""

import sys
import traceback
from multiprocessing import Process


class MPFProcess(Process):
    STOP_KEYWORD = "STOP MPF PROCESS"

    def __init__(self, process_name = "unnamed_mpf_process", loop_wait_period=None):
        Process.__init__(self)
        self._out = None
        self._inp = None
        self._loop_wait_period = loop_wait_period
        self.name = process_name
        self.shared_memory = None
        self.task_checker = None
        self.results_publisher = None
        self._MPFLog = None
        self._successful_termination = False

    def run(self):
        """
        The function to be called when a process is started.
        :return: None
        """

        try:
            #We import everything necessary here to ensure that the libraries we need will be imported into the new
            #process memory instead of the main process memory.
            from distrib_rl.MPFramework import MPFResultPublisher
            from distrib_rl.MPFramework import MPFTaskChecker
            import logging
            import time

            #This are our i/o objects for interfacing with the main process.
            self.task_checker = MPFTaskChecker(self._inp, self.name)
            self.results_publisher = MPFResultPublisher(self._out, self.name)

            self._MPFLog = logging.getLogger("MPFLogger")
            self._MPFLog.debug("MPFProcess initializing...")

            #Initialize.
            self.init()
            self._MPFLog.debug("MPFProcess {} has successfully initialized".format(self.name))

            while True:
                #Here is the simple loop to be executed by this process until termination.

                #Check for new inputs from the main process.
                if self.task_checker.check_for_update():
                    self._MPFLog.debug("MPFProcess {} got update {}".format(self.name, self.task_checker.header))

                    #If we are told to stop running, do so.
                    if self.task_checker.header == MPFProcess.STOP_KEYWORD:
                        self._MPFLog.debug("MPFPROCESS {} RECEIVED STOP SIGNAL!".format(self.name))
                        self._successful_termination = True
                        raise sys.exit(0)

                    #Otherwise, update with the latest main process message.
                    self._MPFLog.debug("MPFProcess {} sending update to subclass".format(self.name))
                    self.update(self.task_checker.header, self.task_checker.latest_data)

                #Take a step.
                self.step()

                #Publish any output we might have.
                self.publish()

                #Wait if requested.
                if self._loop_wait_period is not None and self._loop_wait_period > 0:
                    time.sleep(self._loop_wait_period)

        except:
            #Catch-all because I'm lazy.
            error = traceback.format_exc()
            if not self._successful_termination:
                self._MPFLog.critical("MPFPROCESS {} HAS CRASHED!\n"
                               "EXCEPTION TRACEBACK:\n"
                               "{}".format(self.name, error))

        finally:
            self.cleanup()

            #Clean everything up and terminate.
            if self.task_checker is not None:
                self._MPFLog.debug("MPFProcess {} Cleaning task checker...".format(self.name))
                self.task_checker.cleanup()
                del self.task_checker
                self._MPFLog.debug("MPFProcess {} has cleaned its task checker!".format(self.name))

            if self.results_publisher is not None:
                del self.results_publisher

            self._MPFLog.debug("MPFProcess {} Cleaning up...".format(self.name))

            self._MPFLog.debug("MPFProcess {} Exiting!".format(self.name))
            return

    def set_shared_memory(self, memory):
        self.shared_memory = memory

    def init(self):
        raise NotImplementedError

    def update(self, header, data):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def publish(self):
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError
