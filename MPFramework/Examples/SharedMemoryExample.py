"""
    File name: SharedMemoryExample.py
    Author: Matthew Allen

    Description:
        This file contains an example for using a shared memory block with a MPFProcess.
        Any number of processes can access the same block of memory used in this example.
"""

import time

import numpy as np

from MPFramework import MPFProcess, MPFProcessHandler, MPFSharedMemory


class ExampleProcess(MPFProcess):
    def __init__(self):
        #We set loop wait period to 1 here to pretend the process is doing something intensive.
        super().__init__(loop_wait_period=1)
        self._data = 0

    def init(self):
        #Pull out 5 entries from our memory.
        self._data = self.shared_memory.get(0, 5)

    def update(self, header, data):
        #We aren't going to be accepting messages from the main process in this example.
        pass

    def step(self):
        #Increment our data.
        self._data += 10

    def publish(self):
        #Send off our data to the main process.
        self.results_publisher.publish(self._data, header="example_header")

    def cleanup(self):
        del self._data


def main():
    #Create an instance of our custom process object.
    process_instance = ExampleProcess()

    size = 1000
    dtype = MPFSharedMemory.MPF_INT32 #These are just references to ctypes data types.

    #Ask MPFramework to allocate some memory for us. Here we have requested 1,000 entries of data type int32.
    memory_container = MPFSharedMemory(size, dtype=dtype)
    memory_container.fill_memory(np.ones(size, dtype=dtype))

    #Get the memory object to send off with our process.
    mem = memory_container.get_memory()

    #Create a process handler and use it to setup our process with the memory.
    process_handler = MPFProcessHandler()
    process_handler.setup_process(process_instance, shared_memory=mem)

    #Start an infinite loop in the main process.
    while True:
        #Get the latest data from our process.
        current_data = process_handler.get()

        if current_data is None:
            #If we don't have data, say so.
            print("No data available yet...")
        else:
            #If we've got some data, print it out!
            print(current_data)

            #Break out of this loop if some criteria are met.
            if current_data[1][0] >= 100:
                break

        #Sleep between loop cycles to prevent CPU stress.
        time.sleep(1.0)

    #Close the process and cleanup our memory.
    process_handler.close()
    memory_container.cleanup()

if __name__ == "__main__":
    main()