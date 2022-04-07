"""
    File name: ProcessExample.py
    Author: Matthew Allen

    Description:
        This file contains an example for how to create a MPFProcess object and interface with it from
        the main process.
"""
import time

from MPFramework import MPFProcess, MPFProcessHandler


class ExampleProcess(MPFProcess):
    def __init__(self):
        #We set loop wait period to 1 here to pretend the process is doing something intensive.
        super().__init__(process_name="example_process", loop_wait_period=1)
        self._data = None

    def init(self):
        #Wait for some initial data.
        self.task_checker.wait_for_initialization("initial_data_value")

        #Set our internal variable to whatever we received from the main process.
        self._data = self.task_checker.latest_data

        #Make sure the data is what we expected.
        assert self._data == 10
        print("Initialized process!")

    def update(self, header, data):
        #If we've been given an update from the main process, handle it.
        if 'update' in header:
            self._data = data

    def step(self):
        #This gets called repeatedly on a loop by MPFProcess.
        self._data += 10

    def publish(self):
        #Send off our data to the main process.
        self.results_publisher.publish(self._data, header="example_header")

    def cleanup(self):
        del self._data

def main():
    #Create an instance of our custom process object.
    process_instance = ExampleProcess()

    #Create a process handler.
    process_handler = MPFProcessHandler()

    #Use our process handler to setup our process.
    process_handler.setup_process(process_instance)

    #Send our process some initial data.
    process_handler.put("initial_data_value", 10)

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

            if current_data[1] >= 100:
                #Send off an update to the process.
                process_handler.put("update", -100)

            elif current_data[1] == 0:
                #Decide to be done when the data from our process matches some criteria.
                break

        #Sleep between loop cycles to prevent CPU stress.
        time.sleep(1.0)

    #All cleanup happens inside the process handler, so all we need to do is call close() and we're done!
    process_handler.close()

if __name__ == "__main__":
    main()