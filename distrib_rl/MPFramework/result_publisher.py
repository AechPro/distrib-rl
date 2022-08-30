"""
    File name: results_publisher.py
    Author: Matthew Allen

    Description:
        This is a container for the output queue used by a Process object. It handles putting data on the queue.
"""
import logging
import time

from distrib_rl.mpframework import DataPacket


class ResultPublisher(object):
    def __init__(self, output_queue, name):
        self._output_queue = output_queue
        self._name = name
        self._logger = logging.getLogger("MPFLogger")

    def publish(self, data, header="Process_default_header", block=True, timeout=1.0):
        """
        Function to publish data to our output queue.
        :param data: data to put on the queue.
        :param header: header to identify the data.
        :param block: mp.Queue block argument.
        :param timeout: mp.Queue timeout argument.
        :return: None.
        """

        if not self._output_queue.full():
            data_packet = DataPacket(header, data)
            self._output_queue.put(data_packet, block=block, timeout=timeout)
            # self._logger.debug("Process {} successfully published data!")
            del data_packet
        else:
            self._logger.debug(
                "Unable to publish results, Process {} output queue is full!".format(
                    self._name
                )
            )
            time.sleep(timeout)

        del data

    def is_empty(self):
        # TODO: fixme
        # return self._output_queue.qsize() == 0
        return self._output_queue.empty()
