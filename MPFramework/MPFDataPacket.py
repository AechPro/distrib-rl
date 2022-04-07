"""
    File name: MPFDataPacket.py
    Author: Matthew Allen

    Description:
        This is a simple data packet object to be used when transmitting data to an MPFProcess object.
        It should consist of a header to be used for data identification, and the raw data.
"""

class MPFDataPacket(object):
    def __init__(self, header, data):
        self._header = header
        self._data = data
        del header
        del data

    def __call__(self):
        return self._header, self._data

    def cleanup(self):
        del self._header
        del self._data