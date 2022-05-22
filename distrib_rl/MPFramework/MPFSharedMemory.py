"""
    File name: MPFSharedMemory.py
    Author: Matthew Allen

    Description:
        This file contains two classes used for allocating and handling a block of shared memory which can be accessed
        asynchronously by multiple MPFProcess objects. This is purposefully not thread safe because its intended usage
        is for large (1GB+) blocks of ROM, so returning thread-safe clones of the memory held by this object
        could result in huge memory usage spikes. Safe asynchronous usage is left to the user.
"""

import ctypes
import logging
import multiprocessing as mp
from multiprocessing.managers import BaseManager

import numpy as np


class MPFSharedMemory(object):
    #Supported memory types.
    MPF_FLOAT32 = ctypes.c_float
    MPF_FLOAT64 = ctypes.c_double
    MPF_INT32 = ctypes.c_int32
    MPF_INT64 = ctypes.c_int64

    def __init__(self, size, rng=None, dtype=MPF_FLOAT32):
        self.dtype = dtype
        self.rng=rng
        self._size = size
        self._manager = None
        self._memory = None
        self._MPFLog = logging.getLogger("MPFLogger")
        self._allocate()

    def fill_memory(self, data):
        self._memory.set(0, data)

    def get_memory(self):
        return self._memory


    #Begin memory access wrappers
    def get(self, index, size):
        return self._memory.get(index, size)

    def get_random(self, size):
        return self._memory.get_random(size)

    def get_size(self):
        return self._size
    #End memory access wrappers

    def _allocate(self):
        """
        Private function for allocating memory and creating the manager for handling access to the memory.
        The manager is necessary to ensure that no clones of the memory block are ever spawned. We don't interact with it
        outside of that.
        :return: None.
        """

        self._MPFLog.debug("Allocating MPFMemoryBlock!\nSize: {}\nData type: {}.".format(self._size, self.dtype))

        #Register our shared memory block and start the manager object.
        BaseManager.register('MPFSharedMemoryBlock', MPFSharedMemoryBlock)
        self._manager = BaseManager()
        self._manager.start()

        #Build our memory object through the manager.
        self._memory = self._manager.MPFSharedMemoryBlock(self._size, self.dtype, rng=self.rng)
        self._MPFLog.debug("MPFMemoryBlock allocated successfully!")

    def cleanup(self):
        try:
            self._MPFLog.debug("Cleaning up MPFMemoryBlock...")

            self._memory.cleanup()
            self._MPFLog.debug("Shutting down MPFMemory manager...")

            self._manager.shutdown()
            self._MPFLog.debug("MPFMemoryBlock has closed successfully!")

        except:
            import traceback
            self._MPFLog.critical("MPFMemoryBlock was unable to close!"
                                  "\nException traceback: {}".format(traceback.format_exc()))
        finally:
            if self._memory is not None:
                del self._memory

            if self._manager is not None:
                del self._manager

            if self.rng is not None:
                del self.rng

            self._memory = None
            self._manager = None
            self.rng = None

class MPFSharedMemoryBlock(object):
    def __init__(self, mem_size, dtype, rng=None):
        self._dtype = self._parse_dtype(dtype)
        self._mem_size = mem_size
        self._rng = rng
        self._mem = None
        self._shared_block = None

        self._allocate()

    def set(self, start, data):
        """
        Function to write to our memory block.
        :param start: Index at which to start writing.
        :param data: Data to write.
        :return: None.
        """
        np.copyto(self._mem[start:], data)

    def get(self, index, size):
        """
        Function to read from our memory block.
        :param index: Index at which to start reading.
        :param size: Number of indices to read.
        :return: Memory from index to index+size
        """

        return self._mem[index:index + size]

    def get_random(self, size):
        if self._rng is None:
            return None

        idx = self._rng.randint(0, self._mem_size - size)
        return idx, self.get(idx, size)

    def get_size(self):
        return self._mem_size

    def _parse_dtype(self, code):
        """
        Function to parse the data type code passed as a constructor argument into an appropriate MPF data type.
        :param code: Data-type code to be checked.
        :return: The appropriate MPF data type, defaults to float32.
        """

        code = code

        float_codes = ('float', 'float32', np.float32, 'f', ctypes.c_float)
        double_codes = ('double', 'float64', np.float64, 'd', ctypes.c_double)
        int_codes = ('int', 'int32', np.int32, 'i', ctypes.c_int32)
        long_codes = ('long', 'int64', np.int64, 'l', ctypes.c_int64)

        if code in float_codes:
            return MPFSharedMemory.MPF_FLOAT32

        if code in double_codes:
            return MPFSharedMemory.MPF_FLOAT64

        if code in int_codes:
            return MPFSharedMemory.MPF_INT32

        if code in long_codes:
            return MPFSharedMemory.MPF_INT64

        raise NameError

    def _allocate(self):
        """
        Private function to allocate the shared memory.
        :return: None.
        """
        #We use a RawArray because we don't want to deal with any multiprocessing thread-safe nonsense.
        self._shared_block = mp.RawArray(self._dtype, self._mem_size)

        #Here we're loading the shared block into a numpy array so we can use it with little hassle.
        self._mem = np.frombuffer(self._shared_block, dtype=self._dtype)

    def cleanup(self):
        del self._mem
        self._mem = None

        del self._shared_block
        self._shared_block = None

        del self._rng
        self._rng = None