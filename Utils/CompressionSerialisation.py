import msgpack
import lz4.frame
from msgpack_numpy import patch
patch()

LZ4 = True
with_compression = False
def set_compression(compression_to_use):
    global with_compression
    with_compression = compression_to_use

def compress(data):
    return lz4.frame.compress(data)

def decompress(data):
    return lz4.frame.decompress(data)

def pack(data):
    data = msgpack.packb(data)
    if with_compression:
        data = compress(data)
    return msgpack.packb((with_compression, data))

def unpack(data):
    compressed, data = msgpack.unpackb(data)
    if compressed:
        data = decompress(data)
    return msgpack.unpackb(data)
