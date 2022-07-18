import msgpack
from lz4.frame import compress, decompress
from msgpack_numpy import patch
patch()

MIN_SIZE_TO_COMPRESS = 1024

LZ4 = True
NONE = False

def pack(data):
    data = msgpack.packb(data)
    compressed = False
    if MIN_SIZE_TO_COMPRESS <= len(data):
        compressed = True
        data = compress(data)
    return msgpack.packb((compressed, data))

def unpack(data):
    compressed, data = msgpack.unpackb(data)
    if compressed:
        data = decompress(data)
    return msgpack.unpackb(data)