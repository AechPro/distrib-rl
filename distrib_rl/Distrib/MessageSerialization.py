import msgpack
import lz4.frame
from msgpack_numpy import patch
patch()

MIN_SIZE_TO_COMPRESS = 1024

LZ4 = True
NONE = False

class NullMessageCompressor(object):
    compression_type = "NONE"

    def __init__(self):
        super()

    def compress(data):
        return data

    def decompress(data):
        return data

class LZ4MessageCompressor(object):
    compression_type = "LZ4"

    def __init__(self):
        super()

    def compress(data):
        return lz4.frame.compress(data)

    def decompress(data):
        return lz4.frame.decompress(data)


class MessageSerializer(object):
    _compressors = {
        NullMessageCompressor.compression_type: NullMessageCompressor,
        LZ4MessageCompressor.compression_type: LZ4MessageCompressor
    }

    def __init__(self, compression_type=LZ4MessageCompressor.compression_type, min_size_to_compress = MIN_SIZE_TO_COMPRESS):
        super()
        self._compression_type = compression_type.upper()
        self._min_size_to_compress = min_size_to_compress

    @classmethod
    def register_compressor(compressor):
        MessageSerializer._compressors[compressor.compression_type.upper()] = compressor

    def pack(self, data):
        data = msgpack.packb(data)
        compression_type = "NONE"
        if self._compression_type and self._min_size_to_compress <= len(data):
            compression_type = self._compression_type
            compressor = MessageSerializer._compressors[self._compression_type]
            data = compressor.compress(data)
        return msgpack.packb((compression_type, data))

    def unpack(self, data):
        compression_type, data = msgpack.unpackb(data)
        try:
            data = MessageSerializer._compressors[compression_type].decompress(data)
            return msgpack.unpackb(data)
        except KeyError:
            raise ValueError(f"Received message with unknown compression type '{compression_type}'."
                             f" Supported types are {','.join(ct for ct in MessageSerializer._compressors.keys)}")

