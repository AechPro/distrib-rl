import numpy as np

# The following two functions come from:
# https://stackoverflow.com/questions/55311399/fastest-way-to-store-a-numpy-array-in-redis
def decode_numpy(serialized_arr: str) -> np.array:
    if serialized_arr is None:
        return None

    sep = '|'.encode('utf-8')
    i_0 = serialized_arr.find(sep)
    i_1 = serialized_arr.find(sep, i_0 + 1)
    arr_dtype = serialized_arr[:i_0].decode('utf-8')
    arr_shape = tuple([int(a) for a in serialized_arr[i_0 + 1:i_1].decode('utf-8').split(',')])
    arr_str = serialized_arr[i_1 + 1:]
    arr = np.frombuffer(arr_str, dtype = arr_dtype).reshape(arr_shape)
    return arr.tolist()

def encode_numpy(arr: np.array) -> bytes:
    arr_dtype = bytearray(str(arr.dtype), 'utf-8')
    arr_shape = bytearray(','.join([str(a) for a in arr.shape]), 'utf-8')
    sep = bytearray('|', 'utf-8')
    arr_bytes = arr.ravel().tobytes()
    to_return = arr_dtype + sep + arr_shape + sep + arr_bytes
    return bytes(to_return)


def atomic_pop_all(redis, key):
    pipe = redis.pipeline()
    pipe.lrange(key, 0, -1)
    pipe.delete(key)
    return pipe.execute()[0]
