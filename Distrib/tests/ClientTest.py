from Distrib import RedisClient
import numpy as np
import time

DATA_SIZE = 10000
NUM_PUSHES = 1000
PUSH_DELAY = 0

def run_test():
    client = RedisClient()
    client.connect()

    for i in range(NUM_PUSHES):
        client.push_data(np.random.randn(DATA_SIZE))

        if PUSH_DELAY > 0:
            time.sleep(PUSH_DELAY)

    client.disconnect()

if __name__ == "__main__":
    run_test()