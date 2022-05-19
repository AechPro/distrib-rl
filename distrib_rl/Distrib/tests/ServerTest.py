from distrib_rl.Distrib import RedisServer


NUM_PULLS = 1000

def run_test():
    server = RedisServer()
    server.connect(clear_existing=True)

    data = server.collect(NUM_PULLS)
    print("Collected {} returns from redis".format(len(data)))

    server.disconnect()

if __name__ == "__main__":
    run_test()