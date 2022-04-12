from PolicyOptimization.DistribPolicyGradients import Client
import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
print("SET TORCH TO 1 THREAD IN RUN_CLIENT")
torch.set_num_threads(1)


def main():
    client = Client()
    client.configure()

    try:
        client.train()
    except:
        import traceback
        print("EXCEPTION OCCURRED IN CLIENT!")
        print(traceback.format_exc())
    finally:
        try:
            client.cleanup()
        except:
            import traceback
            print("!!UNABLE TO CLEANUP CLIENT, SHUTDOWN FAILED!!")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
