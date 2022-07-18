import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
print("SET TORCH TO 1 THREAD IN RUN_CLIENT")
torch.set_num_threads(1)

from distrib_rl.PolicyOptimization.DistribPolicyGradients import Client

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
