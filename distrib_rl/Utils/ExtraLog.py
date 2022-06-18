import numpy as np
from distrib_rl.Distrib import RedisClient, RedisKeys

from datetime import datetime
import csv
from pathlib import Path
from time import sleep

class ExtraLogger():
    def __init__(self, cfg):

        # How many times aggregate_data will be called before combining and sending values
        self.report_every = cfg.get("rounds_per_aggregate", 60)

        values_cfg = cfg["values"]
        self.keys = values_cfg.keys()

        self.agg_funcs = {}
        self.per_funcs = {}
        self.per_vals = {}
        self.out_names = {}

        for key, k_cfg in values_cfg.items():
            agg = k_cfg.get("agg", None)
            if agg is None or agg == "mean":
                self.agg_funcs[key] = np.mean
                self.out_names[key] = ""
            elif agg == "mean_per":
                self.agg_funcs[key] = np.sum
                self.out_names[key] = ""
            elif agg == "std":
                self.agg_funcs[key] = np.std
                self.out_names[key] = "std_"
            elif agg == "rms":
                self.agg_funcs[key] = _root_mean_square
                self.out_names[key] = "rms_"

            per_value = None
            if "per" in k_cfg:
                self.per_funcs[key] = _per
                per_value = k_cfg["per"]
                per_name = f"{key}_per_{per_value}"
            elif "inv_per" in k_cfg:
                self.per_funcs[key] = _inv_per
                per_value = k_cfg["inv_per"]
                per_name = f"{per_value}_per_{key}"

            if per_value is None or per_value == "instance":
                self.per_funcs[key] = None
                self.out_names[key] += key
            else:
                self.per_vals[key] = per_value
                self.out_names[key] += per_name

        self.reset()

        if cfg.get("log_to_csv", False):
            self.csv_path = None
            self.csv = True

            if "csv_path" in cfg:
                self.csv_dir = Path(cfg["csv_path"])
                self.csv_dir.mkdir(exist_ok=True)
            else:
                self.csv_dir = Path(".")
        else:
            self.csv = False
            
        if cfg.get("wandb_via_redis", False):
            self.redis_client = RedisClient()
            self.redis_client.connect()
        else:
            self.redis_client = None


    def reset(self):
        self.running_data = { k: [] for k in self.keys }
        self.running_per_vals = { k: 0 for k in set(self.per_vals.values())}
        self.last_report = 1


    def log(self, key, data):
        if key in self.keys:
            self.running_data[key] += [ float(data) ]

    def log_multi(self, key_data_dict):
        log_keys = key_data_dict.keys()
        for key in set(self.keys) & log_keys:
            self.running_data[key] += [ float(key_data_dict[key]) ]


    def aggregate_data(self, **kwargs):
        """
        Keywords expected to contain values to aggregate values against.  
        i.e. aggregate_data(steps=1000) for data with config {"per": "step"}
        """

        for k in kwargs.keys():
            if k in self.running_per_vals:
                self.running_per_vals[k] += kwargs[k]
            else:
                # Debug, take out eventually
                print(k, " sent in to logger and not used")

        if self.last_report < self.report_every:
            self.last_report += 1
            return None

        agg_data = {}
        for key in self.keys:
            out_name = self.out_names[key]
            data = self.running_data[key]
            if len(data) == 0:
                agg_data[out_name] = 0
            else:
                x = self.agg_funcs[key](data)
                if self.per_funcs[key] is not None:
                    per_value = self.per_vals[key]
                    x = self.per_funcs[key]( x, self.running_per_vals[per_value] )
                agg_data[out_name] = x

        if self.redis_client is not None:
            self.redis_client.push_data(RedisKeys.EXTRA_LOG_AGGREGATE_KEY, agg_data)

        if self.csv:
            self._write_to_csv(agg_data)
            
        self.reset()
        
        return agg_data


    def pop_redis_mean_aggregates(self):
        if self.redis_client is None:
            return None

        results = self.redis_client.atomic_pop_all(RedisKeys.EXTRA_LOG_AGGREGATE_KEY)
        if len(results) == 0:
            return None
        
        combined_agg = {}
        for res in results:
            for k,v in res.items():
                if k not in combined_agg:
                    combined_agg[k]  = [v]
                else:
                    combined_agg[k] += [v]

        for k in combined_agg.keys():
            combined_agg[k] = np.mean(combined_agg[k])
        
        return combined_agg
    

    def _start_csv(self):
        # Kept running into multiple processes grabbing the same files, even to the microsecond!  Mix it up.
        sleep(np.random.uniform(0,1))
        self.csv_path = self.csv_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f.csv") 
        
        with open(self.csv_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)  
            row = ["datetime"] + [ self.out_names[k] for k in self.keys ]
            writer.writerow(row)

        ## Apparently closing files is a big performance penalty in windows, lets try and leave it open (even if not ideal)
        ## Update: This is mitigated now by only updating every N updates
        ## https://gregoryszorc.com/blog/2015/10/22/append-i/o-performance-on-windows/
        # try:
        #     del self.writer
        #     self.cur_file.close()
        # except:
        #     pass
        
        # self.csv_file = open(self.csv_path, "a", newline='') 
        # self.writer = csv.writer(self.csv_file)   
        # row = ["datetime"] + [ self.out_names[k] for k in self.keys ]
        # self.writer.writerow(row)

    def _write_to_csv(self, agg_data):
        if (self.csv_path is None) or (not self.csv_path.exists()):
            self._start_csv()

        row = [ datetime.now() ] +  [ agg_data[self.out_names[k]] for k in self.keys ]

        with open(self.csv_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)   
            writer.writerow(row)

        # self.writer.writerow(row)


def _per(x, per_value):
    return x / per_value

def _inv_per(x, per_value):
    return per_value / x

def _root_mean_square(x):
    return np.sqrt(np.mean(np.square(x)))