import numpy as np

DEBUG_WRITE_TO_CSV = True

if DEBUG_WRITE_TO_CSV:
    from datetime import datetime
    import csv

class ExtraLogClient():
    def __init__(self, cfg):
        self.keys = cfg.keys()
        self.reset()

        self.agg_funcs = {}
        self.per_funcs = {}
        self.per_vals = {}
        self.out_names = {}

        for key, k_cfg in cfg.items():
            agg = k_cfg.get("agg", None)
            if agg is None or agg == "mean":
                self.agg_funcs[key] = np.mean
                self.out_names[key] = "mean_"
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

        
        if DEBUG_WRITE_TO_CSV:
            self.csv_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
            with open(self.csv_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)   
                row =  [ self.out_names[k] for k in self.keys ]
                writer.writerow(row)

    def reset(self):
        self.running_data = { k: [] for k in self.keys }

    def log(self, key, data):
        self.running_data[key] += [ float(data) ]

    def aggregate_data(self, **kwargs):
        """
        Keywords expected to contain values to aggregate values against.  
        i.e. aggregate_data(steps=1000) for data with config {"per": "step"}
        """

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
                    x = self.per_funcs[key]( x, kwargs[per_value] )
                agg_data[out_name] = x

        if DEBUG_WRITE_TO_CSV:
            self._write_to_csv(agg_data)

        self.reset()
        
        return agg_data
    
    def _write_to_csv(self, agg_data):
        with open(self.csv_path, "a", newline='') as csvfile:
            # writer = csv.DictWriter(csvfile, fieldnames=agg_data.keys())
            # writer.writerow(agg_data)
            writer = csv.writer(csvfile)   
            row =  [ agg_data[self.out_names[k]] for k in self.keys ]
            writer.writerow(row)

def _per(x, per_value):
    return x / per_value

def _inv_per(x, per_value):
    return per_value / x

def _root_mean_square(x):
    return np.sqrt(np.mean(np.square(x)))