import torch.nn as nn

def parse_function(function_name):
    if function_name is None:
        return None, False

    f = function_name.lower().strip()
    needs_features = False
    function = None

    if f == 'relu':
        function = nn.ReLU

    elif f == 'tanh':
        function = nn.Tanh

    elif f == 'prelu':
        function = nn.PReLU

    elif f == 'softmax' or f == 'soft max':
        function = nn.Softmax

    elif f in ('sigmoid','logit','logits'):
        function = nn.Sigmoid

    elif f in ("clamped", "clamped_linear", "clamp"):
        from distrib_rl.Utils.Torch.TorchFunctions import ClampedLinear
        function = ClampedLinear

    elif f == 'flatten':
        from distrib_rl.Utils.Torch.TorchFunctions import Flatten
        function = Flatten

    elif f in ('bn','batch_norm','batchnorm','batch norm'):
        function = nn.BatchNorm1d
        needs_features = True

    return function, needs_features

def parse_layer_type(layer_info):
    if type(layer_info) not in (str,):
        layer_info = "{}".format(layer_info)

    lf = layer_info.lower().strip()
    layer = None
    out_features = None

    if "lstm" in lf:
        raise NotImplementedError
        # distrib_rl.from Policies.PyTorch.Recurrent import LSTMModule
        #
        # out_features = int(lf[lf.find("lstm")+len("lstm"):])
        # layer = LSTMModule

    else:
        out_features = int(lf)
        layer = nn.Linear

    return layer, int(out_features)
