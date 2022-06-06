import numpy as np
import torch.nn as nn
from distrib_rl.Utils.Torch.TorchFunctions import *
#from distrib_rl.Policies.PyTorch.Recurrent import LSTMModule

def build_from_json(model_json, input_shape, output_shape, channels_first=True):
    #This assumes that the input shape is (seq, features) for recurrent/ff architectures and
    #(channels, width, height) for cnn.

    type = model_json["type"][0].lower().strip()
    in_features = None
    if type in ("ff", "ffnn", "mlp", "feed_forward", "feedforward", "feed forward", "fully_connected", "fullyconnected", "fully connected"):
        in_features = np.prod([input_shape])

    elif type in ("recurrent", "rec", "rnn", "lstm", "gru"):
        # in_features = input_shape[1]
        in_features = np.prod([input_shape])

    elif type in ("cnn", "conv", "convolutional", "convolution", "conv2d"):
        in_features = input_shape[0] #channels first

    conv_shapes = prepare_conv_shapes(model_json, input_shape, output_shape)
    model_objects = []
    for layer_name, layer_json in model_json["layers"].items():
        layer_objects, in_features = parse_and_build_layer(layer_json, in_features, output_shape, conv_shapes)
        model_objects += layer_objects

    model = nn.Sequential(*model_objects)
    return model


def parse_and_build_layer(layer_json, in_features, output_shape, conv_shapes):
    layer_type = layer_json["type"].lower().strip()
    if layer_type is not None:
        layer_type = layer_type.lower().strip()

    layer_objects = []
    layer_object = None
    activation_object = None
    extra_object = None

    if layer_type in ("out", "output"):
        output_features = np.prod((output_shape,))
    elif layer_type in ("diag_out", "diagonal_output", "diagonal_out", "gaussian_out", "diagonal_gaussian",):
        output_features = np.prod((output_shape,))*2
    else:
        output_features = layer_json["num_nodes"]

    if type(layer_json["extra"]) in (tuple, list):
        for extra_name in layer_json["extra"]:
            if extra_name in ("flatten", "flat"):
                in_features = np.prod(conv_shapes[-1])

            extra_object = parse_and_build_extra(extra_name, in_features)
            if extra_object is not None:
                layer_objects.append(extra_object)
    else:
        extra_name = layer_json["extra"]
        if extra_name in ("flatten", "flat"):
            in_features = np.prod(conv_shapes[-1])

        extra_object = parse_and_build_extra(extra_name, in_features)
        if extra_object is not None:
            layer_objects.append(extra_object)

    if layer_type in ("lstm",):
        raise NotImplementedError
        # layer_object = LSTMModule(in_features=in_features,
        #                           out_features=output_features)

    elif layer_type in ("ff", "feedforward", "feed_forward","fc", "fullyconnected", "fully connected", "fully_connected",
                        "out", "output", "diag_out", "diagonal_output", "diagonal_out", "gaussian_out", "diagonal_gaussian"):
        layer_object = nn.Linear(in_features=in_features,
                                 out_features=output_features)

    elif layer_type in ("conv", "cnn", "conv2d", "convlution", "convolution2d"):
        layer_object = nn.Conv2d(in_channels=in_features,
                                 out_channels=output_features,
                                 kernel_size=layer_json["kernel"],
                                 stride=layer_json["stride"],
                                 padding=layer_json["padding"])

    activation_object = parse_and_build_activation(layer_json)

    if layer_object is not None:
        layer_objects.append(layer_object)
    if activation_object is not None:
        layer_objects.append(activation_object)
    return layer_objects, output_features

def prepare_conv_shapes(model_json, input_shape, output_shape):
    if type(input_shape) not in (tuple, list):
        return None
    elif len(input_shape) < 3:
        return None

    layer_shapes = []
    shape_2d = (input_shape[1], input_shape[2])

    layer_shapes.append(shape_2d)

    for layer_name, layer_json in model_json["layers"].items():
        layer_type = layer_json["type"].lower().strip()
        if layer_type not in ("conv", "cnn", "conv2d", "convlution", "convolution2d"):
            continue

        kernel_size = layer_json["kernel"]
        stride = layer_json["stride"]
        padding = layer_json["padding"]

        if type(kernel_size) not in (tuple, list):
            kernel_size = [kernel_size, kernel_size]
        if type(stride) not in (tuple, list):
            stride = [stride, stride]
        if type(padding) not in (tuple, list):
            padding = [padding, padding]

        height_out = ((shape_2d[1] + 2 * padding[0] - (kernel_size[0])) // stride[0]) + 1
        width_out = ((shape_2d[0] + 2 * padding[1] - (kernel_size[1])) // stride[1]) + 1

        old = (shape_2d[0], shape_2d[1])
        shape_2d = (width_out, height_out)
        layer_shapes.append((layer_json["num_nodes"], width_out, height_out))

    return layer_shapes

def parse_and_build_extra(extra_name, in_features):
    t = extra_name
    if t is not None and type(t) == str:
        t = t.lower().strip()

    obj = None
    if t in ("batch_norm1d", "batch norm 1d", "batchnorm1d", "bn", "bn1d"):
        obj = nn.BatchNorm1d(num_features=in_features)

    elif t in ("batch_norm2d", "batch norm 2d", "batchnorm2d", "bn2d"):
        obj = nn.BatchNorm2d(num_features=in_features)

    elif t in ("flat", "flatten"):
        obj = Flatten()

    elif t in ("layernorm", "ln","layer_norm","layer norm"):
        obj = nn.LayerNorm(normalized_shape=in_features)

    return obj

def parse_and_build_activation(layer_json):
    t = layer_json["activation_function"].lower().strip()
    if t is not None and type(t) == str:
        t = t.lower().strip()

    function = None
    if t in ('relu',):
        function = nn.ReLU()

    elif t in ('tanh',):
        function = nn.Tanh()

    elif t in ('continuous_map', "cont_map", "map_cont", "mapped_continuous"):
        function = MapContinuousToAction()

    elif t in ("softmax", "soft max"):
        function = nn.Softmax(dim=-1)

    elif t in ('sigmoid', 'logit', 'logits'):
        function = nn.Sigmoid()

    elif t in ("clamped", "clamped_linear", "clamp"):
        function = ClampedLinear()

    elif t in ("selu",):
        function = nn.SELU()

    return function
