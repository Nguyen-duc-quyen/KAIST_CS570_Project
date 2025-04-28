from utils.registry import *

# Backbones registry
BACKBONE = Registry()

# Define the number of output features of each backbone
model_dict = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
    "resnext50_32x4d": 2048,
    "resnext101_32x8d": 2048,
    "wide_resnet50_2": 2048,
    "wide_resnet101_2": 2048,
    "alexnet": 4096,
    "efficientnet_b0": 1280,
    "efficientnet_b1": 1280,
    "efficientnet_b2": 1408,
    "efficientnet_b3": 1536,
    "efficientnet_b4": 1792,
    "efficientnet_b5": 2048,
    "efficientnet_b6": 2304,
    "efficientnet_b7": 2560,
    "efficientnet_v2_s": 1280,
    "efficientnet_v2_m": 1280,
    "efficientnet_v2_l": 1280,
    "vgg11": 25088,
    "vgg13": 25088,
    "vgg16": 25088,
    "vgg19": 25088,
    "vgg11_bn": 25088,
    "vgg13_bn": 25088,
    "vgg16_bn": 25088,
    "vgg19_bn": 25088,
    "googlenet": None,
    "inception_v2": None,
    "inception_v3": None,
}

# Classifiers registry
CLASSIFIER = Registry()