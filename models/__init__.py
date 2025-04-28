from .backbones.alexnet import *
from .backbones.efficientnet import *
from .backbones.googlenet import *
from .backbones.internimage import *
from .backbones.resnet import *
from .backbones.vgg import *
from .heads.classification_heads import *
from .registry import BACKBONE, CLASSIFIER, model_dict
import torch.nn as nn


def build_model(key, multi_scale):
    print(key)
    if key not in BACKBONE:
        raise Exception("Not implemented model architecture!")
    model = BACKBONE[key]()
    output_d = model_dict[key]
    return model, output_d


def build_classifier(key):
    if key not in CLASSIFIER:
        raise Exception("Not implemented classifier!")
    return CLASSIFIER[key]


class GenericModel(nn.Module):
    def __init__(self, backbone, classifier, bn_wd=True, multi_scale=False):
        super().__init__()
        
        self.backbone = backbone
        self.classifier = classifier
        self.bn_wd = bn_wd
        
        
    def fresh_params(self):
        return self.classifier.fresh_params(self.bn_wd)
    
        
    def finetune_params(self):
        if self.bn_wd:
            return self.backbone.parameters()
        else:
            return self.backbone.named_parameters()
    
    def forward(self, x):
        feat_map = self.backbone(x)
        logits, feat = self.classifier(feat_map)
        return logits, feat