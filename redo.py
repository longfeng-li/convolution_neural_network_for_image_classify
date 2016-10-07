import tensorflow as tf
from data_utils import load_CIFAR100
from sys import argv
from conv_net import ConvLayer 
import os
import time
import numpy as np
script, dataset = argv

print time.strftime('%H-%M-%S')
path = dataset + "/cifar-100-python"
Xtr, Ytr, Xte, Yte = load_CIFAR100(path)
mean_image = np.mean(Xtr, axis=0)
Xtr -= mean_image
Xte -= mean_image
exe = ConvLayer(3072, 5, 20)
exe.conv_net(Xtr, Ytr, Xte, Yte)
print time.strftime('%H-%M-%S')
