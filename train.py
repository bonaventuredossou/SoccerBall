import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo_custom.cfg",
           "load": "bin/yolo.weights",
           "batch": 8,
           "epoch": 100,
           "gpu": 1.0,
           "train": True,
           "annotation": "./annotations/",
           "dataset": "./images/"}

tfnet = TFNet(options)
tfnet.train()
# this line of code lets you save the built graph to a protobuf file (.pb)
# this step is unnecessary for this notebook
tfnet.savepb()
