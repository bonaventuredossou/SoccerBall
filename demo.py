import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo_custom.cfg",
           "gpu": 0}

tfnet2 = TFNet(options)

import pprint as pp

original_img = cv2.imread("sample_img/test_image1.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
results = tfnet2.return_predict(original_img)
for result in results:
    confidence = result['confidence']
    if confidence > 0.3:
        print(result)

def boxing(original_img , predictions):
    newImage = original_img.copy()

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))
        
        if confidence > 0.3:
            cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)
        
    return newImage


cv2.imshow('', boxing(original_img, results))
