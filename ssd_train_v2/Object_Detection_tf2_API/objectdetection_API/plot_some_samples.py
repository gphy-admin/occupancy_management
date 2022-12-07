#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:02:45 2021

@author: ahmad
"""



from PIL import Image
import pandas as pd
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
csv_path = './ssd_data/train_data.csv'
examples = pd.read_csv(csv_path)
examples = examples.rename(columns={'image_name': 'filename'})
examples['class'] = 'human'
examples['width'] = 80
examples['height'] = 80

image_filename = 'acqui3_imag_10179.png'
image_filename = 'acqui3_imag_3723.png'
image_filename = 'acqui2_image_1509.png'
img = Image.open(os.path.join('./ssd_data/images', image_filename))
img = np.array(img)
img = np.stack((img,)*3, axis=-1)
temp = examples.loc[:][examples['filename']==image_filename]
for _, row in temp.iterrows():
    x_min = row['xmin']
    x_max = row['xmax']
    y_min = row['ymin']
    y_max = row['ymax']
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)


Img = Image.fromarray(img)
Img.save(image_filename)

# plt.figur&w(img)