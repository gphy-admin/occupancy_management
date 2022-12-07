#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:16:12 2021

@author: ahmad
"""

import os
import argparse

import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="export frozen ssd object detection model into tflite")
  parser.add_argument("--saved_model_directory", type=str, help="full path to the trained ssd model.", default="../ssd_model")
  args = parser.parse_args()
  saved_model_directory = args.saved_model_directory
  
  # Convert the model
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_directory) # path to the SavedModel directory
  tflite_model = converter.convert()
  
  # Save the model.
  with open(os.path.join(saved_model_directory, 'saved_model.tflite'), 'wb') as f:
    f.write(tflite_model)