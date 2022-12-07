import os
import cv2
import time
import argparse
import pandas as pd

from detector import DetectorTF2
from matplotlib import pyplot as plt

def DetectImagesFromFolder(detector, images_dir, save_output=False, output_dir='output/'):
  df_list= list()
  for file in os.scandir(images_dir):
      if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png')) :
        image_path = os.path.join(images_dir, file.name)
        img = cv2.imread(image_path)
        det_boxes = detector.DetectFromImage(img,file.name)
        df_list.extend(det_boxes)
        img = detector.DisplayDetections(img, det_boxes)
        
        # plt.show(img)
        # cv2.imshow('TF2 Detection', img)
        # cv2.waitKey(0)
        if save_output:
          img_out = os.path.join(output_dir, file.name)
          cv2.imwrite(img_out, img)
  df = pd.DataFrame(df_list, columns =['image_name', 'xmin','xmax','ymin','ymax', 'class_id', 'score'])
  df.to_csv(os.path.join(output_dir, 'output.csv'), encoding='utf-8', index=False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Detect bboxes for a given set of images.")
  parser.add_argument("--threshold", type=float, help="confidence threshold for being a valid detected object.", default=0.5)
  parser.add_argument("--path_to_model", type=str, help="full path to the trained ssd model.", default="../ssd_model")
  parser.add_argument("--path_to_label_map", type=str, help="full path to label_map file provided for ssd pipline configuration.",default="./model_info/label_map.pbtxt")
  parser.add_argument("--path_to_images", type=str, help="full path to images being feeded to the ssd model.", default="../test_images")
  parser.add_argument("--path_to_save_images", type=str, help="full path for exporting predictions.",default="../test_predictions")
  args = parser.parse_args()
  threshold = args.threshold
  model_path = args.path_to_model
  path_to_labelmap = args.path_to_label_map
  images_dir = args.path_to_images
  output_directory = args.path_to_save_images
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
  detector = DetectorTF2(model_path, path_to_labelmap, class_id=[1], threshold=threshold)
  DetectImagesFromFolder(detector, images_dir, save_output=True, output_dir=output_directory)