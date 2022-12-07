import tensorflow as tf
import os
import cv2
from pathlib import Path
import numpy as np
import copy
from PIL import Image
from matplotlib import pyplot as plt
from keras import backend as K 
import pandas as pd
K.clear_session()
img_path_dir='./test_images'
interpreter = tf.lite.Interpreter(model_path='./saved_model.tflite')
interpreter.allocate_tensors()
_,input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
print(input_height, input_width)

input_details   = interpreter.get_input_details()
output_details  = interpreter.get_output_details()
output_names = [output_detail['name'] for output_detail in output_details]
tensor_details = interpreter.get_tensor_details()
# if details are needed just print above mentioned 


'''
In our model the tfl1.reshape7 is the input to the concatination that has anchors before preprocessing In this code I 
finding the index of those anchors 
tfl.logstics has classes of those anchors before processing again getting index of that
please not you need to view the tflite model to find these names it changes as per the model structuee 


'''
index_anchors= None
index_classes= None
for index, tensor in enumerate(tensor_details):
    if len(tensor['name'])!=0:
        print(Path(tensor['name']).parts[0])
        if Path(tensor['name']).parts[0]== 'tfl.reshape7':index_anchors = index
        elif Path(tensor['name']).parts[0]== 'tfl.logistic': index_classes= index
            
            
print(index_anchors,index_classes )




import os
arr = os.listdir(img_path_dir)
'''
taking the list of all files in test_image folder 
'''


'''
funcrtion for loading the image 
'''

def preprocess(resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return  resized_inputs/128 - 1.0


def construct_input_tensor(img_path, input_details):
    'load an image and rehsape according to the model input layer'
    img_in = cv2.imread(img_path,0)
    

    img_in = cv2.resize(img_in,(80,80))
    img = preprocess(img_in)
    input_array = np.reshape(img, input_details[0]['shape'])
    image = input_array.astype(np.float32)
    return image
#function for extracting box details
def ExtractBBoxes(bboxes, num_det, bscores, im_width, im_height):

    bbox = []
    for idx in range(num_det):
        y_min = int(bboxes[idx][0] * im_height)
        x_min = int(bboxes[idx][1] * im_width)
        y_max = int(bboxes[idx][2] * im_height)
        x_max = int(bboxes[idx][3] * im_width)
        class_label = 'person'
        bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])

    return bbox
# for printing final image with bounding boxes
def DisplayDetections(image, boxes_list, img_name,final_list, final_list_count_per_image,  num_det):
    final_list_count_per_image.append([img_name,  num_det, len(boxes_list)])
    if len(boxes_list)== 0: return image, final_list, final_list_count_per_image
    else:
        img = image.copy()
        for idx in range(len(boxes_list)):
            x_min = boxes_list[idx][0]
            y_min = boxes_list[idx][1]
            x_max = boxes_list[idx][2]
            y_max = boxes_list[idx][3]
            final_list.append([img_name, x_min, x_max, y_min, y_max, boxes_list[idx][5]])
            
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    return img, final_list, final_list_count_per_image

# opens the image and preprocess it and loads in tflite model and then resutls are saved in test_images folder
final_list=[]
final_list_count_per_image=[]
for i in arr:

    img_path= img_path_dir + '/' + i
    input_array= construct_input_tensor(img_path, input_details)
    interpreter.set_tensor(input_details[0]['index'], input_array)
    interpreter.invoke()
    num_det = int(interpreter.get_tensor(output_details[3]['index']))
    boxes = interpreter.get_tensor(output_details[0]['index'])[0][:num_det]
    classes = interpreter.get_tensor(output_details[1]['index'])[0][:num_det]
    scores = interpreter.get_tensor(output_details[2]['index'])[0][:num_det]
    
    bbox= ExtractBBoxes(boxes, num_det, scores, 80, 80)
    img_in = cv2.imread(img_path,0)
    out_img,  final_list, final_list_count_per_image= DisplayDetections(img_in, bbox,i, final_list, final_list_count_per_image, num_det)
    #I was using this file to show discrpencies between tflite and Nntool output hence we have final_list_count_per_image this we don't need for testing the tflite just we need output csv

    cv2.imwrite('./test_predictions/' + i, out_img )

df = pd.DataFrame(final_list, columns =['image_name', 'xmin','xmax','ymin','ymax', 'score']) 
df.to_csv('./annotation_results_tflite.csv', encoding='utf-8', index=False)
