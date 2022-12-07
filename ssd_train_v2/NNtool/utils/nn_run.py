""" Follwing code opens the json saved model and creates the csv for quantized and non quantized output"""

from execution.graph_executer import GraphExecuter
from execution.quantization_mode import QuantizationMode
import numpy as np
import os
import glob
import ntpath
import random
import logging
import pandas as pd
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import pandas as pd


#please set follwing fields DUMP_QUANT(you want to do quantizarion or non qauantization)
# threshold 
#dataset_files (images that need to be evaluated on), NUM_FILES (this parameter is based on how many files you want to select from test_images folder )
# quantization_result_path(the path where quantization images are stored, quantized csv output), if not created the code creates it
# non_quantization_result_path(the path where non quantization images are stored, non quantized csv output), if not created the code creates it 

DUMP_QUANT = True
# Please make sure that you use a lower or equal threshold to the one used for evaluation 
# (in basic_analysis.py inside the evaluation folder)
threshold = 0.75
# This is not needed anymore since we use all the test set
#NUM_FILES = 620 
# Set to true to save files with bounding boxes
CREATE_DETECTION_FILES=False
# Use images inside the training folder:
dataset_files = glob.glob("../Object_Detection_tf2_API/objectdetection_API/test_images/*")
quantization_result_path = "quantized_output"
non_quantization_result_path ="non_quantized_output"

class_names = ("null","person")

if not os.path.exists(quantization_result_path):
	os.makedirs(quantization_result_path)

if not os.path.exists(non_quantization_result_path):
	os.makedirs(non_quantization_result_path)

# subsample the whole dataset
#random.seed(32)
#subsample_files = random.sample(dataset_files, NUM_FILES)
LOG = logging.getLogger('nntool.'+__name__)
executer = GraphExecuter(G, qrecs=G.quantization)
def ExtractBBoxes(bboxes, bclasses, bscores, im_width, im_height, threshold,image_name):
	bbox = []
	csv_row =[]
	#print(bscores)
	for idx in range(len(bboxes)):
		if bclasses[idx] == 1:
			if bscores[idx] >= threshold:
				
				y_min = int(bboxes[idx][0] * im_height)
				x_min = int(bboxes[idx][1] * im_width)
				y_max = int(bboxes[idx][2] * im_height)
				x_max = int(bboxes[idx][3] * im_width)
				class_label = 'person'
				bbox.append([x_min, y_min, x_max, y_max, class_label, float(bscores[idx])])
				csv_row.append([image_name,x_min, x_max, y_min, y_max, class_label, float(bscores[idx])])
	return csv_row, bbox

def DisplayDetections(image, boxes_list, image_name):

	img = image.copy()
	for idx in range(len(boxes_list)):
		
		x_min = boxes_list[idx][0]
		y_min = boxes_list[idx][1]
		x_max = boxes_list[idx][2]
		y_max = boxes_list[idx][3]
		cls =  str(boxes_list[idx][4])
		score = str(np.round(boxes_list[idx][-1], 2))
		#print((y_max-y_min)*(x_max-x_min))

		text = cls + ": " + score
		#print(x_min, x_max, y_min, y_max)

		cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)

	return img

def preprocess(image):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      image: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return  (image/128) - 1.0


df_list= list()
if DUMP_QUANT:
	print("Running Quantized NN")
else:
	print("Running Float NN")
iter_n=1
for j, file in enumerate(dataset_files):
	#LOG.info("{}/{}: {}".format(str(j), NUM_FILES, file))
	name= file.split('/')
	image_name= name[-1]
	print("Processing ("+str(iter_n)+"/"+str(len(dataset_files))+") : "  + image_name+"")
	iter_n+=1
	original_img = cv2.imread(file,0)
	image = cv2.resize(original_img, (80, 80))
	image = preprocess(image)
	#image= np.transpose(image,(2,0,1)) # use this if the model has three channels 
	image = image.astype(np.float32)
	image_q = image

	if not DUMP_QUANT:
		data =[image]
		#print(data[0])
		outputs = executer.execute(data, qmode=None, silent=True)
		bboxes   = np.array(outputs[137][0])# taking just one dimension
		bclasses= np.array(outputs[138][0])
		bscores =np.array(outputs[139][0])
		#print(bboxes.shape, bscores.shape, bclasses.shape)
		csv_row, bbox= ExtractBBoxes(bboxes, bclasses,bscores,  80, 80, threshold,image_name)
		df_list.extend(csv_row)
		if not bbox:
			print('box_list is empty', image_name)
		else:
			if CREATE_DETECTION_FILES:
				out_img = DisplayDetections(original_img ,bbox,image_name)
				cv2.imwrite(non_quantization_result_path + '/' + image_name, out_img)

	else:
		# quant
		data_q    = [image_q]
		outputs = executer.execute(data_q, qmode=QuantizationMode.all_dequantize(), silent=True)
		bboxes   = np.array(outputs[137][0])# taking just one dimension
		bclasses= np.array(outputs[138][0])
		bscores =np.array(outputs[139][0])
		csv_row, bbox= ExtractBBoxes(bboxes, bclasses,bscores,  80, 80, threshold,image_name)
		df_list.extend(csv_row)
		if CREATE_DETECTION_FILES:
			out_img  =DisplayDetections(original_img ,bbox, image_name)
			cv2.imwrite(quantization_result_path + '/' + image_name, out_img )

if not DUMP_QUANT:
	df = pd.DataFrame(df_list, columns =['image_name', 'xmin','xmax','ymin','ymax', 'class_id', 'score'])
	df.to_csv(os.path.join(non_quantization_result_path, 'output_tflite.csv'), encoding='utf-8', index=False)
else:
	df = pd.DataFrame(df_list, columns =['image_name', 'xmin','xmax','ymin','ymax', 'class_id', 'score'])
	df.to_csv(os.path.join(quantization_result_path, 'output_tflite_quant.csv'), encoding='utf-8', index=False)








