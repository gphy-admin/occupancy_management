"""
below code has many functions here is the summary of each:
conversion2cocevalformat since cocoevaluation requires center x and yy coordinates and Height and Width this fuction does that 
getFiles; this function creates the list of all image names present in test_image folder 
extract_ground_truth: we are only intrested in ground truth annotations of the the images in test_images folder so this function takes input the image_list formed by above funtion and the annotation csv, returns the data frame for thoes only images that are in the folder
image_indexing: the json coversion requires the unique index for each image, this image does same for the for the images in image_list returning the dict with key the index and value image name
json_creation= this creates the json for the ground truth annotation
json_result= this creates the json for quantized and non quantized 

one last thing if you want to change where created json file saves please make change in end at json_creation, json_result function
"""



import pandas as pd
import json
import os
pd.options.mode.chained_assignment = None 

#path for quantization and non quantization csv as this will be created by previous code in the respective folders
#annotationa dn test_images path
quantization_csv_path= 'quantized_output/output_tflite_quant.csv'
not_quantization_csv_path= 'non_quantized_output/output_tflite.csv'
annotation_path=  './test_data.csv'
test_images_path= './test_images'
def conversion2cocevalformat(df):
	df['height']      = df['xmax']- df['xmin']
	df['width']       = df['ymax']- df['ymin']
	df['x_center']   = df['xmin'] + (df['height']/2)
	df['y_center']   = df['ymin'] + (df['width']/2)
	return df

def extract_ground_truth(df, img):
	#print(df)
	new_df = df[df['image_name'].isin(img)]
	return new_df

def image_indexing(img_list):
	img_set= set(img_list)
	temp= dict()
	count=0
	for i in img_set:
		temp[count]= i
		count+=1
	return temp

def json_creation(df, img_dict):
	#print(df.shape)
	temp={	
		"info":{	"year": 2021,
                	"version": 1,
                	"description": 'Greenwaves Ajiesh',
                	"contributor": 'Ajitesh',
               		"url": 'Greenwaves Ajiesh',
               		"date_created": '26/03/2021'
               	},

        "licenses":[
        			{
						"id": 121,
						"name": 'Greenwaves Ajiesh',
						"url": 'Greenwaves Ajiesh',
					}
				],
		"images": [],
		"annotations":[],
		"categories":[
					{
						"id": 1,
						"name":"Person",
						"supercategory": "Person"
					}
					

					]

		}
	c=0
	for i in img_dict.keys():
		temp['images'].append({
								"id": i,
								"width": 80,
								"height": 80,
								"file_name": img_dict[i],
								"license": 121,
								"flickr_url": 'url',
								"coco_url": 'testing',
								"date_captured":"26/03/2021"
								})



		indicies= df.index[df['image_name'] == img_dict[i]].tolist()
		for j in indicies:

			temp['annotations'].append({
							"id": c,
							"image_id": i,
							"category_id": int(df.loc[j]['class_id']),
							"segmentation": [[int(df.loc[j]['xmin']), int(df.loc[j]['ymin']), int(df.loc[j]['xmin']), int(df.loc[j]['ymax']), int(df.loc[j]['xmax']), int(df.loc[j]['ymax']), int(df.loc[j]['xmax']), int(df.loc[j]['ymin'])]],
							"area": float(df.loc[j]['height']*df.loc[j]['width']),
							"bbox": [float(df.loc[j]['x_center']),float(df.loc[j]['y_center']),int(df.loc[j]['width']),int(df.loc[j]['height'])],
							"iscrowd": 0
							})
			c+=1
	
	with open('./input.json', 'w') as fp:
		json.dump(temp, fp)
	return

def json_result(df, img_dict,file_name):
	temp=list()

	for i in img_dict.keys():
		indicies= df.index[df['image_name'] == img_dict[i]].tolist()
		for j in indicies:
			x={
				"image_id": i,
				"category_id":  int(df.loc[j]['class_id']),
				"bbox":[float(df.loc[j]['x_center']), float(df.loc[j]['y_center']),int(df.loc[j]['width']),int(df.loc[j]['height'])],
				"score": float(df.loc[j]['score'])
			}
			temp.append(x)
		indicies=[]
	with open('./results' + file_name +'.json', 'w') as fp:
		json.dump(temp, fp)
	return
def getFiles(path):
	images_list=[]
	for file in os.listdir(path):
		if file.endswith(".png"):
			images_list.append(file)
	return images_list




if __name__ == '__main__':
	df_quantized= pd.read_csv(quantization_csv_path)
	df_not_quantized = pd.read_csv(not_quantization_csv_path)
	df_quantized_converted=conversion2cocevalformat(df_quantized)
	df_not_quantized_converted= conversion2cocevalformat(df_not_quantized)
	df_annotation= pd.read_csv(annotation_path)

	#get name of images in test_images folder 


	# since it would be same put it in list
	img_list= getFiles(test_images_path)
	img_dict = image_indexing(img_list)
	dataframe_annotated = extract_ground_truth(df_annotation, img_list)
	df_annotation_converted = conversion2cocevalformat(dataframe_annotated)
	#print(len(df_annotation_converted), len(df_not_quantized_converted),len(df_quantized_converted), img_dict )



	json_creation(df_annotation_converted, img_dict)

	json_result(df_not_quantized_converted, img_dict, 'not_quantized' )
	json_result(df_quantized_converted, img_dict, 'quantized' )








