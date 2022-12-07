# READ ME


1. The basic_analysis.py provides:
	+ TP Count
	+ FP Count
	+ Precision
	+ Recall 
	+ F1 score
	+ Accuracy 

2. The complete_analysis.ipynb provides the section wise analysis and Complete breakdown of area and confidence for each section. There is also an optional border correction, this desribed the pixels to be excluded from the boundary to have less perturbed detections.

3. Both files need change input for ground truth annotations and predicted annotations when you changes the files.
df_annotation_ground_truth= pd.read_csv("./test_data.csv")
df_annotations_predicted= pd.read_csv("./output_tflite_quant.csv")


Also IOU (intersection over union), that is what percentage of intersection you want with ground truth. If you choose IOU 0.70 it will be counted as FP even if it is detected because IOU calculated is less than the mentioned.

Note: API provides box’s with 60 percent accuracy and wrt project people counting we could use IOU=0.0 as we associate best predicted with ground truth value. IF there are multiple box’s detected I take best one and the remainings are associated as FP. 

Rest is explained in code 