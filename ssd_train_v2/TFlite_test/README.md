# READ ME

- Place test images in test_images folder 
- Please change the layer no. if there is change in Model structure. The script works for custom MobileNET, for other please make sure the interpreter takes the correct output.
- Place tflite model in the correct place, it should be named saved_model.tflite or if you want to change the name please make changes in script 
- RUN SCRIPT python tflite_test.py 
	1. this creates the annotation_results_tflite.csv that has the results on bounding box's 
	2. test_prediction folder will have  predicted images with bounding box 


