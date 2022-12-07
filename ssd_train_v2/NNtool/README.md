# NNTool Testing Help:

1. Please place your tflite model in model/ folder named saved_model.tflite, you can change name but please make changes in corresponding scripts. 
2. Please place the test_images in to the test_images folder and copy some images in ‘ images_pour_quantization’ folder 
3. Please replace the test_data.csv file to corresponding test_images Ground truth CSV. Keep the name same.
4. Install and source the GAP SDK. For a detailed guide please refere to this [link](https://github.com/GreenWaves-Technologies/gap_sdk). 
5. Now you can run 
```./run_test.sh```
to save the model as .json 

This will create the json model file in model/ folder.

6. In utils there is nn_run.py file you can change DUMP_QUANT to False or True to get either quantized or non quantized output (by default it is set at False). There is NUM_FILES field it is set at 726 as that is total number of test_images. If you decide to only take less images for test change this number to the corresponding value. If there is change in model structure, you need to mention which layer is the output layer, e.g.:
```
	bboxes   = np.array(outputs[137][0])# taking just one dimension
	bclasses= np.array(outputs[138][0])
	bscores =np.array(outputs[139][0])
```
In my case layer no. 137, 138 and 139 correspond to these but if model structure changes this changes.

7. Run nntool -s ./model/run_nntool_test, this will create quantized and non quantized output based upon DUMP_QUANT field and save result in respective folders.

8. Then run in terminal python cocoeval_apres_quantization.py this will use the  quantized, non quantized output and annotations all there csv files it will create three jsons.
9. Then activate the environment created for object_detection_api to run python CocoEval_utils.py to get CocoEval ouput.

Note: You need to run it twice once by commenting cocoDt = cocoGt.loadRes(result_quantized_path)
next time uncommenting this and commenting cocoDt = cocoGt.loadRes(result_not_quantized), basically once for quantized and next time for non quantized. 
 

Below are the details from my old ReadME that explains above steps clearly.


Following should be the structure for the folders to run this:

- images_for_quantization(this folder contains all images needed for quantization process in nntool)
- model(this folder should contain the tflite model that needs to be quantized, and saved model json file by nntool will be created in it and  the bash script (run_nntool_test) that runs the python files to create quantization and nonquantization images)
- test_images(this folder contains the test_images that are needed for the evaluation)
- utils(this folder contains the code file (nn_run.py) this file is resonsible for test output for non_quantized_ouptput and quantized_ouptput )
- non_quantized_ouptput(this folder if not created will be created automatically this has the non_quantized output images and coorelating csv)
- quantized_ouptput(this folder if not created will be created automatically this has the quantized output images and coorelating csv)
- annotations.csv the ground truth csv for all the images
- run_test.sh(bash file)
- cocoeval.py( this python file is to be run after output has been created for quantization and non quantization)
- CocoEval_utils.py does cocoevaluation


**Please NOTE: you should add annotations.csv, test_images, images_pour_quantization, saved_model.tflite(in model folder) before beginning**

### Step1:

Is to go in your gap_sdk folder and open terminal run
source sourceme.sh


### Step2:

navigate to nntool_test folder from same terminal window and run
./run_test.sh
this uses images in folder 'images_pour_quantization' to do the quantization and this will create the saved_model.json files in the model folder


### Step3:

Now in terminal window of above which is at nntool folder run
nntool -s ./model/run_nntool_test
this bash script just open saved_model.json and runs it with it the nn_run.py that is located in utils. Please not to get non_quantized_ouptput set DUMP_QUANT = False and for quantized_ouptput set DUMP_QUANT = True, NUM_FILES = X (no. x should be less than equal to total images in test_images folder )
you will need to run this script twice as once for DUMP_QUANT = False and next one for DUMP_QUANT = True
This should create the outputs in folder non_quantized_ouptput and quantized_ouptput, respecive csvs's


### Step4:

Once you have created the non_quantized_ouptput.csv and quantized_ouptput.csv in the respective folder runfollowing command again you are in nntool_test folder:
python cocoeval.py
this will create 3 json files:
--input.json the json contatins ground truth annotation for given test_images csv
--resultsnot_quantized.json the result for not quantized
--resultsquantized.json the result for the quantized


### Step 5

Next you should have installed conda pycocotools in a env, activate that env and run python CocoEval_utils.py

(this file does the final coocevaluation) In this file you may need to comment one of  "cocoDt = cocoGt.loadRes(result_quantized_path)" or "cocoDt = cocoGt.loadRes(result_quantized_path)" and run twice (once with cocoDt = cocoGt.loadRes(result_quantized_path) and next time commenting this and running with cocoDt = cocoGt.loadRes(result_quantized_path)) as to get output for both quantized and non quantized outputs


Also this might give a numpy linespace error when running this file, the error  as following:

```
TypeError: object of type <class 'numpy.float64'> cannot be safely interpreted as an integer.
```

To solve this there are two ways:

If numpy version is higher than 1.18 either you can switch to Numpy 1.17.5 in this env
or second make following change in pycocoeval.py file
self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
with
self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)