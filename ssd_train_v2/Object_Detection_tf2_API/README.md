# Help:

Please read the tf2 object detection Installation to understand how to install the object detection Module.
Also please read the blog post on greenWaves site to better understand object detection Module.


## General

In objectdetection_API/ssd_data/ folder replace images,train_data.csv , test_data.csv, val_data.csv for the corresponding datasets. Put test images in objectdetection_API/test_images/ folder to get evaluation on corresponding test_images. Make changes in objectdetection_API/model_info/model.config with repect to the model being trained.

make changes for GPU and batch size 

Steps:

1. step1__Tfrecord_data_generation.sh : this is responsible for tfrecord generation corresponding .record file will be stored in ssd_data
2. step2__train_model.sh : this is responsible for the training the checkpoints are stored in ssd_data folder 
3. step2__evaluate_model.sh: Responsible for traiing evaluation 
4. step3__convert_checkpoints_to_pb.sh: this created pb model file in ssd_model folder
5. step4__tensorbord.sh  this gives lossvs epoch comparison on training vs evaluation 
6. step5__predictions_for_test_images.sh:  this uses step3 created model to make prediction for test images 
7. step6__conversion_to_tflite.sh  is used for tflite conversion of the model the tflite model is located in ssd_tflite_model folder 


Finally all the trained data is available in saved Models folder in Occupancy Managment Ajitesh Bhan Google Drive Folder, custom models created for EfficentNEt and MobileNEt are there for CenterNet api feature extractor hourglass 10 was used. 


The dataset can be found at:

https://drive.google.com/file/d/1yXAeixvb_f1p5SuB4gQR0TjhJRcm0iuF/view?usp=sharing


## TF 2.4 Api installation from scratch

### Requirements
```
conda create --name tf2.4_api python=3.7
conda activate tf2.4_api
conda install -c anaconda tensorflow-gpu=2.4.1
conda install -c anaconda spyder
conda install -c conda-forge tf-slim=0.1
conda install -c anaconda protobuf=3.15.6
conda install -c conda-forge cython=0.29.22
pip install tf-models-official==2.4.0
conda install -c anaconda tensorflow-gpu=2.4.1 ===> do it again
pip install yarl==1.6.2
pip install avro-python3==1.9.2.1
pip install pillow==8.1.2
pip install contextlib2==0.6.0.post1
pip install pycocotools==2.0.2
pip install apache-beam==2.28.0
pip install lvis==0.5.3
```

### Clone the models:

```
git clone https://github.com/tensorflow/models.git
```

Afterwards, apply all modifications required for new model constructions.
1. adding custom model into the object detection models:
../object_detection/models/ssd_custome_keras_feature_extractor.py
2. adding the custom model name into the mapping of the models
located in the model builder:
../object_detection/builders/model_builder.py

### Remember to activate your python environment first
```
cd models/research
```
### Compile protos:
```
protoc object_detection/protos/*.proto --python_out=.
```
### Copy the installation file:
```
cp object_detection/packages/tf2/setup.py .Apply modifications to ../models/research/setup.py

REQUIRED_PACKAGES = [
# Required for apache-beam with PY3
#'avro-python3',
'apache-beam',
#'pillow',
#'lxml',
'matplotlib',
#'Cython',
#'contextlib2',
#'tf-slim',
#'six',
#'pycocotools',
#'lvis',
'scipy',
'pandas',
#'tf-models-official'
]
```

Afterwards, install TensorFlow Object Detection API as a python package:

```
python -m pip install --use-feature=2020-resolver setup.py
```

copy object_detection folder into lynred project

Ready to use!!!
