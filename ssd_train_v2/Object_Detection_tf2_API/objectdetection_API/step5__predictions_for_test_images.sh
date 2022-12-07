PATH_TO_MODEL=./ssd_model/pb_model/saved_model
PATH_TO_LABEL_MAP=./model_info/labelmap.pbtxt
PATH_TO_IMAGES=./test_images
PATH_TO_SAVE_IMAGES=./test_predictions
python3 ./detector/detect_objects.py \
      --threshold 0.5 \
      --path_to_model=$PATH_TO_MODEL \
      --path_to_label_map=$PATH_TO_LABEL_MAP \
      --path_to_images=$PATH_TO_IMAGES \
      --path_to_save_images=$PATH_TO_SAVE_IMAGES
      
