DATASET_PATH=./ssd_data
CSV_TRAIN=./ssd_data/train_data.csv
CSV_VAL=./ssd_data/val_data.csv
CSV_TEST=./ssd_data/test_data.csv

# generate tfrecords for training object detection
# train
python3 ./TFRecord_data_gen/generate_tfrecord.py --path_to_images $DATASET_PATH/images --path_to_annot $CSV_TRAIN \
                            --path_to_label_map ./model_info/labelmap.pbtxt \
                            --path_to_save_tfrecords $DATASET_PATH/train.record
# validation
python3 ./TFRecord_data_gen/generate_tfrecord.py --path_to_images $DATASET_PATH/images --path_to_annot $CSV_VAL \
                            --path_to_label_map ./model_info/labelmap.pbtxt \
                            --path_to_save_tfrecords $DATASET_PATH/val.record


# validation
python3 ./TFRecord_data_gen/generate_tfrecord.py --path_to_images $DATASET_PATH/images --path_to_annot $CSV_TEST \
                            --path_to_label_map ./model_info/labelmap.pbtxt \
                            --path_to_save_tfrecords $DATASET_PATH/test.record
