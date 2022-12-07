CONFIG_FILE=./ssd_model/pipeline.config
CHECKPOINT_PATH=./ssd_model/
OUTPUT_DIR=ssd_tflite_model
OUTPUT_DIR_TFLITE="ssd_tflite_model/saved_model/"

python3 object_detection/export_tflite_graph_tf2.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_dir=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \

#mv ssd_model/pb_model/saved_model $OUTPUT_DIR


python3 pb2tflite.py \
--saved_model_directory=$OUTPUT_DIR_TFLITE 

#cp ssd_model/pipeline.config $OUTPUT_DIR









