# uncomment this to run the test on CPU
export CUDA_VISIBLE_DEVICES="-1"

PIPELINE_CONFIG_PATH=./model_info/model.config
MODEL_DIR=./ssd_model
CHECKPOINT_DIR=./ssd_model
python3 ssd_model_tf2_evaluation.py \
    --pipeline_config_path=$PIPELINE_CONFIG_PATH \
    --model_dir=$MODEL_DIR \
    --checkpoint_dir=$CHECKPOINT_DIR \
    --alsologtostderr
    
