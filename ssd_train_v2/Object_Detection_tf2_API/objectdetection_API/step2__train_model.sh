out_dir=./ssd_model
mkdir -p $out_dir
python3 ./ssd_model_tf2.py \
		 --alsologtostderr --model_dir=$out_dir --checkpoint_every_n=500  \
          	 --pipeline_config_path=./model_info/model.config \
                 --eval_on_train_data 2>&1 | tee $out_dir/train.log
