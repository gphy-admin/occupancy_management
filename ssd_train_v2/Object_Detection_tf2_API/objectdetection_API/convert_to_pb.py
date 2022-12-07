# Lint as: python2, python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl import app
from absl import flags

import tensorflow.compat.v2 as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2
import argparse
import os

tf.enable_v2_behavior()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Convert model checkpoints into a pb model.")
  parser.add_argument("--path_to_model", type=str, help="full path to the trained ssd model.", default="../ssd_model")
  args = parser.parse_args()
  path_to_model = args.path_to_model
  out_dir=os.path.join(path_to_model, 'pb_model')
  input_type='image_tensor'
  pipeline_config_path = os.path.join(path_to_model, 'pipeline.config')
  trained_checkpoint_dir = path_to_model
  output_directory = out_dir
  config_override = ''
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.io.gfile.GFile(pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(config_override, pipeline_config)
  exporter_lib_v2.export_inference_graph(
      input_type, pipeline_config, trained_checkpoint_dir,
      output_directory)
