# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""SSDFeatureExtractor for MobilenetV2 features."""

import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.models.keras_models import mobilenet_v2
from object_detection.utils import ops
from object_detection.utils import shape_utils

class SSDMCustomKerasFeatureExtractor(
    ssd_meta_arch.SSDKerasFeatureExtractor):
  """SSD Feature Extractor using MobilenetV2 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=4,
               override_base_feature_extractor_hyperparams=False,
               name=None):
    """MobileNetV2 Feature Extractor for SSD Models.

    Mobilenet v2 (experimental), designed by sandler@. More details can be found
    in //knowledge/cerebra/brain/compression/mobilenet/mobilenet_experimental.py

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor (Functions
        as a width multiplier for the mobilenet_v2 network itself).
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(SSDMCustomKerasFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams=conv_hyperparams,
        freeze_batchnorm=freeze_batchnorm,
        inplace_batchnorm_update=inplace_batchnorm_update,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        num_layers=num_layers,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams,
        name=name)

    ''' Construction of model layers '''
    self.construct_layers()


    self._feature_map_layout = {
        'from_layer': ['mp_1', 'mp_2', 'mp_3', 'mp_4'][:self._num_layers],
        'layer_depth': [-1, -1, -1, -1][:self._num_layers],
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding,
    }

    self.classification_backbone = None
    self.feature_map_generator = None
    
  def construct_layers(self):

    layer_number=1
    self.conv_1      = tf.keras.layers.Conv2D(32,(3,3), strides=(2, 2), padding="same", name = str(layer_number)+'_conv')
    self.bn_1        = tf.keras.layers.BatchNormalization(momentum=0.99, name = str(layer_number)+'_bn')
    self.act_1       = tf.keras.layers.Activation('relu', name = str(layer_number)+'_activation')

    layer_number=2
    self.deapth_conv_2   = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (1, 1), padding='same', use_bias= False,  name = str(layer_number)+'_deapthwise')
    self.deapth_bn_2     = tf.keras.layers.BatchNormalization(momentum=0.99 , name=str(layer_number)+'_BN_deapthwise')
    self.deapth_act_2    = tf.keras.layers.Activation('relu' , name =str(layer_number)+'_activation_deapthwise')

    laye_number=3
    self.point_conv_3   = tf.keras.layers.Conv2D(24, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_pointwise')
    self.point_bn_3     = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_BN_pointwise')

    layer_number=4
    self.point_conv_4_1  = tf.keras.layers.Conv2D(144, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_4_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_4_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')

    self.deapth_conv_4_1 = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (1,1), padding='same', use_bias=False, name= str(layer_number)+'_1_deaptwise')
    self.deapth_bn_4_1   = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_1_BN_deapthwise')
    self.deapth_act_4_1  = tf.keras.layers.Activation('relu', name = str(layer_number)+'_1_activation_deapthwise')

    self.point_conv_4_2  = tf.keras.layers.Conv2D(32, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_2_pointwise')
    self.point_bn_4_2    = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_2_BN_pointwise')


    layer_number=5
    self.point_conv_5_1  = tf.keras.layers.Conv2D(192, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_5_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_5_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')

    self.deapth_conv_5_1 = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (1,1), padding='same', use_bias=False, name= str(layer_number)+'_1_deaptwise')
    self.deapth_bn_5_1   = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_1_BN_deapthwise')
    self.deapth_act_5_1  = tf.keras.layers.Activation('relu', name = str(layer_number)+'_1_activation_deapthwise')

    self.point_conv_5_2  = tf.keras.layers.Conv2D(32, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_2_pointwise')
    self.point_bn_5_2    = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_2_BN_pointwise')

    self.add_1  = tf.keras.layers.Add()

    layer_number=6
    self.point_conv_6_1  = tf.keras.layers.Conv2D(192, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_6_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_6_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')

    self.deapth_conv_6_1 = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (1,1), padding='same', use_bias=False, name= str(layer_number)+'_1_deaptwise')
    self.deapth_bn_6_1   = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_1_BN_deapthwise')
    self.deapth_act_6_1  = tf.keras.layers.Activation('relu', name = str(layer_number)+'_1_activation_deapthwise')

    self.point_conv_6_2  = tf.keras.layers.Conv2D(32, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_2_pointwise')
    self.point_bn_6_2    = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_2_BN_pointwise')
    
    self.add_2  = tf.keras.layers.Add()



    layer_number =7
    self.point_conv_7_1  = tf.keras.layers.Conv2D(192, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_7_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.mp1              = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')
    #out_1
    layer_number =8

    self.deapth_conv_8_1 = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (2,2), padding='same', use_bias=False, name= str(layer_number)+'_1_deaptwise')
    self.deapth_bn_8_1   = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_1_BN_deapthwise')
    self.deapth_act_8_1  = tf.keras.layers.Activation('relu', name = str(layer_number)+'_1_activation_deapthwise')

    self.point_conv_8_2  = tf.keras.layers.Conv2D(64, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_2_pointwise')
    self.point_bn_8_2    = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_2_BN_pointwise')

    layer_number =9

    self.point_conv_9_1  = tf.keras.layers.Conv2D(384, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_9_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_9_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')


    self.deapth_conv_9_1 = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (1,1), padding='same', use_bias=False, name= str(layer_number)+'_1_deaptwise')
    self.deapth_bn_9_1   = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_1_BN_deapthwise')
    self.deapth_act_9_1  = tf.keras.layers.Activation('relu', name = str(layer_number)+'_1_activation_deapthwise')

    self.point_conv_9_2  = tf.keras.layers.Conv2D(64, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_2_pointwise')
    self.point_bn_9_2    = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_2_BN_pointwise')

    self.add_3           = tf.keras.layers.Add()


    layer_number=10
    self.point_conv_10_1  = tf.keras.layers.Conv2D(384, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_10_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_10_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')

    self.deapth_conv_10_1 = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (1,1), padding='same', use_bias=False, name= str(layer_number)+'_1_deaptwise')
    self.deapth_bn_10_1   = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_1_BN_deapthwise')
    self.deapth_act_10_1  = tf.keras.layers.Activation('relu', name = str(layer_number)+'_1_activation_deapthwise')

    self.point_conv_10_2  = tf.keras.layers.Conv2D(64, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_2_pointwise')
    self.point_bn_10_2    = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_2_BN_pointwise')

    self.add_4  = tf.keras.layers.Add()

    layer_number=11
    self.point_conv_11_1  = tf.keras.layers.Conv2D(384, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_11_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_11_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')

    self.deapth_conv_11_1 = tf.keras.layers.DepthwiseConv2D( (3,3), strides= (1,1), padding='same', use_bias=False, name= str(layer_number)+'_1_deaptwise')
    self.deapth_bn_11_1   = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_1_BN_deapthwise')
    self.deapth_act_11_1  = tf.keras.layers.Activation('relu', name = str(layer_number)+'_1_activation_deapthwise')

    self.point_conv_11_2  = tf.keras.layers.Conv2D(96, (1,1), strides= (1,1), padding="same", use_bias=False, name=str(layer_number)+'_2_pointwise')
    self.point_bn_11_2    = tf.keras.layers.BatchNormalization(momentum=0.99, name=str(layer_number)+'_2_BN_pointwise')


    layer_number= 12
    self.point_conv_12_1  = tf.keras.layers.Conv2D(576, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_12_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.mp2              = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')
    #out_2

    layer_number= 13

    self.point_conv_13_1  = tf.keras.layers.Conv2D(128, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_13_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_13_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')

    self.conv_13      = tf.keras.layers.Conv2D(256,(3,3), strides=(2, 2), padding="same", name = str(layer_number)+'_conv')
    self.bn_13        = tf.keras.layers.BatchNormalization(momentum=0.99, name = str(layer_number)+'_bn')
    self.mp3          = tf.keras.layers.Activation('relu', name = str(layer_number)+'_activation')

    #out_3

    layer_number=15

    self.point_conv_14_1  = tf.keras.layers.Conv2D(128, (1, 1), strides= (1, 1), padding="same", use_bias=False, name=str(layer_number)+'_1_pointwise')
    self.point_bn_14_1    = tf.keras.layers.BatchNormalization( momentum=0.99, name=str(layer_number)+'_1_BN_pointwise')
    self.point_act_14_1   = tf.keras.layers.Activation('relu' , name=str(layer_number)+'_1_activation_pointwise')

    self.conv_14      = tf.keras.layers.Conv2D(256,(3,3), strides=(2, 2), padding="same", name = str(layer_number)+'_conv')
    self.bn_14        = tf.keras.layers.BatchNormalization(momentum=0.99, name = str(layer_number)+'_bn')
    self.mp4       = tf.keras.layers.Activation('relu', name = str(layer_number)+'_activation')
    #out_4

  def build(self, input_shape):
    
    'input layer'
    Input = tf.keras.layers.Input(shape=(input_shape[1], input_shape[2],input_shape[3]), batch_size = 8)
    
    #layer1
    out_conv_1 = self.conv_1(Input)
    out_bn_1   = self.bn_1(out_conv_1)
    out_act_1  = self.act_1(out_bn_1)

    #layer2
    out_deapth_conv_2 = self.deapth_conv_2(out_act_1)
    out_deapth_bn_2   = self.deapth_bn_2(out_deapth_conv_2)
    out_deapth_act_2 =  self.deapth_act_2(out_deapth_bn_2)
    
    #layer3
    out_point_conv_3 = self.point_conv_3(out_deapth_act_2)
    out_point_bn_3   = self.point_bn_3(out_point_conv_3)

    #layer4
    out_point_conv_4_1  = self.point_conv_4_1(out_point_bn_3)
    out_point_bn_4_1    = self.point_bn_4_1(out_point_conv_4_1)
    out_point_act_4_1   = self.point_act_4_1(out_point_bn_4_1)
    out_deapth_conv_4_1 = self.deapth_conv_4_1(out_point_act_4_1)
    out_deapth_bn_4_1   = self.deapth_bn_4_1(out_deapth_conv_4_1)
    out_deapth_act_4_1  = self.deapth_act_4_1(out_deapth_bn_4_1)
    out_point_conv_4_2  = self.point_conv_4_2(out_deapth_act_4_1)
    out_point_bn_4_2    = self.point_bn_4_2(out_point_conv_4_2)


    #layer5
    out_point_conv_5_1  = self.point_conv_5_1(out_point_bn_4_2)
    out_point_bn_5_1    = self.point_bn_5_1(out_point_conv_5_1)
    out_point_act_5_1   = self.point_act_5_1(out_point_bn_5_1)
    out_deapth_conv_5_1 = self.deapth_conv_5_1(out_point_act_5_1)
    out_deapth_bn_5_1   = self.deapth_bn_5_1(out_deapth_conv_5_1)
    out_deapth_act_5_1  = self.deapth_act_5_1(out_deapth_bn_5_1)
    out_point_conv_5_2  = self.point_conv_5_2(out_deapth_act_5_1)
    out_point_bn_5_2    = self.point_bn_5_2(out_point_conv_5_2)

    out_add_1 = self.add_1([out_point_bn_4_2, out_point_bn_5_2])

    #layer6
    out_point_conv_6_1  = self.point_conv_6_1(out_add_1)
    out_point_bn_6_1    = self.point_bn_6_1(out_point_conv_6_1)
    out_point_act_6_1   = self.point_act_6_1(out_point_bn_6_1)
    out_deapth_conv_6_1 = self.deapth_conv_6_1(out_point_act_6_1)
    out_deapth_bn_6_1   = self.deapth_bn_6_1(out_deapth_conv_6_1)
    out_deapth_act_6_1  = self.deapth_act_6_1(out_deapth_bn_6_1)
    out_point_conv_6_2  = self.point_conv_6_2(out_deapth_act_6_1)
    out_point_bn_6_2    = self.point_bn_6_2(out_point_conv_6_2)

    out_add_2 = self.add_2([out_add_1, out_point_bn_6_2])

    #layer7
    out_point_conv_7_1  = self.point_conv_7_1(out_add_2)
    out_point_bn_7_1    = self.point_bn_7_1(out_point_conv_7_1)
    out_mp1             = self.mp1(out_point_bn_7_1)


    #layer 8
    out_deapth_conv_8_1 = self.deapth_conv_8_1(out_mp1)
    out_deapth_bn_8_1   = self.deapth_bn_8_1(out_deapth_conv_8_1)
    out_deapth_act_8_1  = self.deapth_act_8_1(out_deapth_bn_8_1)
    out_point_conv_8_2  = self.point_conv_8_2(out_deapth_act_8_1)
    out_point_bn_8_2    = self.point_bn_8_2(out_point_conv_8_2)


    #layer9
    out_point_conv_9_1  = self.point_conv_9_1(out_point_bn_8_2)
    out_point_bn_9_1    = self.point_bn_9_1(out_point_conv_9_1)
    out_point_act_9_1   = self.point_act_9_1(out_point_bn_9_1)
    out_deapth_conv_9_1 = self.deapth_conv_9_1(out_point_act_9_1)
    out_deapth_bn_9_1   = self.deapth_bn_9_1(out_deapth_conv_9_1)
    out_deapth_act_9_1  = self.deapth_act_9_1(out_deapth_bn_9_1)
    out_point_conv_9_2  = self.point_conv_9_2(out_deapth_act_9_1)
    out_point_bn_9_2    = self.point_bn_9_2(out_point_conv_9_2)

    out_add_3 = self.add_3([out_point_bn_8_2, out_point_bn_9_2])

    #layer10
    out_point_conv_10_1  = self.point_conv_10_1(out_add_3)
    out_point_bn_10_1    = self.point_bn_10_1(out_point_conv_10_1)
    out_point_act_10_1   = self.point_act_10_1(out_point_bn_10_1)
    out_deapth_conv_10_1 = self.deapth_conv_10_1(out_point_act_10_1)
    out_deapth_bn_10_1   = self.deapth_bn_10_1(out_deapth_conv_10_1)
    out_deapth_act_10_1  = self.deapth_act_10_1(out_deapth_bn_10_1)
    out_point_conv_10_2  = self.point_conv_10_2(out_deapth_act_10_1)
    out_point_bn_10_2    = self.point_bn_10_2(out_point_conv_10_2)

    out_add_4 = self.add_3([out_add_3, out_point_bn_10_2])


    #layer11
    out_point_conv_11_1  = self.point_conv_11_1(out_add_4)
    out_point_bn_11_1    = self.point_bn_11_1(out_point_conv_11_1)
    out_point_act_11_1   = self.point_act_11_1(out_point_bn_11_1)
    out_deapth_conv_11_1 = self.deapth_conv_11_1(out_point_act_11_1)
    out_deapth_bn_11_1   = self.deapth_bn_11_1(out_deapth_conv_11_1)
    out_deapth_act_11_1  = self.deapth_act_11_1(out_deapth_bn_11_1)
    out_point_conv_11_2  = self.point_conv_11_2(out_deapth_act_11_1)
    out_point_bn_11_2    = self.point_bn_11_2(out_point_conv_11_2)


    #layer12
    out_point_conv_12_1  = self.point_conv_12_1(out_point_bn_11_2)
    out_point_bn_12_1    = self.point_bn_12_1(out_point_conv_12_1)
    out_mp2              = self.mp2(out_point_bn_12_1)

    #layer 13

    out_point_conv_13_1  = self.point_conv_13_1(out_mp2)
    out_point_bn_13_1    = self.point_bn_13_1(out_point_conv_13_1)
    out_point_act_13_1   = self.point_act_13_1(out_point_bn_13_1)
    out_conv_13          = self.conv_13(out_point_act_13_1)
    out_bn_13            = self.bn_13(out_conv_13)
    out_mp3              = self.mp3(out_bn_13)   

    #LAYUER 14
    out_point_conv_14_1  = self.point_conv_14_1(out_mp3)
    out_point_bn_14_1    = self.point_bn_14_1(out_point_conv_14_1)
    out_point_act_14_1   = self.point_act_14_1(out_point_bn_14_1)
    out_conv_14          = self.conv_14(out_point_act_14_1)
    out_bn_14            = self.bn_14(out_conv_14)
    out_mp4              = self.mp4(out_bn_14)



    'construct a base model without any prediction/classification layers'
    base_model = tf.keras.Model(inputs = Input, outputs=[out_mp1, out_mp2, out_mp3, out_mp4])
    
    'employ base model as a backbone for classification'
    self.classification_backbone = base_model

    '''
    construct feature map generator
    
    this should be remined that there should be correspondence between the layers' name provided
    in feature_map_layout and backbone_classification model, as feature map generator takes those
    layers' outputs as inputs for feature maps construction.
    
    '''
    self.feature_map_generator = (
    feature_map_generators.KerasMultiResolutionFeatureMaps(
        feature_map_layout=self._feature_map_layout,
        depth_multiplier=self._depth_multiplier,
        min_depth=self._min_depth,
        insert_1x1_conv=True,
        is_training=self._is_training,
        conv_hyperparams=self._conv_hyperparams,
        freeze_batchnorm=self._freeze_batchnorm,
        name='FeatureMaps'))
    self.built = True


  def preprocess(self, resized_inputs):
    resized_inputs = resized_inputs[:,:,:,0:1]
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def _extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    'pass preprocessed input to the classification backbone'
    image_features = self.classification_backbone(
        ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))
    self.image_features_ab = image_features
    
    'construct feature maps from image features.'
    feature_maps = self.feature_map_generator({
        'mp_1': image_features[0],
        'mp_2': image_features[1],
        'mp_3': image_features[2],
        'mp_4': image_features[3]})

    self.feature_maps = feature_maps

    return list(feature_maps.values())
