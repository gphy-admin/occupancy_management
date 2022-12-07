
rm -rf images_for_quantization
mkdir images_for_quantization

shuf -zn50 -e ../../Object_Detection_tf2_API/objectdetection_API/ssd_data/images/*.png | xargs -0 cp -vt images_for_quantization/