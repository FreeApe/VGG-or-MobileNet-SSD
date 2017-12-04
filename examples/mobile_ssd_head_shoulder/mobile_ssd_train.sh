#!/bin/sh
if ! test -f examples/mobile_ssd/MobileNetSSD_train.prototxt ;then
	echo "error: example/MobileNetSSD_train.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
./build/tools/caffe train -solver="examples/mobile_ssd_head_shoulder/solver_train.prototxt" \
-weights="examples/mobile_ssd_head_shoulder/my_mobilenet_ssd_iter_10000.caffemodel" \
-gpu 0
