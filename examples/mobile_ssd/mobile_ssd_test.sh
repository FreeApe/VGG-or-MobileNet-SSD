#!/bin/sh
if ! test -f examples/mobile_ssd/MobileNetSSD_test.prototxt ;then
	echo "error: example/MobileNetSSD_test.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot
./build/tools/caffe test -model="examples/mobile_ssd/MobileNetSSD_test.prototxt" \
-weights="snapshot/my_mobilenet_ssd_iter_10000.caffemodel" \
-gpu 0
