#!/bin/sh

python3 -m oneshotdetection.main \
    --image data/images/zidane.jpg \
    --model bin/yolov5s_weights.pt \
    --config config/size_s.yaml \
    --input-shape 640 640 \
    --output-dir runs \
    --output-file result.jpg
