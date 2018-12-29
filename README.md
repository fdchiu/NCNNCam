# NCNNCam

A framework that incoporate android camera and ncnn deep learning model deployment in a single app. The default android samples provided from ncnn repo work on a single image. This sample shows how to open android camera and feed frames to ncnn models for inference continuously. 


If you want to knoe about ncnn, please refer to: https://github.com/Tencent/ncnn 

About the yolo model included with the repo:
Model name: yolov2-tiny_voc
For more info on yolo, please go to: https://pjreddie.com/darknet/yolov2/ and https://github.com/thtrieu/darkflow
I used darknet2caffe to convert yolo to caffe, then used caffe2ncnn to convert the model into NCNN deployable model. Also ncnn2mem is used to fruther convert the model format to binary for use with the android framework: NCNNCam.

The Camera part comes from Google's tensorflow android example and I adapted it for use in this framework. Tensorflow repo is available from: https://github.com/tensorflow/tensorflow






