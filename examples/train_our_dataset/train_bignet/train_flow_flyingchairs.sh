#!/bin/bash 
LOG=log/train_flow_flyingchairs-`date +%Y-%m-%d_%H-%M-%S`-$MODEL.log 
CAFFE=/home/s-lab/rgh/video-parsing2/build/tools/caffe
#../deeplab-superbear/build/tools/caffe 

#export PYTHONPATH=/home/bear/Documents/Studio/multi-rf/caffe-0902/python

$CAFFE train\
     --solver=solver_flow_flyingchairs.prototxt \
     --weights=flow_iter_220000-flychair-noscale.caffemodel,_iter_7500.caffemodel \
     --gpu=1 \
     2>&1 | tee $LOG
     
#--weights=flow_iter_600000.caffemodel,_iter_14520.caffemodel \


#  0.7527     0.5076         0.8918          0.6300       0.5455 
  
#   accuray   fg_accuracy   avg_precision    avg_recall     avg_f1
#  0.7235     0.6300         0.8918          0.5076       0.5455 

#   --weights=/media/bear/Models/trainning-models/video_256_32_nd_iter_137500.caffemodel\
#  --weights=../../pubmodels/alexnet/bvlc_alexnet.caffemodel\
#  --weights=../../pubmodels/wxl/color_model.caffemodel\
#   --weights=/media/bear/Models/trainning-models/makeup_256_32_fc67_mirror_iter_200000.caffemodel\
# --weights=/media/bear/Models/trainning-models/hole2_1e4_iter_7500.caffemodel\
#     --weights=/media/bear/Models/trainning-models/1path_lr7_batn_iter_24000.caffemodel\
#     --weights=init.caffemodel \
#     --weights=/media/bear/Models/trainning-models/1path_lr7_batn_iter_13200.caffemodel\
#   --snapshot=/media/bear/Models/trainning-models/3path_4loss_sel_fix_iter_5400.solverstate\
#   --weights=/media/bear/Models/multi-rf/base_model_train_x2_rf308/base_model_train_x2_rf308_iter_55000.caffemodel\
