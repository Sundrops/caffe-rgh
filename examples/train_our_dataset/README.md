## train_our_dataset
### train_1path
原版deeplab

### train_3path

3帧deeplab pool5后进行global的融合，辅助当前帧的parsing结果(具体思想见weiliu的ParseNet)
注: 此处的global融合可用1*1卷积进行融合，也可以reshape后只学习3个权重参数进行相加

### train_bignet

#### train_flow_flyingchairs.prototxt

这个prototxt直接用在flyingchair上训练的光流进行finetuning :flow_iter_220000-flychair-noscale.caffemodel
得到：当前帧+flow_flyingchairs
(当然这个prototxt也可以利用在我们数据库通过remap训练的model进行fine : flow_iter_500000-remap-noscale.caffemodel
得到：当前帧+flow_remap)
#### train_bignet.prototxt

这个prototxt是在上一个prototxt的基础上加了3帧的global
然后两种finetuning
flow_iter_220000-flychair-noscale.caffemodel : 当前帧+3帧global+flow_flyingchairs
flow_iter_500000-remap-noscale.caffemodel : 当前帧+3帧global+flow_remap


