#ifndef CAFFE_CONF_LAYER_HPP_
#define CAFFE_CONF_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Compute confidence of the predicted optimalflow
 */
template <typename Dtype>
class ConfLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param no
   */
  explicit ConfLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
      
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);    

  virtual inline const char* type() const { return "Conf"; }

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
 

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs: image t_warped, image t.
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs: conf_warped
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);

  /**
   * @brief no backwards
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  int conf_ch;
};

}  // namespace caffe

#endif  // CAFFE_CONF_LAYER_HPP_
