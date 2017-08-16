#include <algorithm>
#include <vector>

#include "caffe/layers/mask_layer.hpp"

namespace caffe {

template <typename Dtype>
void MASKLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  /* only one background
  Dtype background = this->layer_param_.mask_param().background();
//LOG(INFO) << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~background: "<<background;
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i]==background?Dtype(0):Dtype(1);
  }
  */
  // vector<int> background;
  background.clear();
  std::copy(this->layer_param_.mask_param().background().begin(),
      this->layer_param_.mask_param().background().end(),
      std::back_inserter(background));
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < background.size(); ++j) {
      if(bottom_data[i]==Dtype(background[j])){
        top_data[i] = Dtype(0);
        break;
      }else{
        top_data[i] = Dtype(1);
      }
    }
  }
}

template <typename Dtype>
void MASKLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
return;
}


#ifdef CPU_ONLY
STUB_GPU(MASKLayer);
#endif

INSTANTIATE_CLASS(MASKLayer);
REGISTER_LAYER_CLASS(MASK);

}  // namespace caffe
