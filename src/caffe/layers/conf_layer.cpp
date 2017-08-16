#include <vector>

#include "caffe/layers/conf_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void ConfLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), 3);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());  
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());

  conf_ch = this->layer_param_.conf_channels_param().conf_ch();
  top[0]->Reshape(bottom[0]->num(), conf_ch, bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void ConfLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
  const Dtype* bottom_t_warped = bottom[0]->cpu_data();
  const Dtype* bottom_t = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int dim = height * width;
  
  conf_ch = this->layer_param_.conf_channels_param().conf_ch();
  
  
//LOG(INFO)<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~num= "<< num;
  
  caffe_set(top[0]->count(), Dtype(0.0), top_data);
  
  for (int n = 0; n < num; n++) {
  
    // step 1: minus, abs and sum  
    for (int c = 0; c < channels; c++) {
      for (int i = 0; i < dim; i++) {      
        top_data[n*conf_ch*dim + i] += abs(bottom_t_warped[(n*channels+c)*dim + i] - bottom_t[(n*channels+c)*dim + i]);      
      }
    }
  
  
    // step 2: compute reg 
    Dtype sum = 0.0;
    int s = 0;  
    
    for (int i = 0; i < dim; i++) {  
      if (bottom_t_warped[(n*channels)*dim+i] != 0 || bottom_t_warped[(n*channels+1)*dim+i] != 0 
           || bottom_t_warped[(n*channels+2)*dim+i] != 0) {
        sum += top_data[n*conf_ch*dim + i];
        s++;           
      }
    }
  
    Dtype reg = sum * 2.0 / s;

//LOG(INFO)<<"reg= "<< reg;
  
    // step 3: compute exp
    caffe_cpu_scale(dim, Dtype(-1.0/reg), top_data+n*conf_ch*dim, top_data+n*conf_ch*dim);//y = alpha*x 
    caffe_exp(dim, top_data+n*conf_ch*dim, top_data+n*conf_ch*dim);

    for (int i = 0; i < dim; i++) {  
      if (bottom_t_warped[(n*channels)*dim+i] == 0 && bottom_t_warped[(n*channels+1)*dim+i] == 0 
           && bottom_t_warped[(n*channels+2)*dim+i] == 0) {
        top_data[n*conf_ch*dim + i] = 0;
      }
    }
    
    
    // duplicate
    for (int c = 1; c < conf_ch; c++) {
        for (int i = 0; i < dim; i++) {        
            top_data[(n*conf_ch+c)*dim + i] = top_data[n*conf_ch*dim + i];        
        }    
    }
  }
}

template <typename Dtype>
void ConfLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    // nothing
    
}

#ifdef CPU_ONLY
STUB_GPU(ConfLayer);
#endif

INSTANTIATE_CLASS(ConfLayer);
REGISTER_LAYER_CLASS(Conf);

}  // namespace caffe
