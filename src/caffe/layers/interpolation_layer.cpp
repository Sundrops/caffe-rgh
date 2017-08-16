/*
# An inflation operation on feature maps with a learnable scaling factor. //revise
#
# Author: Sun Yao @ IIE, CAS
# Create on: 2016-07-16
# Last modify: 2016-12-29
#
*/

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/interpolation_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))


namespace caffe {

template <typename Dtype>
void InterpolationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}



template <typename Dtype>
void InterpolationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // blob shape check
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                                       << "corresponding to (num, channels, height, width)";


    // set factor
    factor_value_h_ = (bottom[1]->height() - 1.0) / (bottom[0]->height() - 1.0);
    factor_value_w_ = (bottom[1]->width() - 1.0) / (bottom[0]->width() - 1.0);    
    CHECK_LE(factor_value_h_, this->MAX_FACTOR);
    CHECK_GE(factor_value_h_, this->MIN_FACTOR);
    CHECK_LE(factor_value_w_, this->MAX_FACTOR);
    CHECK_GE(factor_value_w_, this->MIN_FACTOR);


    // calculate the top's shape 
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(), bottom[1]->width());
    
}



template <typename Dtype>
void InterpolationLayer<Dtype>::inflate_forward(const Dtype *bottom_data, const int bottom_height, const int bottom_width, 
                                                Dtype *top_data, const int top_height, const int top_width, 
                                                const float factor_h, const float factor_w) {
    
    int margin_ = 1;

    for (int y_t = 0; y_t < top_height; y_t++) {
        for (int x_t = 0; x_t < top_width; x_t++) {
            
            // coordinate on target map
            const int idx_t = y_t * top_width + x_t;
            
            top_data[idx_t] = 0;
            
            float y_s = y_t / factor_h;
            float x_s = x_t / factor_w;
            CHECK_LT(y_s, bottom_height);
            CHECK_LT(x_s, bottom_width);
            
            for (int n = MAX(floor(y_s - margin_) + 1, 0); n < MIN(y_s + margin_, bottom_height); n++) {
                for (int m = MAX(floor(x_s - margin_) + 1, 0); m < MIN(x_s + margin_, bottom_width); m++) {
             
                    top_data[idx_t] += bottom_data[n * bottom_width + m] * (margin_ - abs(x_s - m)) * (margin_ - abs(y_s - n));

                }
            }
        }
    }
}



template <typename Dtype>
void InterpolationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    
    // new shape
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();

    // resize
    for (int n = 0; n < num; n++) {  
        for (int c = 0; c < channels; c++) {
            const int index_in = (n * channels + c) * height * width;
            const int index_out = (n * channels + c) * top_height * top_width;
            inflate_forward(bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, factor_value_h_, factor_value_w_);
        }
    }
}




template <typename Dtype>
void InterpolationLayer<Dtype>::inflate_backward(Dtype *bottom_diff, const int bottom_height, const int bottom_width, 
                                                 const Dtype *top_diff, const int top_height, const int top_width, 
                                                 const float factor_h, const float factor_w) {

    const float normalizer = factor_h * factor_w;
    
    int margin_ = 1;
    
    for (int n = 0; n < bottom_height; n++) {
        for (int m = 0; m < bottom_width; m++) {
        
            // coordinate on target map
            const int idx_s = n * bottom_width + m;
            bottom_diff[idx_s] = 0;
            
            for (int y_t = MAX(floor((n - margin_) * factor_h) + 1, 0); y_t < MIN((n + margin_) * factor_h, top_height); y_t++) { 
                for (int x_t = MAX(floor((m - margin_) * factor_w) + 1, 0); x_t < MIN((m + margin_) * factor_w, top_width); x_t++) {
        
                    // diff
                    bottom_diff[idx_s] += top_diff[y_t * top_width + x_t] 
                                          * (margin_ - abs((x_t / factor_w) - m)) * (margin_ - abs((y_t / factor_h) - n));               
                }
            }

            // normalize
            bottom_diff[idx_s] /= normalizer;
        
        }
    }
}




template <typename Dtype>
void InterpolationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();
    
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();



    if (propagate_down[0]) {

        // compute diff for bottom    
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channels; c++) {
                const int index_in = (n * channels + c) * height * width;
                const int index_out = (n * channels + c) * top_height * top_width;
                inflate_backward(bottom_diff + index_in, height, width, top_diff + index_out, top_height, top_width, factor_value_h_, factor_value_w_);
            }
        }
    }

    
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
                   << " Layer cannot backpropagate to the second input.";
    }
    

}

#ifdef CPU_ONLY
STUB_GPU(InterpolationLayer);
#endif
INSTANTIATE_CLASS(InterpolationLayer);
REGISTER_LAYER_CLASS(Interpolation);
}  // namespace caffe
