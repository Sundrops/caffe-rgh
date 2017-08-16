#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/interpolation_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))



namespace caffe {

template <typename Dtype>
__global__ void InflateForwardGPU(const int nthreads,
          const Dtype* bottom_data, const int bottom_height, const int bottom_width, 
          Dtype *top_data, const int top_height, const int top_width, 
          const float factor_h, const float factor_w) {
          
    
          
    CUDA_KERNEL_LOOP(index, nthreads) {
    
        const int margin = 1;
        
        // index refers to to top_data
        const int y_t = index / top_width;
        const int x_t = index % top_width;
        
        
        // coordinate on target map
        const int idx_t = y_t * top_width + x_t;
            
        top_data[idx_t] = 0;
            
        float y_s = y_t / factor_h;
        float x_s = x_t / factor_w;
            
        for (int n = MAX(floor(y_s - margin) + 1, 0); n < MIN(y_s + margin, bottom_height); n++) {
            for (int m = MAX(floor(x_s - margin) + 1, 0); m < MIN(x_s + margin, bottom_width); m++) {
             
                top_data[idx_t] += bottom_data[n * bottom_width + m] * (margin - abs(x_s - m)) * (margin - abs(y_s - n));

            }
        }
        
    }
}





template <typename Dtype>
void InterpolationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    
    // new shape
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();

    // resize
    const int nthreads = top_height * top_width;
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channels; c++) {    
      
            const int index_in = (n * channels + c) * height * width;
            const int index_out = (n * channels + c) * top_height * top_width;

            InflateForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, factor_value_h_, factor_value_w_);

        }

    }
}




template <typename Dtype>
__global__ void InflateBackwardGPU(const int nthreads, 
            Dtype *bottom_diff, const int bottom_height, const int bottom_width, 
            const Dtype *top_diff, const int top_height, const int top_width, 
            const float factor_h, const float factor_w) {
            

    const float normalizer = factor_h * factor_w;

    CUDA_KERNEL_LOOP(index, nthreads) {
    
        const int margin = 1;
        
        // index refers to to top_data
        const int n = index / bottom_width;
        const int m = index % bottom_width;
        
        const int idx_s = n * bottom_width + m;
        bottom_diff[idx_s] = 0;
        
        for (int y_t = MAX(floor((n - margin) * factor_h) + 1, 0); y_t < MIN((n + margin) * factor_h, top_height); y_t++) {
            for (int x_t = MAX(floor((m - margin) * factor_w) + 1, 0); x_t < MIN((m + margin) * factor_w, top_width); x_t++) {
                
                // diff
                bottom_diff[idx_s] += top_diff[y_t * top_width + x_t] 
                                      * (margin - abs((x_t / factor_w) - m)) * (margin - abs((y_t / factor_h) - n));
            }
        }
        
        bottom_diff[idx_s] /= normalizer;
    }
}





template <typename Dtype>
void InterpolationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();
    
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();



    if (propagate_down[0]) {

        // compute diff for bottom
        const int nthreads = height * width;
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channels; c++) {
                const int index_in = (n * channels + c) * height * width;
                const int index_out = (n * channels + c) * top_height * top_width;
                InflateBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_diff + index_in, height, width, top_diff + index_out, top_height, top_width, factor_value_h_, factor_value_w_);
            }
        }
    }
        
    
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
                   << " Layer cannot backpropagate to the second input.";
    }

    
}

INSTANTIATE_LAYER_GPU_FUNCS(InterpolationLayer);

}  // namespace caffe
