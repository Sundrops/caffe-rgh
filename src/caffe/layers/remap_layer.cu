#include <vector>
#include <iostream>
#include "caffe/layers/remap_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {







template <typename Dtype>
__global__ void RemapForwardGPU(const int nthreads,
          const Dtype* Vb,
          const Dtype* coords,
          Dtype *Vt, 
          const int H1, const int W1) {
          
    CUDA_KERNEL_LOOP(index, nthreads) {
    
        const int h = index / W1;
        const int w = index % W1;
        
        float x, y, wx, wy, w00, w01, w10, w11, v00, v01, v10, v11;
        int x0, y0, x1, y1;
        
        x = w + coords[h*W1+w];
		y = h + coords[(H1+h)*W1+w];  // from 1 to 2
										
		x0 = floor(x);
		y0 = floor(y);
		x1 = x0 + 1;
		y1 = y0 + 1;
		wx = x - x0;
		wy = y - y0;
		w00 = (1 - wx) * (1 - wy);
		w01 = (1 - wx) * wy;
		w10 = wx * (1 - wy);
		w11 = wx * wy;
		
		v00 = (x0 < 0 || x0 > W1 - 1 || y0 < 0 || y0 > H1 - 1) ? 0 : Vb[y0*W1+x0];	
		v01 = (x0 < 0 || x0 > W1 - 1 || y1 < 0 || y1 > H1 - 1) ? 0 : Vb[y1*W1+x0];		
		v10 = (x1 < 0 || x1 > W1 - 1 || y0 < 0 || y0 > H1 - 1) ? 0 : Vb[y0*W1+x1];		
		v11 = (x1 < 0 || x1 > W1 - 1 || y1 < 0 || y1 > H1 - 1) ? 0 : Vb[y1*W1+x1];
		
		Vt[h*W1+w] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;

    }
}



template <typename Dtype>
void RemapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    
	const Dtype* Vb = bottom[0]->gpu_data(); // image2
	const Dtype* coords = bottom[1]->gpu_data();
	Dtype* Vt = top[0]->mutable_gpu_data();
	int N  = bottom[0]->num();
	int C  = bottom[0]->channels();
	int W0 = bottom[0]->width();
	int H0 = bottom[0]->height();
	int W1 = bottom[1]->width();
	int H1 = bottom[1]->height();
	
	CHECK_EQ(W0, W1);
	CHECK_EQ(H0, H1);
	
	const int nthreads = H1 * W1;
	for ( int n = 0; n < N; n++ ) {
		for ( int c = 0; c < C; c++ ) {
		
		    const int idx_map = (n * C + c) * H1 * W1;
		    const int idx_coord = (n * 2) * H1 * W1;
		    
		    RemapForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, Vb + idx_map, coords + idx_coord,  Vt + idx_map, H1, W1);

		}
	} 


}







template <typename Dtype>
void RemapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


    Backward_cpu(top, propagate_down, bottom);
   
   
}



INSTANTIATE_LAYER_GPU_FUNCS(RemapLayer);


}  // namespace caffe
