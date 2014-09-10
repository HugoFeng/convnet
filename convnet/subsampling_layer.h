#include <numeric>

#include "util.h"
#include "layer.h"

#pragma once
namespace convnet{
	class SubsamplingLayer :public Layer
	{
	public:
		SubsamplingLayer(size_t in_width, size_t in_height, size_t in_depth) :
			Layer(in_width, in_height, in_depth, in_width / 2, in_height / 2, in_depth)
		{
			output_.resize(out_depth_ * out_width_ * out_height_);
		}

		void forward(){
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t h_ = 0; h_ < in_height_; h_+= 2){
					for (size_t w_ = 0; w_ < in_width_; w_+= 2){
						output_[out * out_width_ * out_height_ + 
							h_ / 2 * out_width_ + (w_ / 2)] = average_In_(out, h_, w_);
					}
				}
			}
		}

		void back_prop(){

		}

	private:
		inline float_t average_In_(size_t in_index, size_t h_, size_t w_){
			float_t sum = 0;
			for (size_t x = 0; x < 2; x++){
				for (size_t y = 0; y < 2; y++){
					sum += input_[(in_index * in_width_ * in_height_) +
						((h_ + y) * in_width_) + (w_ + x)];
				}
			}
			return sum / 4;
		}
	};
}//namespace convnet