#include <numeric>
#include <unordered_map>

#include "util.h"
#include "layer.h"

#pragma once
namespace convnet{
	class MaxpoolingLayer :public Layer
	{
	public:
		MaxpoolingLayer(size_t in_width, size_t in_height, size_t in_depth) :
			Layer(in_width, in_height, in_depth, in_width / 2, in_height / 2, in_depth, 0, 0)
		{
			output_.resize(out_depth_ * out_width_ * out_height_);
		}

		void forward_cpu(){
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t h_ = 0; h_ < in_height_; h_+= 2){
					for (size_t w_ = 0; w_ < in_width_; w_+= 2){
						output_[getOutIndex(out, h_, w_)] = max_In_(out, h_, w_, 
							getOutIndex(out, h_, w_));
					}
				}
			}
		}

        void forward_batch(int batch_size){
            output_batch_.resize(batch_size*out_depth_ * out_width_ * out_height_);
            for (size_t batch = 0; batch < batch_size; batch++){
                for (size_t out = 0; out < out_depth_; out++){
                    for (size_t h_ = 0; h_ < in_height_; h_ += 2){
                        for (size_t w_ = 0; w_ < in_width_; w_ += 2){
                            output_batch_[getOutIndex_batch(batch, out, h_, w_)] = max_In_batch_(batch, out, h_, w_);
                        }
                    }
                }
            }
        }
		/*
		 In forward propagation, k¡Ák blocks are reduced to a single value. 
		 Then, this single value acquires an error computed from backwards 
		 propagation from the previous layer. 
		 This error is then just forwarded to the place where it came from. 
		 Since it only came from one place in the k¡Ák block, 
		 the backpropagated errors from max-pooling layers are rather sparse.
		*/
		void back_prop(){
			g_.clear();
			g_.resize(in_width_ * in_height_ * in_depth_);
			for (auto pair : max_loc)
				g_[pair.second] = this->next->g_[pair.first];
		}

		void init_weight(){}

	//private:
		inline float_t max_In_(size_t in_index, size_t h_, size_t w_, size_t out_index){
			float_t max_pixel = 0;
			size_t tmp;
			for (size_t x = 0; x < 2; x++){
				for (size_t y = 0; y < 2; y++){
					tmp = (in_index * in_width_ * in_height_) +
						((h_ + y) * in_width_) + (w_ + x);
					if (max_pixel < input_[tmp]){
						max_pixel = input_[tmp];
						max_loc[out_index] = tmp;
					}
				}
			}
			return max_pixel;
		}

        inline float_t max_In_batch_(size_t batch, size_t in_index, size_t h_, size_t w_){
            float_t max_pixel = 0;
            size_t tmp;
            for (size_t x = 0; x < 2; x++){
                for (size_t y = 0; y < 2; y++){
                    tmp = (batch*in_depth_*in_width_*in_height_) + (in_index * in_width_ * in_height_) +
                        ((h_ + y) * in_width_) + (w_ + x);
                    if (max_pixel < input_batch_[tmp]){
                        max_pixel = input_batch_[tmp];
                    }
                }
            }
            return max_pixel;
        }

		inline size_t getOutIndex(size_t out, size_t h_, size_t w_){
			return out * out_width_ * out_height_ +
				h_ / 2 * out_width_ + (w_ / 2);
		}

        inline size_t getOutIndex_batch(size_t batch, size_t out, size_t h_, size_t w_){
            return batch*out_depth_*out_width_*out_height_ + out * out_width_ * out_height_ +
                h_ / 2 * out_width_ + (w_ / 2);
        }

		/*
		for each output, I store the connection index of the input,
		which will be used in the back propagation,
		for err translating.
		*/
		std::unordered_map<size_t, size_t> max_loc;
	};
}//namespace convnet