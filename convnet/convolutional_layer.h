#pragma once
#include "util.h"
#include "layer.h"

namespace convnet{
	class ConvolutionalLayer :public Layer
	{
	public:
		ConvolutionalLayer(size_t in_width, size_t in_height, size_t in_depth, 
			size_t kernel_size, size_t out_depth) :
			Layer(in_width, in_height, in_depth, in_width - kernel_size + 1, in_height - kernel_size + 1, out_depth, 0.3, 0.01),
			kernel_size_(kernel_size)
		{
			W_.resize(kernel_size * kernel_size * in_depth_ * out_depth_);
			deltaW_.resize(kernel_size * kernel_size * in_depth_ * out_depth_);
			b_.resize(out_depth * out_width_* out_height_);
			output_.resize(out_depth * out_width_ * out_height_);
			this->init_weight();
		}

		void init_weight(){
			uniform_rand(W_.begin(), W_.end(), -1, 1);
			uniform_rand(b_.begin(), b_.end(), -1, 1);
		}

		void forward(){
			for (size_t out = 0; out < out_depth_; out++){  /* for each output feature map */
				for (size_t in = 0; in < in_depth_; in++){  /* for each input feature map */
					for (size_t h_ = 0; h_ < out_height_; h_++){
						for (size_t w_ = 0; w_ < out_width_; w_++){
							output_[getOutIndex(out, h_, w_)] +=
								conv(getInforKernel(in, h_, w_), getW_(in, out));
						}
					}
				}
                /* use activate function to get output */
				for (size_t h_ = 0; h_ < out_height_; h_++){
					for (size_t w_ = 0; w_ < out_width_; w_++){
						output_[getOutIndex(out, h_, w_)] =
							sigmod(output_[getOutIndex(out, h_, w_)] + /*eh?*/ b_[getb_(out, h_, w_)]);
					}
				}
			}
		}

		void back_prop(){
			g_.clear();
			g_.resize(in_width_ * in_height_ * in_depth_);
			/*update err terms of this layer.*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					for (size_t w_ = 0; w_ < out_width_; w_++){
						for (size_t h_ = 0; h_ < out_height_; h_++){
							for (size_t y_ = 0; y_ < kernel_size_; y_++){
								for (size_t x_ = 0; x_ < kernel_size_; x_++){
									auto ff = in * in_width_ * in_height_ + (h_ + y_) *
										in_width_ + (x_ + w_);
									g_[ff] += /*next layer err terms*/
										this->next->g_[out * out_width_ *
										out_height_ + h_ * out_width_ + w_] * 
										/*weight*/
										W_[in * out_depth_ * kernel_size_ * kernel_size_ +
                                           out * kernel_size_ * kernel_size_ +
                                           kernel_size_ * (kernel_size_ - y_ - 1) +
                                           (kernel_size_ - 1 - x_)] *
										/*df of input*/
										df_sigmod(input_[ff]);
								}
							}
						}
					}
				}
			}

			/*update weight*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					for (size_t h_ = 0; h_ < out_height_; h_++){
						for (size_t w_ = 0; w_ < out_height_; w_++){
							auto tt = getb_(out, h_, w_);
							for (size_t y_ = 0; y_ < kernel_size_; y_++){
								for (size_t x_ = 0; x_ < kernel_size_; x_++){
									/*find update pixel*/
									auto target = in * out_depth_ * kernel_size_ * kernel_size_ +
										out * kernel_size_ * kernel_size_ +
										kernel_size_ * (kernel_size_ - y_ - 1) +
										(kernel_size_ - 1 - x_);
									/*cal delta*/
									auto delta =
                                        /*learning rate*/
										alpha_ *
										/*input*/
										input_[in * in_width_ * in_height_ + (h_ + y_) *
                                               in_width_ + (x_ + w_)] *
										/*next layer err terms*/
										this->next->g_[tt]
										/*weight momentum*/
										+ lambda_ * deltaW_[target];
										
                                    W_[target] += delta;
                                    /*update momentum*/
                                    deltaW_[target] = delta;
								}
							}
							b_[tt] += alpha_ * this->next->g_[tt];
						}
					}
				}
			}
		}

	private:
		inline size_t getOutIndex(size_t out, size_t h_, size_t w_){
			return out * out_height_ * out_width_ + h_ * out_width_ + w_;
		}

		inline vec_t getInforKernel(size_t in, size_t h_, size_t w_){
			vec_t r;
			for (size_t x = 0; x < kernel_size_; x++){
				for (size_t y = 0; y < kernel_size_; y++){
					r.push_back(input_[in * (in_width_ * in_height_) + (h_ + y) * in_width_ + x + w_]);
				}
			}
			return r;
		}

		inline vec_t getW_(size_t in, size_t out){
			vec_t r;
			for (size_t i = 0; i < kernel_size_ * kernel_size_; i++)
				r.push_back(W_[in * out_depth_ * kernel_size_ * kernel_size_ 
				+ out * kernel_size_ * kernel_size_ + i]);
			return r;
		}

		inline int getb_(size_t out, size_t h_, size_t w_){
			return out * out_width_ * out_height_ + h_ * out_height_ + w_;
		}

		/*
		2-dimension convoluton:

			1 2 3                    1 -1 0
			3 4 2  conv with kernel  -1 0 1  
			2 1 3                    1  1 0

			---->
			1*0 + 2*1 + 3*1 + 3*1 + 4*0 + 2*-1 + 2*0 + 1*-1 + 3*1
			return the sum.

		see also:
		*/
		float_t conv(vec_t a, vec_t b){
			assert(a.size() == b.size());
			float_t sum = 0, size = a.size();
			for (size_t i = 0; i < size; i++){
				sum += a[i] * b[size - i - 1];
			}
			return sum;
		}

		size_t kernel_size_;
	};
}// namespace convnet