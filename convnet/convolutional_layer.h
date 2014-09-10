#include "util.h"
#include "layer.h"

namespace convnet{
	class ConvolutionalLayer :public Layer
	{
	public:
		ConvolutionalLayer(size_t in_width, size_t in_height, size_t in_depth, 
			size_t kernel_size, size_t out_depth) :
			Layer(in_width, in_height, in_depth, in_width - kernel_size + 1, in_height - kernel_size + 1, out_depth),
			kernel_size_(kernel_size)
		{
			W_.resize(kernel_size * kernel_size * in_depth_ * out_depth_);
			b_.resize(out_depth);
			output_.resize(out_depth * out_width_ * out_height_);
			this->init_weight();
		}

		void init_weight(){
			uniform_rand(W_.begin(), W_.end(), -1, 1);
			uniform_rand(b_.begin(), b_.end(), -1, 1);
		}

		void forward(){
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					for (size_t h_ = 0; h_ < out_height_; h_++){
						for (size_t w_ = 0; w_ < out_width_; w_++){
							output_[getOutIndex(out, h_, w_)] +=
								conv(getInforKernel(in, h_, w_), get_W_(in, out));
						}
					}
				}
				for (size_t h_ = 0; h_ < out_height_; h_++){
					for (size_t w_ = 0; w_ < out_width_; w_++){
						output_[getOutIndex(out, h_, w_)] =
							sigmod(output_[getOutIndex(out, h_, w_)] + /*eh?*/ b_[out]);
					}
				}
			}
		}

		void back_prop(){

		}

	private:
		inline size_t getOutIndex(size_t out, size_t h_, size_t w_){
			return out * out_height_ * out_width_ + h_ * out_width_ + w_;
		}

		inline vec_t getInforKernel(size_t in, size_t h_, size_t w_){
			vec_t r;
			for (size_t h = 0; h < h_; h++){
				for (size_t w = 0; w < w_; w++){
					r.push_back(input_[in * (in_width_ * in_height_) + h_ * in_width_ + w]);
				}
			}
			return r;
		}

		inline vec_t get_W_(size_t in, size_t out){
			vec_t r;
			for (size_t i = 0; i < kernel_size_ * kernel_size_; i++)
				r.push_back(W_[in * out_depth_ + out + i]);
			return r;
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