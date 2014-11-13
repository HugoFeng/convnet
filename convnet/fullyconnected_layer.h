#pragma once
#include "layer.h"
#include "util.h"

namespace convnet{
	class FullyConnectedLayer :public Layer
	{
	public:
		FullyConnectedLayer(size_t in_depth, size_t out_depth):
			Layer(1, 1, in_depth, 1, 1, out_depth, 0.3, 0.01)
		{
			output_.resize(out_depth_);
			W_.resize(in_depth_ * out_depth_);
			deltaW_.resize(in_depth_ * out_depth_);
			b_.resize(out_depth_);
			g_.resize(in_depth_);

			this->init_weight();
		}
		
		void forward_cpu(){
			for (size_t out = 0; out < out_depth_; out++){
				output_[out] = sigmod(dot(input_, get_W(out)) + b_[out]);
			}
		}

        void forward_batch(int batch_size){
            output_batch_.resize(batch_size*out_depth_ * out_width_ * out_height_);
            for (size_t batch = 0; batch < batch_size; batch++){
                for (size_t out = 0; out < out_depth_; out++){
                    output_batch_[batch*out_depth_ + out] = sigmod(dot_per_batch(batch, input_batch_, get_W(out)) + b_[out]);
                }
            }
        }

		void back_prop(){
			/*
			Compute the err terms;
			*/
			for (size_t in = 0; in < in_depth_; in++){
				g_[in] = df_sigmod(input_[in]) * dot(this->next->g_, get_W_step(in));
			}
			/*
			Update weights.
			*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					auto delta = alpha_/*learning rate*/
						* input_[in] * this->next->g_[out]/*err terms*/
						/*+ lambda_ weight decay*/
						+ lambda_ * deltaW_[out * in_depth_ + in];
					W_[out * in_depth_ + in] += delta;
					deltaW_[out * in_depth_ + in] = delta;
				}
				b_[out] += alpha_ * this->next->g_[out];
			}
		}

		/*
		for the activation sigmod,
		weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
		see also:http://deeplearning.net/tutorial/references.html#xavier10
		*/
		void init_weight(){
			/*
			uniform_rand(W_.begin(), W_.end(),
				-4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
				4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
			uniform_rand(b_.begin(), b_.end(),
				-4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
				4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
			*/
			uniform_rand(W_.begin(), W_.end(), -2, 2);
			uniform_rand(b_.begin(), b_.end(), -2, 2);
			
		}
	private:
		vec_t get_W(size_t index){
			vec_t v;
			for (int i = 0; i < in_depth_; i++){
				v.push_back(W_[index * in_depth_ + i]);
			}
			return v;
		}

		vec_t get_W_step(size_t in){
			vec_t r;
			for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_){
				r.push_back(W_[i]);
			}
			return r;
		}
	};
} // namespace convnet
