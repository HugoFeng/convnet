#pragma once

#include "layer.h"
#include "util.h"

namespace convnet{
	class OutputLayer :public Layer
	{
	public:
		OutputLayer(size_t in_depth):
			Layer(1, 1, in_depth, 0, 0, 0, 0, 0)
		{}

		void forward_cpu(){
			this->err = 0;
			exp_y_vec.clear();
			exp_y_vec.resize(in_depth_);
			exp_y_vec[this->exp_y] = 1;
			for (size_t i = 0; i < in_depth_; i++){
				err += 0.5 * (exp_y_vec[i] - input_[i]) *
					(exp_y_vec[i] - input_[i]);
			}
			output_ = input_;
		}

        void forward_batch(int batch_size){
            this->err = 0;
            exp_y_vec_batch.clear();
            exp_y_vec_batch.resize(batch_size*in_depth_);
            for (size_t batch = 0; batch < batch_size; batch++){
                exp_y_vec_batch[batch*in_depth_ + this->exp_y_batch[batch]] = 1;
                for (size_t i = 0; i < in_depth_; i++){
                    err += 0.5 * (exp_y_vec_batch[batch*in_depth_ + i] - input_batch_[batch*in_depth_ + i]) *
                        (exp_y_vec_batch[batch*in_depth_ + i] - input_batch_[batch*in_depth_ + i]);
                }
            }
            err = err / batch_size;
            output_batch_ = input_batch_;
        }

		void back_prop(){
			/* compute err terms of output layers */
			g_.clear();
			
			for (size_t i = 0; i < in_depth_; i++){
				g_.push_back((exp_y_vec[i] - input_[i]) * df_sigmod(input_[i]));
				
			}
		}

		void init_weight(){}

	private:
	};
} // namespace convnet
