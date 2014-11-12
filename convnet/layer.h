#include "util.h"

#pragma once
namespace convnet{
	class Layer
	{
	public:
		Layer(size_t in_width, size_t in_height, size_t in_depth,
			size_t out_width, size_t out_height, size_t out_depth, float_t alpha, float_t lambda) :
			in_width_(in_width), in_height_(in_height), in_depth_(in_depth),
			out_width_(out_width), out_height_(out_height), out_depth_(out_depth),
			alpha_(alpha), lambda_(lambda)
		{}

		virtual void init_weight() = 0;
		virtual void forward_cpu() = 0;
        virtual void forward_batch(int batch_size) = 0;
		virtual void back_prop() = 0;

        void forward_gpu(){ forward_cpu(); }

		float_t sigmod(float_t in){
			return 1.0 / (1.0 + std::exp(-in));
		}

		float_t df_sigmod(float_t f_x) {
			return f_x * (1.0 - f_x);
		}

		size_t fan_in(){
			return in_width_ * in_height_ * in_depth_;
		}

		size_t fan_out(){
			return out_width_ * out_height_ * out_height_;
		}

		size_t in_width_;
		size_t in_height_;
		size_t in_depth_;

		size_t out_width_;
		size_t out_height_;
		size_t out_depth_;

		vec_t W_;
		vec_t b_;

		vec_t deltaW_;

		vec_t input_;
		vec_t output_;

        vec_t input_batch_;
        vec_t output_batch_;

		Layer* next;

		float_t alpha_; // learning rate
		float_t lambda_; // momentum
		vec_t g_; // err terms

		/*output*/
		float_t err;
		int exp_y;
		vec_t exp_y_vec;

        vec_t exp_y_batch;
        vec_t exp_y_vec_batch;
	};
}