#pragma once
#include "util.h"
#include "convolutional_layer.h"
#include "mnist_parser.h"
#include "layer.h"
#include "subsampling_layer.h"
#include "output_layer.h"

namespace convnet{
#define MAX_ITER 100000
#define M 10
#define END_CONDITION 1e-10
	class ConvNet
	{
	public:
		ConvNet(){}
		~ConvNet(){}

		void train(vec2d_t train_x, vec_t train_y, size_t train_size){
			train_x_ = train_x;
			train_y_ = train_y;
			train_size_ = train_size;
			/* 
				auto add OutputLayer as the last layer.
			*/
			layers.back()->next = (new OutputLayer(layers.back()->out_width_,
				layers.back()->out_height_, layers.back()->out_depth_));
			layers.back()->next = nullptr;

			/*
				start training...
			*/
			auto stop = false;
			int iter = 0;
			while (iter < MAX_ITER && !stop){
				iter++;
				auto err = train_once();
				if (err < END_CONDITION) stop = true;
			}
		}

		void test(){

		}

		void add_layer(Layer* layer){
			if (!layers.empty())
				this->layers.back()->next = layer;
			this->layers.push_back(layer);
			layer->next = NULL;
		}

	private:
		float_t train_once(){
			float_t err = 0;
			int iter = 0;
			while (iter < M){
				iter++;
				auto train_x_index = uniform_rand(0, train_size_ - 1);
				layers[0]->input_ = train_x_[train_x_index];
				layers.back()->exp_y = (int)train_y_[train_x_index];
				/*
				Start forward feeding.
				*/
				for (auto layer : layers){
					layer->forward();
					if (layer->next != nullptr){
						layer->next->input_ = layer->output_;
					}
				}
				err += layers.back()->err;
				/*
				back propgation
				*/
				for (auto i = layers.rbegin(); i != layers.rend(); i++){
					(*i)->back_prop();
				}
			}
			return err / M;
		}

		std::vector < Layer* > layers;
		
		size_t train_size_;
		vec2d_t train_x_;
		vec_t train_y_;

		size_t test_size_;
		vec2d_t test_x_;
		vec_t test_y_;
	};
#undef MAX_ITER
#undef M
} //namespace convnet