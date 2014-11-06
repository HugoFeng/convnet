#pragma once
#include "util.h"
#include "convolutional_layer.h"
#include "mnist_parser.h"
#include "layer.h"
#include "maxpooling_layer.h"
#include "output_layer.h"
#include "mnist_parser.h"
#include "fullyconnected_layer.h"

namespace convnet{
#ifndef DEBUG
    #define MAX_ITER 100000
#else
    #define MAX_ITER 200
#endif
#define M 10
#define END_CONDITION 1e-3
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
			this ->add_layer(new OutputLayer(layers.back()->out_depth_));

			/*
				start training...
			*/
			auto stop = false;
			int iter = 0;
			while (iter < MAX_ITER && !stop){
				iter++;
				auto err = train_once();
				std::cout << "err: " << err << std::endl;
				if (err < END_CONDITION) stop = true;
			}
		}

		void test(vec2d_t test_x, vec_t test_y, size_t test_size){
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;
			while (iter < test_size_){
				iter++;
				if(test_once()) bang ++;
			}
			std::cout << "bang/test_size_: "<< (float)bang / test_size_ << std::endl;
		}

		void add_layer(Layer* layer){
			if (!layers.empty())
				this->layers.back()->next = layer;
			this->layers.push_back(layer);
			layer->next = NULL;
		}

	private:
		size_t max_iter(vec_t v){
			size_t i = 0;
			float_t max = v[0];
			for (size_t j = 1; j < v.size(); j++){
				if (v[j] > max){
					max = v[j];
					i = j;
				}
			}
			return i;
		}

		bool test_once(){
			auto test_x_index = uniform_rand(0, test_size_ - 1);
			layers[0]->input_ = test_x_[test_x_index];
			for (auto layer : layers){
				layer->forward();
				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}
			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
		}

		float_t train_once(){
			float_t err = 0;
			int iter = 0;
			while (iter < M){
                auto train_x_index = iter % train_size_;
				iter++;
				//auto train_x_index = uniform_rand(0, train_size_ - 1);
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