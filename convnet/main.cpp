#include "convnet.h"



#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>

#include <JC/util.hpp>


using namespace std;
using namespace convnet;

int main(){
	Mnist_Parser m(DATA_PATH);
	m.load_testing();
	//m.load_training();
	vec2d_t x;
	vec_t y;
	vec2d_t test_x;
	vec_t test_y;
	/*
	for (size_t i = 0; i < 60000; i++){
	x.push_back(m.train_sample[i]->image);
	y.push_back(m.train_sample[i]->label);
	}
	
	*/

	for (size_t i = 0; i < 10000; i++){
		test_x.push_back(m.test_sample[i]->image);
		test_y.push_back(m.test_sample[i]->label);
	}
	
	ConvNet n;

	n.add_layer(new ConvolutionalLayer(32, 32, 1, 5, 6));
	n.add_layer(new MaxpoolingLayer(28, 28, 6));
	n.add_layer(new ConvolutionalLayer(14, 14, 6, 5, 16));
	n.add_layer(new MaxpoolingLayer(10, 10, 16));
	n.add_layer(new ConvolutionalLayer(5, 5, 16, 5, 100));
	n.add_layer(new FullyConnectedLayer(100, 10));

#ifdef GPU
    cout << "Using GPU:" << endl;
#else
    cout << "Using CPU:" << endl;
#endif

	n.train(test_x, test_y, 10000);
    int test_sample_count = 1;
    printf("Testing with %d samples:\n", test_sample_count);
    n.test(test_x, test_y, test_sample_count, 1);
	return 0;
}