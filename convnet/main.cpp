#include "convnet.h"
using namespace std;
using namespace convnet;

int main(){
	
    string data_path = "/Users/fenghugo/code/data/mnist/";
	Mnist_Parser m(data_path);
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

	n.train(test_x, test_y, 10000);
	//n.test(test_x, test_y);
	getchar();
	return 0;
}