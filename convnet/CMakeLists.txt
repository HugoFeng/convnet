project( Convnet )
add_executable( CNN ./main.cpp 
                    ./convnet.h 
                    ./convolutional_layer.h
                    ./fullyconnected_layer.h
                    ./layer.h
                    ./maxpooling_layer.h
                    ./mnist_parser.h
                    ./output_layer.h
                    ./util.h )

