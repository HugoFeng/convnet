cmake_minimum_required( VERSION 2.8 ) 
project( Convnet ) 
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} convnet/ )
add_executable( CNN convnet/main.cpp )
target_link_libraries( CNN ${Boost_LIBRARIES} )

