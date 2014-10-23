cmake_minimum_required( VERSION 2.8 ) 
project( Convnet ) 
find_package( Boost REQUIRED )
include_directories( ${Boost_INCLUDE_DIRS} )
add_subdirectory( convnet )
target_link_libraries( CNN ${Boost_LIBRARIES} )

