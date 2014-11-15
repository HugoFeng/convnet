#ifndef __JC_UTIL_H__
#define __JC_UTIL_H__

#include <exception>
#include <fstream>
#include <sstream>

#define __CL_ENABLE_EXCEPTIONS
#include "CL/cl.hpp"

namespace jc {

std::string fileToString(const std::string& file_name) {
    std::string file_text;

    std::ifstream file_stream(file_name.c_str());
    if (!file_stream) {
        std::ostringstream oss;
        oss << "There is no file called " << file_name;
        throw std::runtime_error(oss.str());
    }

    file_text.assign(std::istreambuf_iterator<char>(file_stream), std::istreambuf_iterator<char>());

    return file_text;
}

cl::Program buildProgram(const std::string& file_name, const cl::Context& context, const std::vector<cl::Device>& devices)
{
    std::string source_code = jc::fileToString(file_name);
    std::pair<const char *, size_t> source(source_code.c_str(), source_code.size());
    cl::Program::Sources sources;
    sources.push_back(source);
    cl::Program program(context, sources);
    try {
        program.build(devices);
    }
    catch (cl::Error& e) {
        std::string msg;
        program.getBuildInfo<std::string>(devices[0], CL_PROGRAM_BUILD_LOG, &msg);
        std::cerr << "Your kernel failed to compile" << std::endl;
        std::cerr << "-----------------------------" << std::endl;
        std::cerr << msg;
        throw(e);
    }

    return program;
}

cl_ulong runAndTimeKernel(const cl::Kernel& kernel, const cl::CommandQueue& queue, const cl::NDRange global, const cl::NDRange& local=cl::NullRange)
{
    cl_ulong t1, t2;
    cl::Event evt;
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, 0, &evt);
    evt.wait();
    evt.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &t1);
    evt.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &t2);

    return t2 - t1;
}

const char *readable_status(cl_int status)
{
    switch (status) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
#ifndef CL_VERSION_1_0
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: 
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifndef CL_VERSION_1_0
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
#endif
        default:
            return "CL_UNKNOWN_CODE";
    }
}

unsigned int closestMultiple(unsigned int size, unsigned int divisor)
{
    unsigned int remainder = size % divisor;
    return remainder == 0 ? size : size - remainder + divisor;
}

template <class T>
void showMatrix(T *matrix, unsigned int width, unsigned int height)
{
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            std::cout << matrix[width*row + col] << " ";
        }
        std::cout << std::endl;
    }
    return;
}

}

#endif
