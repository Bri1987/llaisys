// 引入 llaisys 的头文件
#include "../runtime_api.hpp"

#ifdef ENABLE_NVIDIA_API

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <cstring>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            char _cuda_err_msg[512];                                        \
            snprintf(_cuda_err_msg, sizeof(_cuda_err_msg),                 \
                     "CUDA Error at %s:%d, code=%d, reason: %s",          \
                     __FILE__, __LINE__, static_cast<int>(err),            \
                     cudaGetErrorString(err));                             \
            throw std::runtime_error(_cuda_err_msg);                        \
        }                                                                   \
    } while (0)


namespace llaisys::device::nvidia {

namespace runtime_api {

static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
        case LLAISYS_MEMCPY_H2D:
            return cudaMemcpyHostToDevice;
        case LLAISYS_MEMCPY_D2H:
            return cudaMemcpyDeviceToHost;
        case LLAISYS_MEMCPY_D2D:
            return cudaMemcpyDeviceToDevice;
        default:
            char _msg_buf[256];
            snprintf(_msg_buf, sizeof(_msg_buf), "Unknown or unsupported llaisysMemcpyKind_t: %d", (int)kind);
            throw std::invalid_argument(_msg_buf);
    }
}

// 【已修正】重写了整个函数以修复逻辑错误
int getDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

    // 首先，处理可以接受的“错误”，例如没有找到设备
    if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) {
        return 0;
    }

    // 其次，对于任何其他不是“成功”的返回码，抛出异常以便调用方处理
    if (err != cudaSuccess) {
        char _msg_buf[512];
        snprintf(_msg_buf, sizeof(_msg_buf), "CUDA Error at %s:%d, code=%d, reason: %s",
                 __FILE__, __LINE__, static_cast<int>(err), cudaGetErrorString(err));
        throw std::runtime_error(_msg_buf);
    }
    
    return count;
}

void setDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaStreamDestroy(cuda_stream));
}

void streamSynchronize(llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), cuda_stream));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia

#endif // ENABLE_NVIDIA_API