#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    size_t ndim = this->ndim();

    // 0维张量（标量）或1维张量总是连续的。
    if (ndim <= 1) {
        return true;
    }

    // 预期步长，从最后一个维度开始计算。
    // 最后一个维度的预期步长总是 1。
    size_t expected_stride = 1;

    // 从后往前遍历所有维度 (i 从 ndim-1 到 0)。
    for (int i = ndim - 1; i >= 0; --i) {
        // 对于维度大小为1的情况，其步长不影响连续性，可以跳过检查。
        if (this->shape()[i] > 1) {
            // 检查当前维度的实际步长是否等于预期步长。
            // 使用 static_cast 来解决 signed/unsigned 比较警告。
            if (static_cast<size_t>(this->strides()[i]) != expected_stride) {
                return false;
            }
        }
        
        // 更新下一个（更前一个）维度所期望的步长。
        expected_stride *= this->shape()[i];
    }

    // 如果所有维度的步长都符合预期，则张量是连续的。
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const size_t ndim = this->ndim();
    if (order.size() != ndim) {
        throw std::invalid_argument("The number of dimensions to permute must be the same as the number of dimensions of the tensor.");
    }
    
    std::vector<bool> seen_dims(ndim, false);
    for (size_t i = 0; i < ndim; ++i) {
        if (order[i] >= ndim || seen_dims[order[i]]) {
            throw std::invalid_argument("Invalid permutation order provided.");
        }
        seen_dims[order[i]] = true;
    }

    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape.resize(ndim);
    new_meta.strides.resize(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        new_meta.shape[i] = this->shape()[order[i]];
        new_meta.strides[i] = this->strides()[order[i]];
    }
    
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1. 前提条件检查：view 操作要求张量必须是连续的。
    if (!this->isContiguous()) {
        // 使用标准的 std::invalid_argument 替代自定义宏
        throw std::invalid_argument("view requires the tensor to be contiguous. "
                                    "Use .contiguous() to create a contiguous copy first.");
    }

    // 2. 验证新旧形状的元素总数是否一致。
    const size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        // 同样使用 std::invalid_argument
        // 注意：为了拼接字符串，需要包含 <string>
        throw std::invalid_argument("Shape is invalid for input of size " + std::to_string(this->numel()));
    }

    // 3. 为新视图计算新的元数据。
    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    new_meta.shape = shape;

    // 4. 计算新的步长（strides）
    size_t ndim = shape.size();
    new_meta.strides.resize(ndim);
    size_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        new_meta.strides[i] = stride;
        if (shape[i] != 0) {
            stride *= shape[i];
        }
    }

    // 5. 创建并返回一个新的 Tensor 对象。
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // --- 参数验证部分 (已在上一段提供) ---
    const size_t ndim = this->ndim();
    if (dim >= ndim) {
        throw std::invalid_argument("Dimension out of range. Got dim " + std::to_string(dim) + 
                                    " but tensor has " + std::to_string(ndim) + " dimensions.");
    }
    if (start > end || end > this->shape()[dim]) {
        throw std::invalid_argument("Slice indices out of range for dimension " + std::to_string(dim) +
                                    ". Expected start <= end and end <= " + std::to_string(this->shape()[dim]) +
                                    ", but got start=" + std::to_string(start) + " and end=" + std::to_string(end));
    }

    // --- 元数据和偏移量计算 ---
    TensorMeta new_meta;
    new_meta.dtype = this->dtype();
    
    // 新的 shape 与旧的几乎一样，只是在被切片的维度上改变了大小。
    new_meta.shape = this->shape();
    new_meta.shape[dim] = end - start;

    // strides 保持不变，因为元素之间的相对距离没有改变。
    new_meta.strides = this->strides();

    // 计算新的偏移量。这是 slice 操作的关键。
    // 新的偏移量 = 旧的字节偏移量 + (start * 该维度的步长 * 单个元素大小)
    size_t new_offset = this->_offset + start * this->strides()[dim] * this->elementSize();

    // 创建并返回一个新的 Tensor 对象。
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 1. 计算需要复制的总字节数。
    //    元素总数 (numel) * 单个元素的字节大小 (elementSize)。
    const size_t size_in_bytes = this->numel() * this->elementSize();

    // 如果没有数据需要加载，则直接返回。
    if (size_in_bytes == 0 || src_ == nullptr) {
        return;
    }

    // 2. 获取目标地址，即当前张量数据的起始指针。
    //    this->data() 会正确处理 storage 和 offset。
    std::byte *dst = this->data();

    // 3. 设置当前的设备上下文，确保操作在正确的设备上执行。
    core::context().setDevice(this->deviceType(), this->deviceId());

    // 4. 调用运行时API执行内存复制。
    //    因为源数据`src`在主机(Host)上，所以使用 LLAISYS_MEMCPY_H2D (Host to Device) 类型。
    //    如果张量本身就在CPU上，这个API内部会处理为一次常规的CPU内存复制。
    core::context().runtime().api()->memcpy_sync(
        dst,             // 目标地址 (可能在设备上)
        src_,             // 源地址 (在主机上)
        size_in_bytes,   // 复制的字节数
        LLAISYS_MEMCPY_H2D // 复制方向：主机到设备
    );
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
