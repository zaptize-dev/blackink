// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Modified for fngine from cinference:src/core/dtype.cpp @ 383e5db. See NOTICE.
#include "core/dtype.h"

#include <stdexcept>

namespace ingine {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return 2;
    case DType::FP32:
        return 4;
    case DType::I32:
        return 4;
    case DType::U8:
        return 1;
    case DType::I64:
        return 8;
    case DType::I8:
        return 1;
    case DType::FP16:
        return 2;
    case DType::FP8_E4M3FN:
        return 1;
    }
    throw std::invalid_argument("invalid DType");
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
    case DType::BF16:
        return "bf16";
    case DType::FP32:
        return "fp32";
    case DType::I32:
        return "i32";
    case DType::U8:
        return "u8";
    case DType::I64:
        return "i64";
    case DType::I8:
        return "i8";
    case DType::FP16:
        return "fp16";
    case DType::FP8_E4M3FN:
        return "fp8_e4m3fn";
    }
    return "invalid";
}

} // namespace ingine
