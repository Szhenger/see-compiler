#ifndef SEECPP_RUNTIME_KERNELS_H_
#define SEECPP_RUNTIME_KERNELS_H_

#include <cstddef>

namespace seecpp::runtime::kernels {

/// @brief Matrix-vector multiplication: y = A * x + bias
/// @pre 'A' and 'x' must be 64-byte aligned (guaranteed by WeightPacker).
/// @pre 'n' (columns) must be a multiple of the vector lane width.
void Gemv(const float* A, const float* x, const float* bias, float* y, 
          size_t m, size_t n);

}  // namespace seecpp::runtime::kernels

#endif  // SEECPP_RUNTIME_KERNELS_H_
