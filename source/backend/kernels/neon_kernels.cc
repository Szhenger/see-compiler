#if defined(__aarch64__) || defined(_M_ARM64)

#include "source/kernels/kernels.h"
#include <arm_neon.h>
#include <cassert>

namespace seecpp::runtime::kernels {

void Gemv(const float* A, const float* x, const float* bias, float* y, 
          size_t m, size_t n) 
{
    // 128 bits / 32 bits per float = 4 floats per vector register.
    assert(n % 4 == 0 && "Tensor column count must be padded to a multiple of 4.");

    for (size_t i = 0; i < m; ++i) {
        const float* row_A = A + (i * n);
        
        // Initialize the 128-bit accumulator register to zero
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        // To mimic the throughput of AVX-512, we unroll the loop by 4 (processing 16 floats)
        // This hides instruction latency and maximizes the NEON pipeline usage.
        size_t j = 0;
        for (; j + 15 < n; j += 16) {
            // Clang/GCC usually handle ARM prefetching well, but we explicitly unroll
            float32x4_t a0 = vld1q_f32(row_A + j);
            float32x4_t x0 = vld1q_f32(x + j);
            sum_vec = vfmaq_f32(sum_vec, a0, x0);

            float32x4_t a1 = vld1q_f32(row_A + j + 4);
            float32x4_t x1 = vld1q_f32(x + j + 4);
            sum_vec = vfmaq_f32(sum_vec, a1, x1);

            float32x4_t a2 = vld1q_f32(row_A + j + 8);
            float32x4_t x2 = vld1q_f32(x + j + 8);
            sum_vec = vfmaq_f32(sum_vec, a2, x2);

            float32x4_t a3 = vld1q_f32(row_A + j + 12);
            float32x4_t x3 = vld1q_f32(x + j + 12);
            sum_vec = vfmaq_f32(sum_vec, a3, x3);
        }

        // Clean up remaining multiples of 4 (if 'n' isn't a multiple of 16)
        for (; j < n; j += 4) {
            float32x4_t a_rem = vld1q_f32(row_A + j);
            float32x4_t x_rem = vld1q_f32(x + j);
            sum_vec = vfmaq_f32(sum_vec, a_rem, x_rem);
        }

        // Horizontal reduction: sum the 4 lanes into a single scalar
        float row_sum = vaddvq_f32(sum_vec);

        // Apply bias and write output
        y[i] = row_sum + (bias ? bias[i] : 0.0f);
    }
}

}  // namespace seecpp::runtime::kernels

#endif  // __aarch64__
