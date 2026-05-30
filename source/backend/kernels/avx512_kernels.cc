#if defined(__x86_64__) || defined(_M_X64)

#include "source/kernels/kernels.h"
#include <immintrin.h>
#include <cassert>

namespace seecpp::runtime::kernels {

void Gemv(const float* A, const float* x, const float* bias, float* y, 
          size_t m, size_t n) 
{
    // 512 bits / 32 bits per float = 16 floats per vector register.
    // In a production compiler, the Frontend/Middle-end should pad tensors 
    // to multiples of 16 to avoid branching tail-loops in the hot path.
    assert(n % 16 == 0 && "Tensor column count must be padded to a multiple of 16.");

    // Hint to the compiler that our pointers meet the WeightPacker's 64-byte alignment
    const float* aligned_A = static_cast<const float*>(__builtin_assume_aligned(A, 64));
    const float* aligned_x = static_cast<const float*>(__builtin_assume_aligned(x, 64));

    for (size_t i = 0; i < m; ++i) {
        const float* row_A = aligned_A + (i * n);
        
        // Initialize the 512-bit accumulator register to zero
        __m512 sum_vec = _mm512_setzero_ps();

        // Process 16 elements per iteration
        for (size_t j = 0; j < n; j += 16) {
            // Software prefetching: Fetch the next cache line into L1 ahead of time
            _mm_prefetch(reinterpret_cast<const char*>(row_A + j + 16), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(aligned_x + j + 16), _MM_HINT_T0);

            // _mm512_load_ps expects strictly aligned memory boundaries
            __m512 vec_a = _mm512_load_ps(row_A + j);
            __m512 vec_x = _mm512_load_ps(aligned_x + j);

            // Fused Multiply-Add (FMA): sum_vec = (vec_a * vec_x) + sum_vec
            sum_vec = _mm512_fmadd_ps(vec_a, vec_x, sum_vec);
        }

        // Horizontal reduction: sum all 16 lanes of the 512-bit register into a scalar
        float row_sum = _mm512_reduce_add_ps(sum_vec);

        // Apply bias if present, and write to the output buffer
        y[i] = row_sum + (bias ? bias[i] : 0.0f);
    }
}

}  // namespace seecpp::runtime::kernels

#endif  // __x86_64__
