/*****************************************************************************
Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

   1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
   3. Neither the name of the OpenBLAS project nor the names of 
      its contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

**********************************************************************************/

#include <stdio.h>
#include "common.h"

void zfill_oncopy(int k, int n, IFLOAT *a, int lda, IFLOAT *b) {
  int whole_n = n - n % GEMM_UNROLL_N;
  GEMM_ONCOPY(k, whole_n, a, lda, b);

  b += whole_n * k;
  n -= whole_n;

  IFLOAT *a_off = a, *b_off = b;
  while (k > 0) {
    for (int i = 0; i < GEMM_UNROLL_N; ++i)
      b_off[i] = (i < n) ? a_off[lda * i] : 0;

    ++a_off;
    b_off += GEMM_UNROLL_N;
    --k;
  }
}

void CNAME(IFLOAT *image, blasint N, blasint H, blasint W, blasint C,
           IFLOAT *filter, blasint FH, blasint FW, blasint M,
           IFLOAT *output, blasint PH, blasint PW,
           IFLOAT *sa, IFLOAT *sb) {

  const int l2_size = GEMM_P * GEMM_Q;
  const int l3_size = GEMM_Q * GEMM_R;
  const int Q = MIN(FW * C, GEMM_Q);
  const int P = l2_size / Q / GEMM_UNROLL_M * GEMM_UNROLL_M;
  const int R = l3_size / W / C;

  const int OH = H + 2 * PH - FH + 1;
  const int OW = W + 2 * PW - FW + 1;

  GEMM_BETA(M, OH * OW, 0, 0, NULL, 0, NULL, 0, output, M);

  for (int js = 0; js < H; js += R) {
    int min_j = MIN(H - js, R);

    zfill_oncopy(W * C, min_j, image + js * W * C, W * C, sb);

    for (int fh = 0; fh < FH; ++fh) {

      for (int m = 0; m < M; m += P) {
        int min_m = MIN(M - m, P);

        for (int k = 0; k < FW * C; k += Q) {
          int min_k = MIN(FW * C - k, Q);

          GEMM_ITCOPY(min_k, min_m, filter + fh * FW * C * M + k * M + m, M, sa);

          for (int jjs = 0, min_jj; jjs < min_j; jjs += min_jj) {
            min_jj = min_j - jjs;
#if defined(SKYLAKEX) || defined(COOPERLAKE) || defined(SAPPHIRERAPIDS)
            /* the current AVX512 s/d/c/z GEMM kernel requires n>=6*GEMM_UNROLL_N to achieve best performance */
            if (min_jj >= 6*GEMM_UNROLL_N) min_jj = 6*GEMM_UNROLL_N;
#else
            if (min_jj > 2*GEMM_UNROLL_N) min_jj = 3*GEMM_UNROLL_N;
            else if (min_jj > GEMM_UNROLL_N) min_jj = 2*GEMM_UNROLL_N;
            else min_jj = GEMM_UNROLL_N;
#endif

            for (int ow = 0; ow < OW; ++ow) {
              int image_start = (ow - PW) * C + k;
              int image_end = MIN(W * C, image_start + min_k);

              IFLOAT *ar = sa;
              if (image_start < 0)
              {
                ar -= image_start * ((min_m - 1) % GEMM_UNROLL_M + 1);
                image_start = 0;
              }

              int K = image_end - image_start;
              if (K <= 0)
                continue;

              IFLOAT *br = sb + jjs * W * C + image_start * min_jj;
              IFLOAT *cr = output + ((js + jjs - fh + PH) * OW + ow) * M + m;

              GEMM_KERNEL_N(min_m, min_jj, K, 1, ar, br, cr, OW * M);
            }
          }
        }
      }
    }
  }
}
