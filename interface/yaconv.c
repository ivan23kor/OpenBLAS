/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "common.h"

void edge_oncopy(blasint k, blasint n, float *b, blasint ldb, float *sb) {
  blasint i;
  blasint whole_n = n - n % GEMM_UNROLL_N;

  GEMM_ONCOPY(k, whole_n, b, ldb, sb);

  b += whole_n * ldb;
  sb += whole_n * k;
  n -= whole_n;

  while (k > 0) {
    i = 0;
    while (i++ < n) {
      *(sb++) = *b;
      b += ldb;
    }
    while (i++ < GEMM_UNROLL_N)
      *(sb++) = 0.0;
    b += 1 - n * ldb;
    --k;
  }
}

blasint yaconv_extra_size_before(blasint fh, blasint ph, blasint ow,
                                 blasint m) {
  return (fh - 1 - ph) * ow * m;
}

static inline blasint yaconv_extra_size_after(blasint h, blasint fh,
                                              blasint ph, blasint ow,
                                              blasint m) {
  blasint extra_h = 0;
  if (h % GEMM_UNROLL_N)
    extra_h = GEMM_UNROLL_N - h % GEMM_UNROLL_N;
  return (extra_h + fh - 1 - ph) * ow * m;
}

blasint yaconv_extra_size(blasint h, blasint fh, blasint ph, blasint ow,
                          blasint m) {
  return yaconv_extra_size_before(fh, ph, ow, m)
       + yaconv_extra_size_after(h, fh, ph, ow, m);
}

void yaconv_single_image(float *image, blasint H, blasint W, blasint C,
                         float *filter, blasint FH, blasint FW, blasint M,
                         float *output, blasint PH, blasint PW,
                         float *sa, float *sb) {

  // Cache sizes used by the conventional GEMM
  const blasint l2_size = GEMM_P * GEMM_Q;
  const blasint l3_size = GEMM_Q * GEMM_R;

  // Compute block sizes for yaconv based on the conventional GEMM block sizes
  const blasint Q = MIN(FW * C, GEMM_Q);
  const blasint P = l2_size / Q / GEMM_UNROLL_M * GEMM_UNROLL_M;
  const blasint R = l3_size / W / C;

  // Compute output sizes
  const blasint OH = H + 2 * PH - FH + 1;
  const blasint OW = W + 2 * PW - FW + 1;

  // Shift output array pointer as yaconv addresses some space before the actual
  // output. This requires additional memory to be allocated for output
  output += yaconv_extra_size_before(FH, PH, OW, M);

  // Zero-out output to use alpha == 1 in every microkernel call later
  GEMM_BETA(M, OH * OW, 0, 0, NULL, 0, NULL, 0, output, M);

  for (blasint js = 0; js < H; js += R) {
    blasint min_j = MIN(H - js, R);

    edge_oncopy(W * C, min_j, image + js * W * C, W * C, sb);

    for (blasint fh = 0; fh < FH; ++fh) {

      for (blasint m = 0; m < M; m += P) {
        blasint min_m = MIN(M - m, P);

        for (blasint k = 0; k < FW * C; k += Q) {
          blasint min_k = MIN(FW * C - k, Q);

          GEMM_ITCOPY(min_k, min_m, filter + fh * FW * C * M + k * M + m, M,
                      sa);

          for (blasint jjs = 0, min_jj; jjs < min_j; jjs += min_jj) {
            min_jj = min_j - jjs;

#if defined(SKYLAKEX) || defined(COOPERLAKE) || defined(SAPPHIRERAPIDS)
            if (min_jj > 5*GEMM_UNROLL_N) min_jj = 6*GEMM_UNROLL_N;
#else
            if (min_jj > 2*GEMM_UNROLL_N) min_jj = 3*GEMM_UNROLL_N;
#endif
            if (min_jj % GEMM_UNROLL_N)
              min_jj += GEMM_UNROLL_N - min_jj % GEMM_UNROLL_N;

            for (blasint ow = 0; ow < OW; ++ow) {
              blasint image_start = (ow - PW) * C + k;
              blasint image_end = MIN(W * C, image_start + min_k);

              float *ar = sa;
              if (image_start < 0)
              {
                ar -= image_start * ((min_m - 1) % GEMM_UNROLL_M + 1);
                image_start = 0;
              }

              blasint K = image_end - image_start;
              if (K <= 0)
                continue;

              float *br = sb + jjs * W * C + image_start * min_jj;
              float *cr = output + ((js + jjs - fh + PH) * OW + ow) * M + m;

              GEMM_KERNEL_N(min_m, min_jj, K, 1, ar, br, cr, OW * M);
            }
          }
        }
      }
    }
  }
}

void yaconv(float **images, blasint n, blasint h, blasint w, blasint c,
            float *filter, blasint fh, blasint fw, blasint m,
            float **outputs, blasint ph, blasint pw) {

  // Allocate buffer for filter and image packing
  float *buffer = (float *)blas_memory_alloc(0);
  float *sa = (float *)((BLASLONG)buffer + GEMM_OFFSET_A);
  float *sb = (float *)(((BLASLONG)sa + ((GEMM_P * GEMM_Q * COMPSIZE * SIZE
            + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);

  for (int i = 0; i < n; ++i)
    yaconv_single_image(images[i], h, w, c, filter, fh, fw, m, outputs[i], ph,
        pw, sa, sb);

  blas_memory_free(buffer);
}
