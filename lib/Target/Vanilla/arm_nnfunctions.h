/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nnfunctions.h
 * Description:  Public header file for CMSIS NN Library
 *
 * $Date:        13. July 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

/**
   \mainpage CMSIS NN Software Library
   *
   * Introduction
   * ------------
   *
   * This user manual describes the CMSIS NN software library,
   * a collection of efficient neural network kernels developed to maximize the 
   * performance and minimize the memory footprint of neural networks on Cortex-M processor cores.
   *
   * The library is divided into a number of functions each covering a specific category:
   * - Neural Network Convolution Functions
   * - Neural Network Activation Functions
   * - Fully-connected Layer Functions
   * - Neural Network Pooling Functions
   * - Softmax Functions
   * - Neural Network Support Functions
   *
   * The library has separate functions for operating on different weight and activation data
   * types including 8-bit integers (q7_t) and 16-bit integers (q15_t). The descrition of the
   * kernels are included in the function description. The implementation details are also 
   * described in this paper [1]. 
   *
   * Block Diagram
   * --------
   * \image html CMSIS-NN-OVERVIEW.PNG
   *
   * Examples
   * --------
   *
   * The library ships with a number of examples which demonstrate how to use the library functions.
   *
   * Pre-processor Macros
   * ------------
   *
   * Each library project have differant pre-processor macros.
   *
   * - ARM_MATH_DSP:
   *
   * Define macro ARM_MATH_DSP, If the silicon supports DSP instructions.
   *
   * - ARM_MATH_BIG_ENDIAN:
   *
   * Define macro ARM_MATH_BIG_ENDIAN to build the library for big endian targets. By default library builds for little endian targets.
   *
   * - ARM_NN_TRUNCATE:
   *
   * Define macro ARM_NN_TRUNCATE to use floor instead of round-to-the-nearest-int for the computation.
   *
   * Copyright Notice
   * ------------
   *
   * Copyright (C) 2010-2018 Arm Limited. All rights reserved.
   *
   * [1] CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs https://arxiv.org/abs/1801.06601
   */

/**
 * @defgroup groupNN Neural Network Functions
 * These functions perform basic operations for neural network layers. 
 */

#ifndef _ARM_NNFUNCTIONS_H
#define _ARM_NNFUNCTIONS_H

//#include "arm_nnsupportfunctions.h"
//#include "arm_nn_tables.h"

#define USE_INTRINSIC

//#define ARM_NN_TRUNCATE /* This config the rounding model to floor or round to the nearest int */

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @defgroup NNConv Neural Network Convolution Functions
 *
 * Perform convolution layer
 *
 * The convolution is implemented in 2 steps: im2col and GEMM
 *
 * im2col is a process of converting each patch of image data into 
 * a column. After im2col, the convolution is computed as matrix-matrix
 * multiplication.
 * 
 * To reduce the memory footprint, the im2col is performed partially.
 * Each iteration, only a few column (i.e., patches) are generated and 
 * computed with GEMM kernels similar to CMSIS-DSP arm_mat_mult functions.
 *
 */

  /**
   * @brief Basic Q7 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimention
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input 
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns <code>ARM_MATH_SUCCESS</code> 
   *
   */
    typedef int8_t q7_t;
    typedef int16_t q15_t;
    typedef unsigned short     int uint16_t;
    double arm_convolve_HWC_q7_basic(const q7_t * Im_in,
                                         const uint16_t dim_im_in,
                                         const uint16_t ch_im_in,
                                         const q7_t * wt,
                                         const uint16_t ch_im_out,
                                         const uint16_t dim_kernel,
                                         const uint16_t padding,
                                         const uint16_t stride,
                                         const q7_t * bias,
                                         const uint16_t bias_shift,
                                         const uint16_t out_shift,
                                         q7_t * Im_out, 
                                         const uint16_t dim_im_out, 
                                         q15_t * bufferA, 
                                         q7_t * bufferB);

  

#ifdef __cplusplus
}
#endif

#endif
