#include <stdio.h>  
#include <stdlib.h>  
#include <math.h>
#include <cuda_runtime.h> 
//for __syncthreads()
#ifndef __CUDACC_RTC__ 
#define __CUDACC_RTC__
#endif // !(__CUDACC_RTC__)
#include <device_functions.h>

#include "c63.h"
#define ISQRT2 0.70710678118654f

extern "C" int InitCUDA(void)
{
    int count = 0;
    int i = 0;
    cudaGetDeviceCount(&count);
    if (count == 0)
    {
        fprintf(stderr, "There is no device./n");
        return 0;
    }
    for (i = 0; i < count; i++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if (prop.major >= 1)
            {
                break;
            }
        }
    }
    if (i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return 0;
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    cudaSetDevice(i);
    fprintf(stderr, "CUDA initialized. Use device: %s.\n", deviceProp.name);
    return 1;
}

__global__ void me_kernel(int padw, int padh, struct macroblock *mbs, 
                        int me_search_range, uint8_t *orig, uint8_t *ref, int cc)  
{
    __shared__ int best_sad;
    best_sad = INT_MAX;

    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;
    struct macroblock *mb = &mbs[mb_y * padw/8 + mb_x];

    int range = me_search_range;

    /* Half resolution for chroma channels. */
    if (cc > 0)
	    range /= 2;

    int left = mb_x*8 - range;
    int top = mb_y*8 - range;
    int right = mb_x*8 + range;
    int bottom = mb_y*8 + range;

    int w = padw;
    int h = padh;

    /* Make sure we are within bounds of reference frame */
    // TODO: Support partial frame bounds
    if (left < 0)
        left = 0;
    if (top < 0)
        top = 0;
    if (right > (w - 8))
        right = w - 8;
    if (bottom > (h - 8))
        bottom = h - 8;
    if (threadIdx.y >= (bottom-top) || threadIdx.x >= (right-left))
        return;

    int mx = mb_x * 8;
    int my = mb_y * 8;

    int row = blockIdx.y*8;
    int col = blockIdx.x*8;

    int i,j;
    int result = 0;
    for (i=0; i<8; ++i)
    {
        for (j=0; j<8; ++j)
            result += abs(*(orig+(row+i)*w+col+j) - 
                        *(ref+(top+threadIdx.y+i)*w+left+threadIdx.x+j));
    }

    // 找出小的sad值
    atomicMin(&best_sad, result);
    __syncthreads();

    // 找出最相似的参考块
    if (result == best_sad)
    {
        mb->mv_x = left + threadIdx.x - mx;
        mb->mv_y = top + threadIdx.y - my;
    }
    mb->use_mv = 1;
}  

__global__ void dct_quantize_kernel(uint8_t * in_data, uint8_t * prediction, 
                                int16_t * out_data, uint8_t * quant_tbl, int width)
{
    __shared__ int16_t block[8][8];
    __shared__ float dct_out[8][8];
    __shared__ float dct_out2[8][8];
    float dctlookup[8][8] = {
        {1.000000f, 0.980785f, 0.923880f, 0.831470f, 0.707107f, 0.555570f, 0.382683f, 0.195090f, },
        {1.000000f, 0.831470f, 0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f, },
        {1.000000f, 0.555570f, -0.382683f, -0.980785f, -0.707107f, 0.195090f, 0.923880f, 0.831470f, },
        {1.000000f, 0.195090f, -0.923880f, -0.555570f, 0.707107f, 0.831470f, -0.382683f, -0.980785f, },
        {1.000000f, -0.195090f, -0.923880f, 0.555570f, 0.707107f, -0.831470f, -0.382683f, 0.980785f, },
        {1.000000f, -0.555570f, -0.382683f, 0.980785f, -0.707107f, -0.195090f, 0.923880f, -0.831470f, },
        {1.000000f, -0.831470f, 0.382683f, 0.195090f, -0.707107f, 0.980785f, -0.923880f, 0.555570f, },
        {1.000000f, -0.980785f, 0.923880f, -0.831470f, 0.707107f, -0.555570f, 0.382683f, -0.195090f, },
    };
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    block[threadIdx.y][threadIdx.x] = (int16_t)*(in_data + row * width + col) - 
                                                *(prediction + row * width + col);
    __syncthreads();

    int i;
    float dct = 0;

    // dct 1d
    for (i=0; i<8; ++i)
        dct += block[threadIdx.y][i] * dctlookup[i][threadIdx.x];
    // transpose
    dct_out[threadIdx.x][threadIdx.y] = dct;
    __syncthreads();

    // dct 1d
    dct = 0;
    for (i=0; i<8; ++i)
        dct += dct_out[threadIdx.y][i] * dctlookup[i][threadIdx.x];
    // transpose
    // dct_out2[threadIdx.x][threadIdx.y] = dct;
    // __syncthreads();
    // scale_block
    float a1 = !threadIdx.x ? ISQRT2 : 1.0f;
    float a2 = !threadIdx.y ? ISQRT2 : 1.0f;
    dct_out2[threadIdx.x][threadIdx.y] = dct * a1 * a2;
    __syncthreads();
    // quantize_block
    uint8_t zigzag_U[64] =
    {
        0,
        1, 0,
        0, 1, 2,
        3, 2, 1, 0,
        0, 1, 2, 3, 4,
        5, 4, 3, 2, 1, 0,
        0, 1, 2, 3, 4, 5, 6,
        7, 6, 5, 4, 3, 2, 1, 0,
        1, 2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 3, 2,
        3, 4, 5, 6, 7,
        7, 6, 5, 4,
        5, 6, 7,
        7, 6,
        7,
    };
    uint8_t zigzag_V[64] =
    {
        0,
        0, 1,
        2, 1, 0,
        0, 1, 2, 3,
        4, 3, 2, 1, 0,
        0, 1, 2, 3, 4, 5,
        6, 5, 4, 3, 2, 1, 0,
        0, 1, 2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 3, 2, 1,
        2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 3,
        4, 5, 6, 7,
        7, 6, 5,
        6, 7,
        7,
    };
    int zigzag = threadIdx.y * 8 + threadIdx.x;
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];
    out_data[zigzag + blockIdx.y*width*8 + blockIdx.x*64] = round((dct_out2[v][u] / 4.0) / quant_tbl[zigzag]);

}

__global__ void dequantize_idct_kernel(int16_t * in_data, uint8_t * prediction, 
                                    uint8_t * out_data, uint8_t * quant_tbl, int width)
{
    __shared__ float mb[8][8];
    __shared__ float mb2[8][8];
    float dctlookup[8][8] = {
        {1.000000f, 0.980785f, 0.923880f, 0.831470f, 0.707107f, 0.555570f, 0.382683f, 0.195090f, },
        {1.000000f, 0.831470f, 0.382683f, -0.195090f, -0.707107f, -0.980785f, -0.923880f, -0.555570f, },
        {1.000000f, 0.555570f, -0.382683f, -0.980785f, -0.707107f, 0.195090f, 0.923880f, 0.831470f, },
        {1.000000f, 0.195090f, -0.923880f, -0.555570f, 0.707107f, 0.831470f, -0.382683f, -0.980785f, },
        {1.000000f, -0.195090f, -0.923880f, 0.555570f, 0.707107f, -0.831470f, -0.382683f, 0.980785f, },
        {1.000000f, -0.555570f, -0.382683f, 0.980785f, -0.707107f, -0.195090f, 0.923880f, -0.831470f, },
        {1.000000f, -0.831470f, 0.382683f, 0.195090f, -0.707107f, 0.980785f, -0.923880f, 0.555570f, },
        {1.000000f, -0.980785f, 0.923880f, -0.831470f, 0.707107f, -0.555570f, 0.382683f, -0.195090f, },
    };
    int zigzag = threadIdx.y * 8 + threadIdx.x;
    // mb[threadIdx.y][threadIdx.x] = in_data[blockIdx.y*width*8 + blockIdx.x*64 + zigzag];

    // dequantize_block
    uint8_t zigzag_U[64] =
    {
        0,
        1, 0,
        0, 1, 2,
        3, 2, 1, 0,
        0, 1, 2, 3, 4,
        5, 4, 3, 2, 1, 0,
        0, 1, 2, 3, 4, 5, 6,
        7, 6, 5, 4, 3, 2, 1, 0,
        1, 2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 3, 2,
        3, 4, 5, 6, 7,
        7, 6, 5, 4,
        5, 6, 7,
        7, 6,
        7,
    };
    uint8_t zigzag_V[64] =
    {
        0,
        0, 1,
        2, 1, 0,
        0, 1, 2, 3,
        4, 3, 2, 1, 0,
        0, 1, 2, 3, 4, 5,
        6, 5, 4, 3, 2, 1, 0,
        0, 1, 2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 3, 2, 1,
        2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 3,
        4, 5, 6, 7,
        7, 6, 5,
        6, 7,
        7,
    };
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];
    mb2[v][u] = round((in_data[blockIdx.y*width*8 + blockIdx.x*64 + zigzag] * quant_tbl[zigzag]) / 4.0);
    __syncthreads();

    // scale_block
    float a1 = !threadIdx.x ? ISQRT2 : 1.0f;
    float a2 = !threadIdx.y ? ISQRT2 : 1.0f;
    mb2[threadIdx.y][threadIdx.x] *= a1 * a2;
    __syncthreads();
    // idct_1d
    int i;
    float idct = 0;
    for (i=0; i<8; ++i)
        idct += mb2[threadIdx.y][i] * dctlookup[threadIdx.x][i];
    // transpose
    mb[threadIdx.x][threadIdx.y] = idct;
    __syncthreads();

    // idct_1d
    idct = 0;
    for (i=0; i<8; ++i)
        idct += mb[threadIdx.y][i] * dctlookup[threadIdx.x][i];
    // transpose
    mb2[threadIdx.x][threadIdx.y] = idct;
    __syncthreads();

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int16_t tmp = (int16_t)mb2[threadIdx.y][threadIdx.x] + (int16_t)*(prediction + row * width + col);
    if (tmp < 0)
        tmp = 0;
    else if (tmp > 255)
        tmp = 255;
    *(out_data + row * width + col) = tmp;
}

void me_block_cuda(struct c63_common *cm, uint8_t *orig_host, uint8_t *ref_host, int cc)
{
    // cudaSetDevice(1);
    struct macroblock *mbs;
    uint8_t *orig, *ref;

    int size_mbs, size_orig, size_ref;
    switch (cc)
    {
    case 0:
        size_mbs = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
        size_orig = cm->ypw*cm->yph;
        size_ref = cm->ypw * cm->yph;
        break;
    case 1:
        size_mbs = cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock);
        size_orig = cm->upw * cm->uph;
        size_ref = cm->upw * cm->uph;
        break;
    case 2:
        size_mbs = cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock);
        size_orig = cm->vpw * cm->vph;
        size_ref = cm->vpw * cm->vph;
        break;
    }
    
    cudaMalloc((void **) &mbs, size_mbs);
    cudaMalloc((void **) &orig, size_orig);
    cudaMalloc((void **) &ref, size_ref);
    cudaMemcpy(orig, orig_host, size_orig, cudaMemcpyHostToDevice);
    cudaMemcpy(ref, ref_host, size_ref, cudaMemcpyHostToDevice);

    int grid_x = cc>0 ? cm->mb_cols/2:cm->mb_cols;
    int grid_y = cc>0 ? cm->mb_rows/2:cm->mb_rows;
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(cm->me_search_range*2,cm->me_search_range*2);

    me_kernel<<<dimGrid, dimBlock>>>(cm->padw[cc], cm->padh[cc], mbs, 
                                    cm->me_search_range, orig, ref, cc);
                                    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    cudaMemcpy(cm->curframe->mbs[cc], mbs, size_mbs, cudaMemcpyDeviceToHost);

    cudaFree(mbs);
    cudaFree(orig);
    cudaFree(ref);
}

extern "C" void c63_motion_estimate_cuda(struct c63_common * cm)
{
    me_block_cuda(cm, cm->curframe->orig->Y, cm->refframe->recons->Y, 0);
    me_block_cuda(cm, cm->curframe->orig->U, cm->refframe->recons->U, 1);
    me_block_cuda(cm, cm->curframe->orig->V, cm->refframe->recons->V, 2);
}

extern "C" void dct_quantize_cuda(uint8_t * in_data_, uint8_t * prediction_, uint32_t width, 
                                uint32_t height, int16_t * out_data_, uint8_t * quantization_)
{
    int size = width*height;
    uint8_t *in_data, *prediction, *quantization;
    int16_t *out_data;

    cudaMalloc((void **) &in_data, size);
    cudaMalloc((void **) &prediction, sizeof(uint8_t) * size);
    cudaMalloc((void **) &out_data, sizeof(int16_t) * size);
    cudaMalloc((void **) &quantization, sizeof(uint8_t) * 64);

    cudaMemcpy(in_data, in_data_, size, cudaMemcpyHostToDevice);
    cudaMemcpy(prediction, prediction_, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(quantization, quantization_, sizeof(uint8_t) * 64, cudaMemcpyHostToDevice);

    int grid_x = width/8;
    int grid_y = height/8;
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(8,8);

    dct_quantize_kernel<<<dimGrid, dimBlock>>>(in_data, prediction, out_data, quantization, width);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    cudaMemcpy(out_data_, out_data, sizeof(int16_t) * size, cudaMemcpyDeviceToHost);
    cudaFree(in_data);
    cudaFree(prediction);
    cudaFree(out_data);
    cudaFree(quantization);
}

extern "C" void dequantize_idct_cuda(int16_t * in_data_, uint8_t * prediction_, uint32_t width, 
                                    uint32_t height, uint8_t * out_data_, uint8_t * quantization_)
{
    int size = width*height;
    uint8_t *out_data, *prediction, *quantization;
    int16_t *in_data;

    cudaMalloc((void **) &in_data, sizeof(int16_t) * size);
    cudaMalloc((void **) &prediction, sizeof(uint8_t) * size);
    cudaMalloc((void **) &out_data, size);
    cudaMalloc((void **) &quantization, sizeof(uint8_t) * 64);

    cudaMemcpy(in_data, in_data_, sizeof(int16_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(prediction, prediction_, sizeof(uint8_t) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(quantization, quantization_, sizeof(uint8_t) * 64, cudaMemcpyHostToDevice);

    int grid_x = width/8;
    int grid_y = height/8;
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(8,8);

    dequantize_idct_kernel<<<dimGrid, dimBlock>>>(in_data, prediction, out_data, quantization, width);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    
    cudaMemcpy(out_data_, out_data, size, cudaMemcpyDeviceToHost);
    cudaFree(in_data);
    cudaFree(prediction);
    cudaFree(out_data);
    cudaFree(quantization);
}