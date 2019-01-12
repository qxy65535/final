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

__global__ void me_kernel(int padw, int padh, struct macroblock *mbs, int me_search_range, uint8_t *orig, uint8_t *ref, int cc)  
{
    __shared__ int sads[16*2][16*2];
    __shared__ int best_sad;
    sads[threadIdx.y][threadIdx.x] = 0;
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
    for (i=0; i<8; ++i)
    {
        for (j=0; j<8; ++j)
        {
            int result = abs(*(orig+(row+i)*w+col+j) - *(ref+(top+threadIdx.y+i)*w+left+threadIdx.x+j));
            atomicAdd(&sads[threadIdx.y][threadIdx.x], result);
        }
    }

    // 找出小的sad值
    atomicMin(&best_sad, sads[threadIdx.y][threadIdx.x]);
    __syncthreads();

    // 找出最相似的参考块
    for (i=0; i<(bottom-top); ++i)
    {
        for (j=0; j<(right-left); ++j)
        {
            if (sads[i][j] == best_sad)
            {
                mb->mv_x = left + j - mx;
                mb->mv_y = top + i - my;
                i = bottom-top;
                break;
            }
        }
    }
    mb->use_mv = 1;
}  

extern "C" void me_block_cuda(struct c63_common *cm, uint8_t *orig_host, uint8_t *ref_host, int cc)
{
    struct macroblock *mbs;
    uint8_t *orig, *ref;

    int size_mbs = 0;
    int size_orig = 0;
    int size_ref = 0;
    switch (cc)
    {
    case 0:
        size_mbs = cm->mb_rows * cm->mb_cols * sizeof(struct macroblock);
        size_orig = cm->width * cm->height;
        size_ref = cm->ypw * cm->yph;
        break;
    case 1:
        size_mbs = cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock);
        size_orig = cm->width * cm->height;
        size_ref = cm->upw * cm->uph;
        break;
    case 2:
        size_mbs = cm->mb_rows/2 * cm->mb_cols/2 * sizeof(struct macroblock);
        size_orig = cm->width * cm->height;
        size_ref = cm->vpw * cm->vph;
        break;
    }
    cudaMalloc((void **)&mbs, size_mbs);
    cudaMalloc((void **) &orig, size_orig);
    cudaMalloc((void **) &ref, size_ref);
    // cudaMemcpy(mbs, cm->curframe->mbs[cc], size_mbs, cudaMemcpyHostToDevice);
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