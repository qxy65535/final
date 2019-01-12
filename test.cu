#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include<cutil_inline.h>
//#include<cutil.h>
typedef struct TestStruct
{
    int a;
    int b;
    int c;
} TestStruct;
bool InitCUDA(void)
{
    int count = 0;
    int i = 0;
    cudaGetDeviceCount(&count);
    if (count == 0)
    {
        fprintf(stderr, "There is no device./n");
        return false;
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
        return false;
    }
    cudaSetDevice(i);
    printf("CUDA initialized.\n");
    return true;
}

/*****************************************************************************************************
*kernel function, matrix adding
******************************************************************************************************/
__global__ void myKernel(TestStruct *dev_test, int *ma)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && j == 0)
    {
        dev_test->a = 101;
        dev_test->b = 102;
        dev_test->c = 103;
        ma[0] = 1011;
        ma[1] = 1022;
        ma[2] = 1033;
    }
}
int main()
{
    if (!InitCUDA())
        return 0;
    int ma[3] = {1, 2, 3};
    TestStruct *test = (TestStruct *)malloc(sizeof(TestStruct));
    TestStruct *host_test = (TestStruct *)malloc(sizeof(TestStruct));
    test->a = 1;
    test->b = 2;
    test->c = 3;
    TestStruct *dev_test;
    int *dev_ma;
    int *host_ma = (int *)malloc(sizeof(int) * 3);
    cudaMalloc((void **)&dev_ma, sizeof(int) * 3);
    cudaMalloc((void **)&dev_test, sizeof(TestStruct));
    cudaMemcpy(dev_test, test, sizeof(TestStruct), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ma, ma, sizeof(int) * 3, cudaMemcpyHostToDevice);
    dim3 Dg(1, 2, 1);                          //define the size and dim of the grid
    dim3 Db(3, 1, 1);                          //deifne the size and dim of each block
    myKernel<<<Dg, Db, 0>>>(dev_test, dev_ma); //using the kenerl function

    cudaMemcpy(host_test, dev_test, sizeof(TestStruct), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_ma, dev_ma, sizeof(int) * 3, cudaMemcpyDeviceToHost);
    printf("%d\n", host_ma[0]);
    printf("%d\n", host_ma[1]);
    printf("%d\n", host_ma[2]);
    printf("%d\n", host_test->a);
    printf("%d\n", host_test->b);
    printf("%d\n", host_test->c);

    return 0;
}
