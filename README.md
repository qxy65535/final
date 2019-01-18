# Final Exam: Video Encoding on GPU using the CUDA framework

## 目录

#### - [简介](#introduction)
#### - [优化说明](#optimization)
- [version 1.0](#v1)
- [version 2.0](#v2)
- [version 2.1](#v2.1)
- [version 3.0](#v3.0)
  - [BUG 修复](#bug)
  - [优化](#optim)
#### - [优化结果](#result)
#### - [执行指令](#shell)

<div id="introduction"> </div>

## 简介

本次作业的任务是利用 CUDA 优化加速 codec63 视频编码器。作业中使用 GPU 对 c63 编码器的运动估计、dct 和量化、反量化和 idct 部分进行并行优化，达到加速视频编码速度的目的。实验运行于 Intel(R) Xeon(R) CPU E5-2620 v2 @ 2.10GHz，Tesla K20c，CentOS Linux release 7.3.1611 平台，得到对 foreman 和 tractor 文件的加速分别达到了原版编码器的 8.08 倍和 14.70 倍，且编码得到的 .c63 文件能被原版解码器正常解码；解码得到的视频文件能够正常播放，相较于原始文件的 PSNR 与原版编解码器得到的结果相同。

<div id="optimization"> </div>

## 优化说明

<div id="v1"> </div>

### version 1.0

**git commit id: 04f7a58e92082ce44f62bf746f1e0cb3910c58e2**

在优化作业中，我首先尝试以最简单的做法利用 GPU 并行加速 C63 编码器，作为 CUDA C 编程初次尝试。在此版本中，我将运动估计用到的当前帧、参考帧、每一宏块记录参考帧中最相似块位置的结构体 mbs (macroblock) 及其他参数从 host 传入 device，将设备的 Grid 划分为 (width/8, height/8) 个 Block，每个 Block 有 8\*8 个 Thread，以 352x288 大小的 forman 为例，如图1所示：

![](image/TIM截图20190113132641.png)

图1 以 forman 为例 GPU 使用示意图

即一个 Grid 处理一张图像大小的数据，每个 Block 对一个宏块与参考帧求 SAD（绝对差的总和）。global 函数中以循环的形式遍历参考帧中的宏块寻找最相似的宏块，因此程序仍需要约 (me\_search\_range \* 2)^2 次数的串行 SAD 操作。在运动估计 kernel（核函数） 中，关键操作如下：

```c
int mb_x = blockIdx.x;
int mb_y = blockIdx.y;

for (y=top; y<bottom; ++y)
{
    for (x=left; x<right; ++x)
    {
        __shared__ uint8_t sad_block[8][8];

        int row = blockIdx.y*blockDim.y+threadIdx.y;
        int col = blockIdx.x*blockDim.x+threadIdx.x;
        
        sad_block[threadIdx.y][threadIdx.x] = abs(*(orig+row*w+col) - 
                                              *(ref+(y+threadIdx.y)*w+x+threadIdx.x));
        
        // 同步点：等待所有线程完成数据计算
        __syncthreads();
        int i,j;
        int sad = 0;

        for (i=0; i<8; ++i)
        {
            for (j=0; j<8; ++j)
                sad += sad_block[i][j];
        }

        if (sad < best_sad)
        {
            mb->mv_x = x - mx;
            mb->mv_y = y - my;
            best_sad = sad;
        }
    }
}
```

数据拷贝和核函数调用的主要操作如下：

```c
extern "C" void me_block_cuda(struct c63_common *cm, uint8_t *orig_host, uint8_t *ref_host, int cc)
{
    struct macroblock *mbs;
    uint8_t *orig, *ref;
    
    ...
        
    cudaMalloc((void **)&mbs, size_mbs);
    cudaMalloc((void **) &orig, size_orig);
    cudaMalloc((void **) &ref, size_ref);
    // cudaMemcpy(mbs, cm->curframe->mbs[cc], size_mbs, cudaMemcpyHostToDevice);
    cudaMemcpy(orig, orig_host, size_orig, cudaMemcpyHostToDevice);
    cudaMemcpy(ref, ref_host, size_ref, cudaMemcpyHostToDevice);

    int grid_x = cc>0 ? cm->mb_cols/2:cm->mb_cols;
    int grid_y = cc>0 ? cm->mb_rows/2:cm->mb_rows;
    dim3 dimGrid(grid_x, grid_y);
    dim3 dimBlock(8,8);

    me_kernel<<<dimGrid, dimBlock>>>(cm->padw[cc], cm->padh[cc], mbs, 
                                    cm->me_search_range, orig, ref, cc);
    cudaMemcpy(cm->curframe->mbs[cc], mbs, size_mbs, cudaMemcpyDeviceToHost);

    cudaFree(mbs);
    cudaFree(orig);
    cudaFree(ref);
}
```

经测试，在这一版代码优化后，对 foreman 文件的处理时间由 35.06s 左右缩短为 15.49s，加速约 2.26 倍；对 tractor 文件的处理时间由 1759.43s 左右缩短为 704.80s，加速约 2.50 倍，文件解码后 psnr 分别为 36.62 和 39.42，对 tractor 文件而言 psnr 略有下降，可能是编码过程中造成的些微误差。利用 nvprof 工具可以看到如下结果：

```shell
==27503== Profiling application: ./c63enc -w 352 -h 288 -o tmp/FOREMAN_352x288_30_orig_01.c63 /home/FOREMAN_352x288_30_orig_01.yuv
==27503== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.82%  11.7943s       891  13.237ms  2.2198ms  35.740ms  me_kernel(int, int, macroblock*, int, unsigned char*, unsigned char*, int)
  0.15%  18.194ms      1782  10.209us  5.4400us  28.672us  [CUDA memcpy HtoD]
  0.03%  3.0515ms       891  3.4240us  2.8160us  12.928us  [CUDA memcpy DtoH]

==27503== API calls:
Time(%)      Time     Calls       Avg       Min       Max  Name
 94.09%  11.8940s      2673  4.4497ms  17.819us  35.768ms  cudaMemcpy
  4.07%  514.13ms      2673  192.34us  5.2330us  256.77ms  cudaMalloc
  1.68%  211.95ms      2673  79.293us  8.5030us  240.43us  cudaFree
  0.14%  18.064ms       891  20.273us  16.660us  70.511us  cudaLaunch
  0.01%  1.7381ms      6237     278ns     168ns  14.858us  cudaSetupArgument
  0.01%  972.22us       166  5.8560us     245ns  214.43us  cuDeviceGetAttribute
  0.00%  597.45us       891     670ns     487ns  12.756us  cudaConfigureCall
  0.00%  161.76us         2  80.881us  58.010us  103.75us  cuDeviceTotalMem
  0.00%  90.792us         2  45.396us  43.837us  46.955us  cuDeviceGetName
  0.00%  4.7840us         2  2.3920us  1.0270us  3.7570us  cuDeviceGetCount
  0.00%  2.0290us         4     507ns     284ns     714ns  cuDeviceGet
```

由于 kernel 的调用是相对于 CPU 异步的，调用 kernel 后不久主机线程就会获得控制，这样一来下一个 cudaMemcpy 会在 kernel 完成执行前启动。 而 cudaMemcpy 会阻塞等待 kernel 的结果，直到 kernel  执行结束才进行数据传输并返回。如图2所示：

![1547261461144](image/1547261461144.png)

图2 核函数和 cudaMemcpy 调用示意

因此在以上结果中 API calls 显示 cudaMemcpy 占用了相近于 Profiling result 中核函数调用的时间，就是因为 cudaMemcpy 阻塞了将近一整个核函数执行的时间。从 Profiling result 的结果可以看出，核函数执行的时间一共为 11.7943s 用了绝大部分，而数据传输事实上只有几个毫秒。

在这一版本中，数据处理的并行度较低，利用的 GPU 计算资源较少，导致 CPU 存在非常大的性能浪费。目前瓶颈仍非常明显地在计算部分，而非数据传输。因此需要考虑更好的并行化方法。

此外，注意到代码中有一行注释行：

```c
// cudaMemcpy(mbs, cm->curframe->mbs[cc], size_mbs, cudaMemcpyHostToDevice);
```

一开始，将每一宏块记录参考帧中最相似块位置的结构体 mbs (macroblock) 也传输到了 GPU 中，但后来认识到 GPU 的数据处理过程中，并不需要 mbs 的原本，只需要开辟存储它的空间及标识它的名字，因此取消了对它的拷贝。在注释掉这行之后，cudaMemcpy 的调用次数由之前的 3564 减少为现在的 2673，对计算结果没有影响。虽然确实存在优化，但由于前文所述，数据传输不是目前的瓶颈，而且占用的时间非常少，对时间结果而言，这个优化带来的影响几乎不可见。

<div id="v2"> </div>

### version 2.0

**git commit id: e86fd841ab7abda00cd58b64d3a8c6e07dfb7237**

这一版本中修改了数据并行的方式，提高了数据的并行度。考虑到每个 Block 最多 1024 个线程，且仅块内线程能比较容易地实现数据共享，而 SAD 操作中不仅需要将 8\*8 的数据块逐差累加，还需要对比参考帧中当前宏块附近的宏块计算得到的 SAD 值以找出最相近的块，当取 16 为搜索步长时，将至多处理 32\*8\*32\*8 个数据。理想情况下需要在块内开启  32\*8\*32\*8 = 65536 个线程对数据并行处理，但由于块内线程数限制，不具有可行性；而通过进一步增加 Block 数量实现的并行会带来关联数据共享（主要是 SAD 比较）的麻烦。

因此在 v2.0 中，我依然将 Grid 划分为 (width/8, height/8) 个 Block，而每个 Block 使用 32\*32 = 1024 个线程，每个线程计算一个宏块与它附近一个宏块之间的 SAD，此时一个线程内存在 8\*8 次的串行操作。如图 3 所示，黑色外框为当前帧一个宏块在参考帧中可能搜索的范围，浅蓝色小框为 Block 中每一个线程处理参考帧的图像范围。由于图像的边界，有些宏块运动估计搜索的宏块数少于 1024，此时根据实际搜索的数量，Block 内部分线程将提前结束。代码中使用 32\*32 大小的共享存储器交换各个线程（即参考帧宏块）计算得到的 SAD，并从中选出最相似的宏块。

![](image/TIM截图20190113135146.png)

图3 运动估计搜索示意

核函数代码如下：

```c
__global__ void me_kernel(int padw, int padh, struct macroblock *mbs, int me_search_range, uint8_t *orig, uint8_t *ref, int cc)  
{
    __shared__ int sads[16*2][16*2];
    __shared__ int best_sad;
    sads[threadIdx.y][threadIdx.x] = 0;
    best_sad = INT_MAX;

    int mb_x = blockIdx.x;
    int mb_y = blockIdx.y;
    
    ...
        
    int row = blockIdx.y*8;
    int col = blockIdx.x*8;

    int i,j;
    for (i=0; i<8; ++i)
    {
        for (j=0; j<8; ++j)
        {
            int result = abs(*(orig+(row+i)*w+col+j) - 
                             *(ref+(top+threadIdx.y+i)*w+left+threadIdx.x+j));
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
```

经测试，在这一版代码优化后，对 foreman 文件的处理时间为 6.96s，加速约 5.04 倍；对 tractor 文件的处理时间由为 278.39s，加速约 6.32 倍，文件解码后 psnr 分别为 36.62 和 39.43，与原版编码器得到的结果相同。利用 nvprof 工具可以看到如下结果：

```shell
==17506== Profiling application: ./c63enc -w 352 -h 288 -o tmp/FOREMAN_352x288_30_orig_01.c63 /home/FOREMAN_352x288_30_orig_01.yuv
==17506== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 99.17%  3.50949s       891  3.9388ms  1.1101ms  12.460ms  me_kernel(int, int, macroblock*, int, unsigned char*, unsigned char*, int)
  0.74%  26.115ms      1782  14.654us  5.4400us  28.481us  [CUDA memcpy HtoD]
  0.09%  3.1170ms       891  3.4980us  2.8160us  11.872us  [CUDA memcpy DtoH]
```

可见核函数的运行时间被极大地压缩了，v2.0 比起 v1.0 具有很大程度的提升。除此之外，由于数据处理的并行化程度与视频分辨率，即每一帧的图像大小相关，而数据传输远远不是目前的瓶颈，因此对分辨率更高的 tractor 文件相比于低分辨率的 foreman 文件获得了更高的加速比，尤其是在更高程度并行化的 v2.0 中，两者之差体现得更为明显，这也体现了 GPU 更适合对大批量的数据进行并行处理，且数据量越大，加速效果越明显。

<div id="v2.1"> </div>

### version 2.1

**git commit id: 3cd646339d6db38eead8fde8a1d35b8b788940ce**

原本由于图 3 中观察到 v2.0 访问参考帧的数据时有非常大量的数据被重复访问，因此考虑每个 Block 将 32\*32 大小的参考帧数据放入共享存储器以加速数据的读取。但在实际操作中发现效果并不理想，进一步了解到对共享存储器的访问存在 bank conflict，若有 x 个同一 warp 的线程同时访问一个 bank，则其访问速度将下降到 1/x。而根据我 v2.0 的操作方法，若使用共享存储器，将存在非常大量的 bank conflict，使数据读取速度不加反减，因此最终放弃了使用共享存储器的做法。

而在 review 代码的过程中，发现找出最相似参考块的过程存在串行冗余，因此将代码修改如下：

```c
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
```

由于在前面的代码中已经通过原子操作和线程同步获得了 best_sad 的最小值，因此只需要每个线程各自对比自己的 best_sad 是不是最小值来决定要不要给 mb 的运动向量赋值就行了，并不需要再通过循环寻找 best_sad，同时也可以取消掉共享存储器的 \_\_shared\_\_ int sads 数组，进一步减少浪费提高性能。在 v2.1 的优化后，对 foreman 文件的处理时间为 4.84s，加速约 7.24 倍；对 tractor 文件的处理时间由为 175.16s，加速约 10.04 倍，文件解码后 psnr 分别为 36.63 和 39.43，与原版编码器得到的结果相比有一些小的偏差，但对视频观看没有任何影响。

此外，由于核函数运行期间控制权回到 CPU 主线程但被 cudaMemcpy 阻塞，因此可以在执行时间较长的 Y 帧核函数执行期间进行接下来 U、V 帧计算前 CPU 的操作。经过实验这样做可以对 forman 和 tractor 分别获得 0.4 秒和 2 秒 左右的加速，但由于较大地降低了代码的可读性，因此最终没有采用这个版本。

<div id="v3.0"> </div>

### version 3.0

**git commit id:  1301d0a443e809ad5ab93bf715c2024ae94ca77e**

v3.0 加入了对 dct、idct 的 cuda 加速，并整理了全部的代码。

<div id="bug"> </div>

#### bug 修复

在原版代码的 dct 和量化过程中，代码如下：

```c
dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[0], cm->padh[0], cm->curframe->residuals->Ydct, cm->quanttbl[0]);
dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[1], cm->padh[1], cm->curframe->residuals->Udct, cm->quanttbl[1]);
dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[2], cm->padh[2], cm->curframe->residuals->Vdct, cm->quanttbl[2]);
```

由于分配内存时 image->Y 的尺寸是 width\*height，而 dct 和量化时使用 padw 和 padh 必然对 image->Y 访问越界。为杜绝后患，在新建帧分配内存时，使用如下代码：

```c
/* Read Y' */
image->Y = calloc(1, cm->ypw*cm->yph);
/* Read U */
image->U = calloc(1, cm->upw*cm->uph);
/* Read V */
image->V = calloc(1, cm->vpw*cm->vph);
```

<div id="optim"> </div>

#### 优化

以 dct 和量化为例，与运动估计 v1.0 优化的思路相同，这里将 Grid 划分为 (pad_width/8, pad_height/8) 个 Block，每个 Block 有 8\*8 个 Thread，如图 1。以 8\*8 的数据块为单位，每个 Block 分别顺序地执行当前帧与预测帧作差、dct 1d、转置、scale、量化等操作，并写入到残差帧中。核函数中使用共享存储器存放原始的 8\*8 数据块和处理后的结果，每个 Thread 处理 8\*8 数据块中的一个数据。

比起运动估计在如何进行数据共享、如何将数据处理并行化上下了些功夫，dct 和量化以及 idct 和反量化只是简单地理清了数据存放的关系，并将 CPU 部分的代码移植到了 GPU 上，没有什么特殊的处理技术。值得注意的是作为 dct 量化的输出和 idct 反量化的输入，在残差帧中每个 8\*8 的宏块是连续顺序存放的，与原始帧的存放方式不同。dct 和量化的核函数代码如下：

```c
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
```

在代码中，dctlookup、zigzag\_U 和 zigzag\_V 采用硬编码的形式放在核函数中，减少数据传输的不必要开销，但一定程度上降低了代码的可读性。数据拷贝和核函数调用的操作如下：

```c
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
```

反量化和 idct 部分的代码与其类似，关于这一部分的代码详见 cuda\_utils.cu，此处不再贴出。

在 v3.0 的修改和优化后，对 foreman 文件的处理时间为 4.34s，加速约 8.08 倍；对 tractor 文件的处理时间由为 119.72s，加速约 14.70 倍，文件解码后 psnr 分别为 36.62 和 39.43，与原版编码器得到的结果相同。对 tractor 文件而言每一帧具有更多的数据，我的优化方法具有更高的并行度，因此其达到了远高于 foreman 的加速效果。

<div id="result"> </div>

## 优化结果

由于 gprof 工具无法记录 GPU 调用的时间，因此本次实验仍采用 clock 方法在程序始末添加时钟计算程序执行的总时间。本表格中，运行时间为在服务器上连续运行十次取平均值，使用的显卡为 Tesla K20c。

|  优化内容   | time(foreman) | time(tractor) | PSNR(foreman) | PSNR(tractor) |
| :---------: | :-----------: | :-----------: | :-----------: | :-----------: |
|  原始代码   |     35.06     |    1759.43    |     36.62     |     39.43     |
| version 1.0 |     15.49     |    704.80     |     36.62     |     39.42     |
| version 2.0 |     6.96      |    278.39     |     36.62     |     39.43     |
| version 2.1 |     4.84      |    175.16     |     36.63     |     39.43     |
| version 3.0 |     4.34      |    119.72     |     36.62     |     39.43     |

最终代码版本为 version 3.0，实验得到对 foreman 和 tractor 文件的加速分别达到了原版编码器的 8.08 倍和 14.70 倍，且编码得到的 .c63 文件能被原版解码器正常解码；解码得到的视频文件能够正常播放，相较于原始文件的 PSNR 与原版编解码器得到的结果相同。

version 3.0 编码器对 foreman 编码后得到的文件大小为 4,934,754 字节，略大于原版编码器得到的 4,933,703 字节；对 tractor 编码后得到的文件大小为 223,546,531 字节，也略大于原版编码器得到的 223,384,379 字节。我也不知道为什么会差一点。。。

<div id="shell"> </div>

## 执行指令

**compile**

```shell
$ make
```

**encode**

```shell
$ nvprof ./c63enc -w 352 -h 288 -o tmp/FOREMAN_352x288_30_orig_01.c63 /home/FOREMAN_352x288_30_orig_01.yuv
$ nvprof ./c63enc -w 1920 -h 1080 -o tmp/1080p_tractor.c63 /home/1080p_tractor.yuv
```

**decode**
```shell
$ ./c63dec tmp/FOREMAN_352x288_30_orig_01.c63  tmp/foreman.yuv
$ ./c63dec tmp/1080p_tractor.c63  tmp/tractor.yuv
```

**play the raw yuv file**

```shell
$ vlc --rawvid-width 352 --rawvid-height 288 --rawvid-fps 30 --rawvid-chroma I420 tmp/foreman.yuv
$ vlc --rawvid-width 1920 --rawvid-height 1080 --rawvid-fps 30 --rawvid-chroma I420 tmp/tractor.yuv
```

**psnr**
```shell
~/Downloads/evalvid/psnr 1920 1080 420 ~/Desktop/codec63-ori/codec63/video/1080p_tractor.yuv ~/Desktop/codec63-ori/codec63/tmp/tractor.yuv
~/Downloads/evalvid/psnr 352 288 420 ~/Desktop/codec63-ori/codec63/video/FOREMAN_352x288_30_orig_01.yuv ~/Desktop/codec63-ori/codec63/tmp/foreman.yuv
```

**time account**
```shell
$ gprof c63enc gmon.out -p
$ nvprof --print-gpu-trace ./c63enc -w 352 -h 288 -o tmp/FOREMAN_352x288_30_orig_01.c63 /home/FOREMAN_352x288_30_orig_01.yuv

$ nvprof --print-gpu-trace ./c63enc -w 1920 -h 1080 -o tmp/1080p_tractor.c63 /home/1080p_tractor.yuv
```