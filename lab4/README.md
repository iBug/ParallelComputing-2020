# 《并行计算》实验 4

## 实验题目

利用 MPI 实现并行排序算法 PSRS

## 实验环境

- 操作系统：Ubuntu 20.04 LTS
- 编译器：GCC 9.3.0
- MPI 库：Open MPI 4.0.3
- 处理器：Intel(R) Xeon(R) Platinum 8153 CPU @ 2.00 GHz (2 sockets, 16 cores each, 32 cores total)
- 内存：32 GiB DDR4 2666 \* 4 (128 GiB total)

## 算法设计与分析

基本按照参考链接[^1]描述的那样实现，整个算法流程如下：

- 0 号进程读取输入数据，将总数 $n$ 广播至每个进程，并将待排序数组分散（Scatter）至每个进程，且每个进程收到的数据量约为 $\frac np$
- 每个进程使用快排\*排序自己收到的那部分数组，并有规律地选取 $p$ 个样例点
  - 这里我没有使用 C 语言标准库函数 `qsort()`，而是自己写了一个基于快排和插入排序的混合排序算法，在快排范围小于 6 的时候停止递归，在所有快排完成后再对整个数组进行一次插入排序，实测这样比纯快排要快一些，比 `qsort()` 还要更快
- 0 号进程收集全部 $p^2$ 个样例点并对它们进行一次多路归并，然后取第 $p,2p,\dots,p^2-p$ 个点（序数从 0 开始）作为分界值，广播至每个进程
- 每个进程将自己排序好的数组扫描一遍，根据收到的 $p-1$ 个分界点划分成 $p$ 段
- 使用 `MPI_Alltoall` 和 `MPI_Alltoallv`，第 $i$ 个进程收集每个进程原先的第 $i$ 段排好序的数，使用一轮多路归并合并成一个数组
- 0 号进程收集全部归并好的数组并输出

由于快排实现和多路归并并不是本实验的重点，因此这里不做详细介绍。

## 核心代码

分配任务（Scatter）

```c
// Prepare parameters
if (mpi_rank == 0) {
    block_sizes = malloc(n * sizeof(int));
    displs = malloc((n + 1) * sizeof(int));
    displs[0] = 0;
    for (int i = 0; i < n; i++) {
        displs[i + 1] = n * (i + 1) / mpi_size;
        block_sizes[i] = displs[i + 1] - displs[i];
    }
}
int this_start = n * mpi_rank / mpi_size;;
int this_size = n * (mpi_rank + 1) / mpi_size - this_start;
int *this_data = malloc(this_size * sizeof(int));

// Dispatch first batch of jobs
MPI_Scatterv(root_data, block_sizes, displs, MPI_INT, this_data, this_size, MPI_INT, 0, MPI_COMM_WORLD);
```

产生与收集样例点、广播分界值

```c
// Sampling
int *samples = malloc(mpi_size * sizeof(int));
for (int i = 0; i < mpi_size; i++) {
    samples[i] = this_data[this_size * (i + 1) / (mpi_size + 1)];
}
MPI_Gather(samples, mpi_size, MPI_INT, root_samples, mpi_size, MPI_INT, 0, MPI_COMM_WORLD);
free(samples);

// Merge samples and select pivots
int *pivots = malloc(mpi_size * sizeof(int));
if (mpi_rank == 0) {
    multi_merge_flat(root_samples, mpi_size, mpi_size);
    for (int i = 0; i < mpi_size - 1; i++) {
        pivots[i] = root_samples[mpi_size * (i + 1)];
    }
    free(root_samples);
}
MPI_Bcast(pivots, mpi_size - 1, MPI_INT, 0, MPI_COMM_WORLD);
```

按分界值分段

```c
int *class_index = malloc((mpi_size + 1) * sizeof(int)),
*rclass_index = malloc((mpi_size + 1) * sizeof(int)),
*class_sizes = malloc(mpi_size * sizeof(int)),
*rclass_sizes = malloc(mpi_size * sizeof(int));
class_index[0] = 0;
class_index[mpi_size] = this_size;
for (int i = 0, class_i = 1; i < this_size && class_i < mpi_size; i++) {
    while (this_data[i] >= pivots[class_i - 1]) {
        class_index[class_i] = i;
        class_i++;
        if (class_i >= mpi_size)
            break;
    }
}
for (int i = 0; i < mpi_size; i++) {
    class_sizes[i] = class_index[i + 1] - class_index[i];
}
```

进程间交换数组

```c
MPI_Alltoall(class_sizes, 1, MPI_INT, rclass_sizes, 1, MPI_INT, MPI_COMM_WORLD);
free(pivots);
pivots = NULL;

rclass_index[0] = 0;
for (int i = 0; i < mpi_size; i++) {
    rclass_index[i + 1] = rclass_index[i] + rclass_sizes[i];
}
int *rdata = malloc(rclass_index[mpi_size] * sizeof(int));
MPI_Alltoallv(this_data, class_sizes, class_index, MPI_INT,
              rdata, rclass_sizes, rclass_index, MPI_INT, MPI_COMM_WORLD);
```

归并与收集结果

```c
multi_merge(this_data, mpi_size, class_sizes);
MPI_Gather(&this_size, 1, MPI_INT, block_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);
if (mpi_rank == 0) {
    displs[0] = 0;
    for (int i = 1; i < mpi_size; i++) {
        displs[i] = displs[i - 1] + block_sizes[i - 1];
    }
}
MPI_Gatherv(this_data, this_size, MPI_INT, root_data, block_sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);
```

## 实验结果

### 运行结果

注：该运行结果在实验 1~3 所述实验环境中完成，下面的运行时间在本报告中所述实验环境中完成。

```text
ubuntu@iBug-Server:~/proj/PC-2020/lab4$ ls
Makefile  check.pl  data.pl  in.txt  main.c  out.txt  qsort.c  time.txt
ubuntu@iBug-Server:~/proj/PC-2020/lab4$ make
mpicc -O2 -Wall -Wno-unused-result -o main main.c
gcc -O2 -Wall -Wno-unused-result -fopenmp -o qsort qsort.c
ubuntu@iBug-Server:~/proj/PC-2020/lab4$ ./data.pl 64 > in.txt
ubuntu@iBug-Server:~/proj/PC-2020/lab4$ mpirun -np 4 main < in.txt > out.txt
ubuntu@iBug-Server:~/proj/PC-2020/lab4$ ./check.pl out.txt
OK
ubuntu@iBug-Server:~/proj/PC-2020/lab4$
```

其中 `data.pl` 和 `check.pl` 为帮助脚本，分别用于生成测试数据和验证输出。使用方法分别为

```shell
perl data.pl <count> [max]
perl check,pl <output>
```

`data.pl` 产生输出到标准输出，共 N + 1 行，第一行为一个数字 N，第 2 到 N+1 行各一个 0 到 max 之间的整数。若 `max` 未在命令行中给出，默认为 2 \* N。

`check.pl` 验证命令行参数中的文件是否正确排序，每行一个数字，且从第二行开始所有数字均不小于其上一行的数字。

### 运行时间（ms）

运行时间的测量方式与实验 1 中的两个 MPI 程序一致，此处不再重复。Ref 列为纯快排（不含并行化相关代码）的参考程序运行时间。

| 规模 \\ 进程数 |  Ref  |   1   |   2   |   4   |  8   |
| :------------: | :---: | :---: | :---: | :---: | :--: |
|   1,000,000    |  112  |  129  | 96.0  | 61.9  | 42.1 |
|   5,000,000    |  617  |  714  |  425  |  277  | 182  |
|   20,000,000   | 2713  | 3122  | 1820  | 1181  | 817  |
|  200,000,000   | 30688 | 34569 | 19594 | 12706 | 8611 |

### 加速比

| 规模 \\ 进程数 | Ref  |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: | :--: |
|   1,000,000    | 1.15 | 1.00 | 1.34 | 2.08 | 3.06 |
|   5,000,000    | 1.16 | 1.00 | 1.68 | 2.58 | 3.92 |
|   20,000,000   | 1.15 | 1.00 | 1.72 | 2.64 | 3.82 |
|  200,000,000   | 1.13 | 1.00 | 1.76 | 2.72 | 4.01 |

## 分析与总结

本实验是本课程四个实验中最复杂（其实也不怎么复杂）也是最有挑战性的一个实验，但其实我觉得没什么好总结的，因为唯一想说的一点我已经在其他实验报告里重复过 N 多遍了：写程序时保持一个清醒的头脑和明确的思路是非常重要的，尤其是面对数组下标 +1 -1 之类的细节问题时。整体来看写的过程还是比较轻松的，因为算法已经给出来了，只要照着实现就行，花时间比较多（也是代码篇幅比较长）的地方反而是多路归并，因为以前没写过，加上它又是多层指针和数组下标嵌套，涉及优先队列等多个数据结构，思维不够清楚的时候容易混乱，debug 花了点时间，其他部分反而一路顺畅。

测试这块，与前三个实验不同，这次“借用”了实验室的超算服务器跑，还是~~核多效果好~~啊。

[^1]: http://csweb.cs.wfu.edu/bigiron/LittleFE-PSRS/build/html/PSRSalgorithm.html
