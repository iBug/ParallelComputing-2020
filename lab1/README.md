# 《并行计算》实验 1

## 实验题目

利用 MPI 和 OpenMP 编写简单的程序，测试并行计算系统性能

## 实验环境

- 操作系统：Ubuntu 20.04 LTS
- 编译器：GCC 9.3.0
- MPI 库：Open MPI 4.0.3
- 处理器：Intel Core i7-8850H (6 cores, 12 threads, 2.6~4.3 GHz)
- 内存：32 GiB DDR4 3200 \* 2 (64 GiB total)

## 算法设计与分析

### 1. 求素数个数 (OpenMP)

求 1 到 n 之间素数个数采用筛选法（Sieve of Eratosthenes），从 $p=2$ 每次将 $p^2$ 开始的全部 $p$ 的倍数标记为合数。

该算法可以使用 OpenMP 进行循环并行化的部分为“素数标记数组”初始化的部分（即整个数组标记为素数）以及每次找到下一个素数后从 $p^2$ 开始标记合数的部分。

### 2. 求素数个数 (MPI)

该实现继续采用筛选法，区别在于只有 0 号进程负责寻找“需要筛掉其倍数的素数”，而其他进程只负责根据 0 号进程找到的素数进行过筛和统计，因此这里有一个额外的要求是 n 不小于进程数 (`mpi_world_size`) 的平方。

### 3. 迭代求 π (OpenMP)

该迭代法采用的公式为
$$
\pi=\lim_{n\to\infty}\cfrac{1}{n}\sum_{i=1}^n\cfrac{4}{1+\left(\cfrac{2i-1}{2n}\right)^2}
$$
由于结果为累加和，因此每个循环可以独立计算，将最终结果相加即可。

### 4. 迭代求 π (MPI)

与 3 没有太多区别，唯一一点需要注意的是，出于精度考虑，每个进程计算的任务不是 $[1,a_1),[a_1,a_2),\cdots,[a_{m-1},n)$，而是第 $i$ 个进程计算 $i,i+m,\cdots,i+km,\cdots$，其中 $m$ 为总进程数。即任务分配不是连续的段，而是交叉的。

## 核心代码

### 1. 求素数个数 (OpenMP)

```c
#pragma omp parallel for
for (uint64_t i = 2; i <= n; i++) {
    prime[i] = 1;
}

size_t count = 0;
for (uint64_t i = 2; i <= n; i++) {
    if (prime[i]) {
        count++;
        if (i <= sqrt_n) {
            #pragma omp parallel for
            for (uint64_t j = i * i; j <= n; j += i)
                prime[j] = 0;
        }
    }
}
```

### 2. 求素数个数 (MPI)

输入

```c
if (rank == 0) {
    scanf(" %u", &n);
    n += 1;
}
if (n < size * size) {
    fprintf(stderr, "Unexpected number %u, need at least %d for %d processes.\n", n, size * size, size);
}
MPI_Bcast(&n, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
```

主循环

```c
unsigned i, k = 2;
while (k * k <= n) {
    if (k * k >= start) {
        i = k * k - start;
    } else {
        i = ((start - 1) / k + 1) * k - start;
    }
    for (; i < length; i += k)
        prime[i] = 0;
    if (rank == 0) {
        // Find next prime
        for (i = k + 1; prime[i] == 0; i++);
        k = i;
    }
    MPI_Bcast(&k, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}
```

统计

```c
unsigned cnt = 0, total = 0;
for (unsigned i = 0; i < length; i++) {
    if (prime[i])
        cnt++;
}
MPI_Reduce(&cnt, &total, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
```

### 3. 迭代求 π (OpenMP)

```c
double sum = 0;

#pragma omp parallel for reduction(+: sum)
for (unsigned long i = 0; i < n; i++) {
    double x = (i + 0.5) / n;
    sum += 4.0 / (1.0 + x * x);
}
sum /= n;
```

### 4. 迭代求 π (MPI)

```c
double sum = 0;
for (unsigned i = rank; i < n; i += size) {
    double x = (i + 0.5) / n;
    sum += 4.0 / (1.0 + x * x);
}
double total;
MPI_Reduce(&sum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
total /= n;
```

## 实验结果

OpenMP 程序通过指定环境变量 `OMP_NUM_THREADS` 来控制线程数，MPI  程序通过指定 `-np` 参数来控制进程数。所有 OpenMP 程序均为单进程，所有 MPI 程序每进程均为单线程。

OpenMP 程序使用 `omp_get_wtime()`，MPI 程序使用 `MPI_Wtime()` 获取时间，两种程序均采用两次获取时间作差作为算法运行时间输出，获取时间的位置是输入完成后和输出开始前，这样测得的时间能更好的反映算法本身的运行时间。所有数值均为多次运行去掉部分最高最低结果后剩余结果的平均值（见代码 `reporttime.pl`）。

### 1. 求素数个数 (OpenMP)

运行结果

```text
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./1-openmp <<< 1000
168
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./1-openmp <<< 10000
1229
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./1-openmp <<< 100000
9592
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./1-openmp <<< 500000
41538
ubuntu@iBug-Server:~/proj/PC-2020/lab1$
```

运行时间（ms）

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 0.08 | 0.12 | 0.09 | 0.18 |
|     10,000     | 0.12 | 0.17 | 0.12 | 0.13 |
|    100,000     | 0.72 | 0.66 | 0.37 | 3.36 |
|    500,000     | 2.03 | 1.18 | 1.12 | 3.27 |
|   50,000,000   | 301  | 233  | 222  | 230  |
|  100,000,000   | 648  | 501  | 482  | 499  |

加速比

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 1.0  |  -   |  -   |  -   |
|     10,000     | 1.0  |  -   |  -   |  -   |
|    100,000     | 1.0  |  -   |  -   |  -   |
|    500,000     | 1.0  | 1.72 | 1.81 | 0.62 |
|   50,000,000   | 1.0  | 1.29 | 1.36 | 1.31 |
|  100,000,000   | 1.0  | 1.29 | 1.34 | 1.30 |

### 2. 求素数个数 (MPI)

运行结果

```text
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 1-mpi <<< 1000
168
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 1-mpi <<< 10000
1229
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 1-mpi <<< 100000
9592
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 1-mpi <<< 500000
41538
ubuntu@iBug-Server:~/proj/PC-2020/lab1$
```

运行时间（ms）

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 0.01 | 0.07 | 0.16 | 0.18 |
|     10,000     | 0.11 | 0.14 | 0.15 | 0.29 |
|    100,000     | 1.06 | 0.41 | 0.24 | 0.28 |
|    500,000     | 7.12 | 1.18 | 0.51 | 0.56 |
|   50,000,000   | 326  | 219  | 184  | 183  |
|  100,000,000   | 680  | 471  | 413  | 418  |

加速比

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 1.0  |  -   |  -   |  -   |
|     10,000     | 1.0  |  -   |  -   |  -   |
|    100,000     | 1.0  | 2.59 | 4.42 | 3.49 |
|    500,000     | 1.0  | 6.03 | 13.9 | 12.7 |
|   50,000,000   | 1.0  | 1.49 | 1.77 | 1.78 |
|  100,000,000   | 1.0  | 1.44 | 1.65 | 1.63 |

### 3. 迭代求 π (OpenMP)

运行结果

```text
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./2-openmp <<< 1000
3.141592736923
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./2-openmp <<< 10000
3.141592654423
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./2-openmp <<< 50000
3.141592653623
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ ./2-openmp <<< 100000
3.141592653598
ubuntu@iBug-Server:~/proj/PC-2020/lab1$
```

运行时间（ms）

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 0.02 | 0.15 | 0.24 | 0.42 |
|     10,000     | 0.10 | 0.20 | 0.29 | 0.47 |
|     50,000     | 0.53 | 0.41 | 0.38 | 2.41 |
|    100,000     | 1.02 | 0.66 | 0.40 | 1.76 |
|   50,000,000   | 93.7 | 47.3 | 34.0 | 25.7 |
|  100,000,000   | 187  | 94.4 | 61.6 | 49.2 |

加速比

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 1.0  |  -   |  -   |  -   |
|     10,000     | 1.0  |  -   |  -   |  -   |
|     50,000     | 1.0  | 1.29 | 1.39 | 0.22 |
|    100,000     | 1.0  | 1.55 | 2.55 | 0.58 |
|   50,000,000   | 1.0  | 1.98 | 2.76 | 3.65 |
|  100,000,000   | 1.0  | 1.98 | 3.04 | 3.80 |

### 4. 迭代求 π (MPI)

运行结果

```text
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 2-mpi <<< 1000
3.141592736757
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 2-mpi <<< 10000
3.141592654423
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 2-mpi <<< 50000
3.141592653623
ubuntu@iBug-Server:~/proj/PC-2020/lab1$ mpirun -np 6 2-mpi <<< 100000
3.141592653598
ubuntu@iBug-Server:~/proj/PC-2020/lab1$
```

运行时间（ms）

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 0.01 | 0.03 | 0.08 | 0.11 |
|     10,000     | 0.02 | 0.03 | 0.04 | 0.09 |
|     50,000     | 0.41 | 0.13 | 0.09 | 0.14 |
|    100,000     | 0.90 | 0.24 | 0.19 | 0.16 |
|   50,000,000   | 115  | 54.6 | 23.8 | 24.7 |
|  100,000,000   | 209  | 103  | 50.0 | 50.3 |

加速比

| 规模 \\ 进程数 |  1   |  2   |  4   |  8   |
| :------------: | :--: | :--: | :--: | :--: |
|     1,000      | 1.0  |  -   |  -   |  -   |
|     10,000     | 1.0  |  -   |  -   |  -   |
|     50,000     | 1.0  |  -   |  -   |  -   |
|    100,000     | 1.0  |  -   |  -   |  -   |
|   50,000,000   | 1.0  | 2.11 | 4.83 | 4.66 |
|  100,000,000   | 1.0  | 2.03 | 4.18 | 4.16 |

## 分析与总结

两个算法的结构都比较简单，而实验文档里给出的 n（即问题规模）又太小，因此在测量性能这里遇到了很大的困难。我一开始采用 Bash 的 `time` 内置命令测量，但是对于 OpenMP 程序 5 ms 和 MPI 程序 400+ ms 的启动时间来说完全无法使用。然后我就 Google 找别的办法，发现 OpenMP 和 MPI 都提供了获取高精度时间的函数，就把测量时间的方法换成在读取完数据之后和输出结果之前分别定时然后取差的办法，但是根本问题还是 n 太小，测出来的结果都在 us 量级，噪音更大了。经过与助教沟通，决定在实验要求指定的参数以外额外增加测试，以便正常完成实验。出于同样的原因，对于规模过小的运行时间结果，我没有计算加速比（对应的格子填 `-`）。

另外 Open MPI 默认不会在一个节点上运行超过物理核数的任务数，因此进程数为 8 的运行命令为 `mpirun --use-hwthread-cpus -np 8 <program>`[^1]。同时由于现代的 SMT 超线程技术对计算密集型负载并不友好（这也是 Open MPI 默认取物理核数为上限的原因）， 8 线程（OpenMP）/ 8 进程（MPI）的性能有时候还不如 4 线程 / 进程，特别是 OpenMP，所以不少超算甚至都禁用了超线程技术。

[^1]: https://github.com/open-mpi/ompi/issues/6020#issuecomment-436760809
