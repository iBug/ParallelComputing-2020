# 《并行计算》实验 2

## 实验题目

利用 MPI 进行蒙特卡洛模拟

## 实验环境

- 操作系统：Ubuntu 20.04 LTS
- 编译器：GCC 9.3.0
- MPI 库：Open MPI 4.0.3
- 处理器：Intel Core i7-8850H (6 cores, 12 threads, 2.6~4.3 GHz)
- 内存：32 GiB DDR4 3200 \* 2 (64 GiB total)

## 算法设计与分析

算法本身没有太多可以设计分析的地方，就是根据给定的规则进行模拟，因此这里的重点是 MPI 并行化。

首先在任务分配上，为了减少进程间通信的需求，尽可能将连续的车辆交给同一个进程处理，实际实现就是从 0 号车开始，每 $n/size$ 辆分配给一个进程，这样只有相邻两个进程每轮循环后需要传输一辆车的速度信息用于更新间距。

每轮循环的实际执行逻辑如下：

1. 若 $v\ge d-1$，那么更新为 $v:=d-1$，否则 $v:=v+1$
2. 若 $v\gt0$，那么以概率 $p$ 随机更新 $v:=v-1$
3. 更新 $car_i.d=car_i.d+car_{i-1}.v-car_i.v$（0 号车为最前面的那辆，且 $car_0.d$ 固定为一个很大的值，为了方便这里就取 `V_MAX+1`）
4. 进程间交换信息：第 i 个进程将其负责的最后一辆车的速度发送给第 i+1 个进程，后者据此更新其负责的最前一辆车的 $d$

最后模拟完毕时由 0 号进程收集所有结果并输出。

## 核心代码

初始化部分

```c
const int this_start = n * rank / size;
int this_size = n * (rank + 1) / size - this_start;
Car *car = malloc(this_size * sizeof(Car));
for (int i = 0; i < this_size; i++) {
    car[i].v = 0;
    car[i].d = 1;
}
if (rank == 0) {
    car[0].d = V_MAX + 1;
}
```

循环模拟的主体

```c
// Do simulation
for (int round = 0; round < rounds; round++) {
    // Speed change
    for (int i = 0; i < this_size; i++) {
        if (car[i].d - 1 <= car[i].v) {
            car[i].v = car[i].d - 1;
        } else if (car[i].v < V_MAX) {
            car[i].v++;
        }
        if (car[i].v > 0 && rand() <= p_cap) {
            car[i].v--;
        }
    }

    // Distance change
    for (int j = 1; j < this_size; j++) {
        car[j].d += car[j - 1].v - car[j].v;
    }

    // Communicate
    if (rank != size - 1) {
        MPI_Send(&car[this_size - 1].v, 1, MPI_INT, rank + 1, rank, MPI_COMM_WORLD);
    }
    if (rank != 0) {
        int that_v; // car[-1].v
        MPI_Status status;
        MPI_Recv(&that_v, 1, MPI_INT, rank - 1, rank - 1, MPI_COMM_WORLD, &status);
        car[0].d += that_v - car[0].v;
    }
}
```

结果收集

```c
// Collect results
int *recvcounts = NULL;
int *displs = NULL;
Car *cars = NULL;
if (rank == 0) {
    recvcounts = malloc(size * sizeof(*recvcounts));
    displs = malloc(size * sizeof(*displs));
    cars = malloc(n * sizeof(*cars));
}
MPI_Gather(&this_size, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);
// Produce indexes for variable-gathering
if (rank == 0) {
    displs[0] = 0;
    for (size_t i = 1; i < size; i++) {
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
}
MPI_Gatherv(car, this_size, MPI_Car, cars, recvcounts, displs, MPI_Car, 0, MPI_COMM_WORLD);
```

## 实验结果

运行时间的测量方式与实验 1 中的两个 MPI 程序一致，此处不再重复。

### 运行时间（ms）

| 规模 \\ 进程数  |  1   |  2   |  4   |  8   |
| :-------------: | :--: | :--: | :--: | :--: |
| 100,000 / 2,000 | 232  | 121  | 71.1 | 75.7 |
|  500,000 / 500  | 263  | 129  | 70.7 | 69.9 |
| 1,000,000 / 300 | 352  | 173  | 99.3 | 88.3 |

### 加速比

| 规模 \\ 进程数  |  1   |  2   |  4   |  8   |
| :-------------: | :--: | :--: | :--: | :--: |
| 100,000 / 2000  | 1.00 | 1.92 | 3.26 | 3.06 |
|  500,000 / 500  | 1.00 | 2.04 | 3.72 | 3.76 |
| 1,000,000 / 300 | 1.00 | 2.03 | 3.54 | 3.99 |

## 分析与总结

与实验 1 中的两个 MPI 程序一样，在本次实验中学会了 `MPI_Send`，`MPI_Recv`，`MPI_Gather` 和 `MPI_Gatherv`，没有别的好总结的。
