# 《并行计算》实验 3

## 实验题目

利用 MPI 进行 N 体问题模拟

## 实验环境

- 操作系统：Ubuntu 20.04 LTS
- 编译器：GCC 9.3.0
- MPI 库：Open MPI 4.0.3
- 处理器：Intel Core i7-8850H (6 cores, 12 threads, 2.6~4.3 GHz)
- 内存：32 GiB DDR4 3200 \* 2 (64 GiB total)

## 算法设计与分析

和实验二一样，逻辑非常明确，直接模拟就行。由于实际发生的过程是连续量，因此实际模拟的办法为取一个很小的时间片（帧），假设每个时间片内各量的变化可以近似视为线性，就不用求解微分方程了，本实现中取的帧长度为 $10^{-3}\operatorname{s}$。每轮循环对每个小球求其受到除自己以外其他所有小球的引力大小，由总引力得到加速度，再由此更新速度和位置。考虑到 MPI 程序在进程之间分配任务，一个比较直观的想法就是每个进程负责一部分小球的模拟计算。由于引力大小只与相对位置有关，因此需要在所有进程间分享的数据只有每轮循环后小球的位置，速度不需要分享，经过查找资料，发现 MPI 有一个 `MPI_Allgather` 和一个 `MPI_Allgatherv` 函数，非常适合用于本实验，所以数据交换就靠这两个函数了。

实际程序中万有引力常数 $G$，小球的质量 $M$ 和模拟的时间片长度 $Frame$ 是写死在源代码里的，程序接受的输入为小球个数 $N$（必须为完全平方数）和需要模拟的时间长度（单位为秒）。

## 核心代码

参数：

```c
#define G 6.67e-11
#define MASS 1e4
#define FRAME 1e-3
```

结构体定义及帮助函数：

```c
typedef struct _Pos {
    double x, y;
} Pos;

inline double distance_2(Pos x, Pos y) {
    return (x.x - y.x) * (x.x - y.x) + (x.y - y.y) * (x.y - y.y);
}
inline double distance(Pos x, Pos y) {
    return sqrt(distance_2(x, y));
}
```

计算某个小球的受力：

```c
void compute_force(const Pos *pos, int n, int this, Pos *result) {
    result->x = result->y = 0.0;
    for (int i = 0; i < n; i++) {
        if (i == this)
            continue;
        double r2 = distance_2(pos[i], pos[this]);
        double f = G * MASS * MASS / r2;
        double r = sqrt(r2);
        double df = f / r;
        result->x += (pos[i].x - pos[this].x) * df;
        result->y += (pos[i].y - pos[this].y) * df;
    }
}
```

计算某个小球的速度：

```c
void compute_velocity(Pos *vel, const Pos *force) {
    vel->x += force->x / MASS * FRAME;
    vel->y += force->y / MASS * FRAME;
}
```

计算某个小球的位置：

```c
void compute_position(Pos *pos, const Pos *vel) {
    pos->x += vel->x * FRAME;
    pos->y += vel->y * FRAME;
}
```

初始化数据：

```c
Pos *pos = malloc(n * sizeof(Pos));
Pos *vel = malloc(this_size * sizeof(Pos));
for (int i = 0; i < this_size; i++) {
    vel[i].x = vel[i].y = 0.0;
}
for (int i = 0; i < n; i++) {
    pos[i].x = 1e-2 * (i % side - (side - 1.0) / 2.0);
    pos[i].y = 1e-2 * (i / side - (side - 1.0) / 2.0);
}
```

循环模拟：

```c
double runtime = 0.0;
Pos force;
while (runtime < duration) {
    for (int i = 0; i < this_size; i++) {
        compute_force(pos, n, this_start + i, &force);
        compute_velocity(&vel[i], &force);
    }
    for (int i = 0; i < this_size; i++) {
        compute_position(&pos[this_start + i], &vel[i]);
    }
    runtime += FRAME;

    // Communicate
    MPI_Allgatherv(MPI_IN_PLACE, this_size, MPI_Pos, pos, recvcounts, displs, MPI_Pos, MPI_COMM_WORLD);
}
```

## 实验结果

### 运行结果

```text
ubuntu@iBug-Server:~/proj/PC-2020/lab3$ mpirun -np 4 main <<< '25 5'
Object   1: (137.13033107, 137.13033107)
Object   2: (-0.42065940, -25.81719312)
Object   3: (-0.00000000, 6.83421253)
Object   4: (0.42065940, -25.81719312)
Object   5: (-137.13033107, 137.13033107)
Object   6: (-25.81719312, -0.42065940)
Object   7: (-137.16102812, -137.16102812)
Object   8: (-0.00000000, -0.29312377)
Object   9: (137.16102812, -137.16102812)
Object  10: (25.81719312, -0.42065940)
Object  11: (6.83421253, -0.00000000)
Object  12: (-0.29312377, -0.00000000)
Object  13: (-0.00000000, -0.00000000)
Object  14: (0.29312377, -0.00000000)
Object  15: (-6.83421253, -0.00000000)
Object  16: (-25.81719312, 0.42065940)
Object  17: (-137.16102811, 137.16102811)
Object  18: (-0.00000000, 0.29312377)
Object  19: (137.16102812, 137.16102812)
Object  20: (25.81719312, 0.42065940)
Object  21: (137.13033105, -137.13033105)
Object  22: (-0.42065940, 25.81719312)
Object  23: (-0.00000000, -6.83421253)
Object  24: (0.42065940, 25.81719312)
Object  25: (-137.13033107, -137.13033107)
ubuntu@iBug-Server:~/proj/PC-2020/lab3$
```

### 运行时间（ms）

运行时间的测量方式与实验 1 中的两个 MPI 程序一致，此处不再重复。

| 规模 \\ 进程数 |  1   |  2   |  4   |  6   |  8   |
| :------------: | :--: | :--: | :--: | :--: | :--: |
|    64 / 100    | 1458 | 786  | 553  | 613  | 827  |
|    256 / 10    | 2319 | 1175 | 678  | 522  | 734  |

### 加速比

| 规模 \\ 进程数 |  1   |  2   |  4   |  6   |  8   |
| :------------: | :--: | :--: | :--: | :--: | :--: |
|    64 / 100    | 1.00 | 1.85 | 2.64 | 2.38 | 1.76 |
|    256 / 10    | 1.00 | 1.97 | 3.42 | 4.44 | 3.16 |

## 分析与总结

与实验 1 中的两个 MPI 程序和实验 2 一样，在本次实验中学会了 `MPI_Allgather` 和 `MPI_Allgatherv`。

另外实验文档里写着

> 引力常数数值取6.67\*10^11

要不是因为~~做过大物实验~~，我还真信了这个鬼呢
