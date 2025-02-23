# 扩散模型的一般概念

## 1. 定义

2015年的论文将扩散模型描述如下：

- The essential idea is to systematically and slowly destroy structure in a data distribution through an iterative **forward diffusion process**. We then learn a **reverse diffusion process** that restores structure in data, yielding a highly flexible and tractable generative model of the data.

- 其关键理念在于通过一个迭代式的**正向扩散流程**，有系统且逐步地破坏数据分布中的结构。随后，我们学习一种**反向扩散流程**，用于恢复数据中的结构，由此生成一个高度灵活且易于处理的关于数据的生成模型。 

<img src="imgs/Basic_Idea.png">

## 2. 前向过程 (Foward Process)

我们从原始图像开始一步一步地向图像中添加越来越多的噪声，如果我们重复此操作足够多次，图像将变成纯噪声。2015年的论文中决定从正态分布中采样噪声，这一点在后续一直没有变化。

**注意：** 在前向过程中，我们不会在每一步都是用相同数量的噪声，而是由一个缩放均值和方差的调度器来调节的，这确保了**随着噪声的添加，方差不会爆炸。**

从一张图片变成纯噪声是相当简单的过程，但是学习相反的过程是极其困难的

## 3. 反向扩散过程 (Reverse Diffusion Process)

神经网络学习怎样逐步从图像中去除噪声，这样我们就可以给模型一个仅由从正态分布中采样得到的噪声图像，让模型逐渐去除噪声，直到我们得到一个清晰的图像。

## 4. 神经网络

2020年的DDPM论文指出神经网络可以预测的三个变量：

1. 每个时间步的噪声平均值。
2. 直接预测原始图像。(X)
3. 直接预测图像中的噪声，然后可以从噪声图像中减去噪声，得到噪声少一点的图像。

大部分人都选择了第三个

# 背后的数学

# 四篇论文的不同改进之处

# 总结