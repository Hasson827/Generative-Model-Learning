# Lec03 Constructing the Training Target

在上一节中，我们构造了流模型和扩散模型，通过模拟ODE/SDE得到轨迹$(X_t)_{0\le t\le1}$
$$
\begin{aligned}
X_0\sim p_{\text{init}},\;dX_t &= u_{t}^{\theta}(X_t)dt\quad\Longrightarrow\quad X_{1}\sim p_{\text{data}}
\\[10pt]
X_0\sim p_{\text{init}},\;dX_t &= u_{t}^{\theta}(X_t)dt+\sigma_tdW_t\quad\Longrightarrow\quad X_{1}\sim p_{\text{data}}
\end{aligned}
$$

- 其中$u_{t}^{\theta}$是一个神经网络，$\sigma_t$是一个固定的扩散系数。

显然，如果我们随机初始化神经网络$u_t^{\theta}$的参数$\theta$，那么模拟ODE/SDE只会产生无意义的结果。在机器学习中，我们需要训练神经网络，我们通过最小化损失函数$\mathcal{L}(\theta)$来实现这一点，例如**均方误差MSE**:
$$
\mathcal{L}(\theta) = \| u_{t}^{\theta}(x)- \underbrace{ u_{t}^{target}(x)}_{\text{training target}} \|^{2}
$$

- 其中$u_{t}^{target}$是我们想要逼近的训练目标。

为了导出训练算法，我们分两步进行：在本章中，我们的目标是**找到训练目标$u_{t}^{target}$的方程**。在下一章中，我们将描述一种近似训练目标$u_{t}^{target}$的算法。一般来讲，就像神经网络$u_t^{\theta}$一样，训练目标本身应该是一个向量场$u_{t}^{target}:R^{d}\times[0,1]\to R^d$。此外，$u_t^{target}$应该做我们希望$u_{t}^{\theta}$做的事情：把噪声转换为数据。因此，本章的目标是推导出训练目标$u_{t}^{ref}$的公式，使得相应的ODE/SDE能够将$p_{init}$转化为$p_{data}$。在这个过程中，我们将遇到物理学和随机微分方程的两个基本结果：**连续性方程 (Continuity Equation)**和**福克-普朗克方程 (Fokker-Planck Equation)**。和前面一样，在将ODEs推广到SDEs之前，我们将首先描述它们的关键思想。

## 3.1 Conditional and Marginal Probability Path

**关键术语**：

- Conditional: Per single data point
- Marginal: Across distribution of data points

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250328140459384.png" alt="image-20250328140459384" style="zoom:20%;" />

构造训练目标$u_t^{target}$的第一步是指定一个**概率路径 (Probability Path)**。直观地说，概率路径指定了噪声$p_{init}$和数据$p_{data}$之间的渐进插值，具体见上图。

### 狄拉克分布

对于一个样本$z\in R^{d}$，我们用$\delta_{z}$来表示**狄拉克$\delta$分布**。这是最简单的分布：从$\delta_{z}$中采样总是返回$z$，即：
$$
z\in R^{d},X\sim\delta(z)\Longrightarrow X=z
$$

### 条件概率路径

**条件概率路径**是一组在空间$R^{d}$上的分布$p_{t}(x\vert z)$，其满足以下关系：
$$
p_0(\cdot\vert z) = p_{init},\;p_1(\cdot\vert z) = \delta_{z}\;\;\;\forall z\in R^{d}
$$
换句话说，条件概率路径逐渐将单个样本转换为分布$p_{init}$。你可以将概率路径视为分布空间中的轨迹。概率路径是生成过程中的“桥梁”，它描述了分布如何从初始噪声$p_{init}$逐渐演变为目标数据$p_{data}$。

**条件概率路径$p_t(x|z)$**

- 对于一个具体的样本$z$（比如一张猫的图片），条件概率路径$p(x\vert z)$描述了从噪声到这个特定样本的演变。
- 在$t=0$时，$p_0(x\vert z)=p_{init}$，完全是噪声，与$z$无关。
- 在$t=1$时，$p_1(x\vert z)=p_{data}$，分布坍缩到$z$本身（狄拉克分布，表示确定性）。
- 这就像一个“个性化”的生成过程：给定$z$，从噪声开始，最终精确还原$z$。

### 边际概率路径

**边际概率路径$p_t(x)$**

- 边际概率路径是对所有可能的$z$（来自$p_{data}$）取平均的结果。
- 计算方式：先在数据分布中采样一个$z\sim p_{\text{data}}$，然后通过$p_t(x\vert z)$采样$x$，得到的分布就是$p_t(x)$。
- 数学上：$p_t(x)=\int{p_t(x\vert z)p_{data}(z)dz}$
- 意义：$p_t(x)$是整个生成过程在时间$t$的总体分布，反映了从噪声到数据的全局演变。

**直观理解**

- 条件概率路径是“单兵作战”，针对某个样本$z$的生成轨迹。
- 边际概率路径是“全局视角”，考虑所有样本$z$的平均效果。

注意，我们知道怎样从$p_t$中采样，但是我们不知道$p_t(x)$的概率密度，因为积分很难处理。

> **高斯条件概率路径**
>
> 一个特别流行的概率路径是高斯概率路径。这是扩散模型去噪所使用的概率路径。令$\alpha_t$，$\beta_t$为噪声调度器：两个连续可微的函数，满足$\alpha_0=\beta_1=0$和$\alpha_1=\beta_0=1$。然后我们定义条件概率路径：
> $$
> p_t(\cdot\vert z) = \mathcal{N}(\alpha_tz,\beta_t^2I_d)
> $$
> 并且根据$\alpha_t$和$\beta_t$的性质，我们有：
> $$
> \begin{aligned}
> p_0(\cdot\vert z) &= \mathcal{N}(\alpha_0z,\beta_0^2I_d) = \mathcal{N}(0,I_d)
> \\[5pt]
> p_1(\cdot\vert z) &= \mathcal{N}(\alpha_1z,\beta_1^2I_d) = \delta_z
> \\[5pt]
> \alpha_{t} = t&,\quad \beta_{t}=1-t
> \end{aligned}
> $$
> 其中我们已知方差为0，均值为$z$的分布就是$\delta_z$。因此，对于$p_t(x\vert z)$的选择满足方程$(3)$，其中$p_{init}=\mathcal{N}(0,I_d)$，因此这是一个有效的条件插值路径。高斯条件概率路径有几个有用的性质，使其特别适用于我们的目标，正因为如此，我们将用它作为条件概率路径的原型例子。在上图中，我们说明了它在图像中的应用。我们可以将在边际路径$p_t$中的采样用以下方程描述：
> $$
> z\sim p_{data},\;\epsilon\sim p_{init} = \mathcal{N}(0,I_d)\Longrightarrow x=\alpha_tz+\beta_t\epsilon\sim p_t
> $$
> 直观地看，上面的过程添加了更多的噪声，直到时间$t=0$，此时只有噪声。在下图中，我们绘制了高斯噪声和简单数据分布之间的插值路径的示例。

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250329003621787.png" alt="image-20250329003621787" style="zoom:30%;" />



## 3.2 Conditional and Marginal Vector Field

我们现在使用概率路径$p_t$的概念来为流模型构造一个训练目标$u_t^{target}$。我们的想法是从简单的组建构造出$u_t^{target}$。

**目标**：构造$u_t^{target}$使得$X_t\sim p_t$。

### 条件向量场

对于每个样本点$z$，其条件向量场$u_t^{\text{target}}(x|z)$定义了一个特定的 ODE，使得轨迹$X_t$的分布从$p_{init}$演变为$p_{t}(\cdot\vert z)$，直到最终的$p_{1}(\cdot|z) = p_{\text{data}}=\delta(z)$。它是一个“个性化”的向量场，专注于从噪声生成某个特定样本$z$。因此我们可以给出其定义：
$$
u_{t}^{\text{target}}(x|z)\qquad(0\le t\le 1,\quad x,z\in R^{d})
\\[10pt]
s.t.\quad X_{0}\sim p_{\text{init}},\; \frac{d}{dt}X_{t} = u_{t}^{\text{target}}(X_{t}|z)\Longrightarrow X_{t}\sim p_{t}(\cdot|z)\;(0\le t\le1)
$$

### 边际向量场

我们已经知道了条件向量场的定义，则边际向量场就是将条件向量场边际化。**边际向量场**$u_t^{target}(x)$就由下式定义：
$$
u_{t}^{target}(x)=\int u_t^{target}(x\vert z)\frac{p_t(x\vert z)p_{data}(z)}{p_t(x)}dz
$$
边际向量场$u_t^{target}(x)$可以理解为对所有$z$的加权平均，权重是后验概率$\frac{p_t(x|z)p_{data}(z)}{p_t(x)}$(由贝叶斯公式得来)。它驱动全局轨迹$X_t$沿着边际概率路径$p_t$演变。

则边际概率路径就是：
$$
X_0\sim p_{\text{init}},\;\frac{d}{dt}X_t=u_t^{\text{target}}(X_t)\Rightarrow X_t\sim p_t\;\;(0\le t\le1)
$$
特别的，对于这个ODE，$X_0\sim p_{\text{init}}$，因此我们会说：“$u_t^{\text{target}}$将噪声$p_{\text{init}}$转化为数据$p_{\text{data}}$”

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250329004839227.png" alt="image-20250329004839227" style="zoom:30%;" />

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250329004850545.png" alt="image-20250329004850545" style="zoom:30%;" />

> 该图是对于边缘化技巧的说明。用ODE模拟概率路径，数据分布$p_{\text{data}}$为背景中的蓝色部分。高斯分布$p_{\text{init}}$是红色部分。
>
> 上面一行是条件概率路径。左：条件路径$p_t(\cdot\vert z)$的真值样本。中：随时间变化的ODE样本。右：模拟方程中目标为$u_t^{\text{target}}(x|z)$的ODE的轨迹。
>
> 下面一行是边际概率路径。左：来自$p_t$的真值样本。中：随时间变化的ODE样本。右：利用边际向量场$u_t^{\text{flow}}(x)$模拟ODE的轨迹。
>
> 可以看出，条件向量成遵循条件概率路径，边际向量场遵循边际概率路径。

接下来通过为高斯概率路径的运行示例推导一个条件向量场$u_{t}(x\vert z)$来说明这一点。

> **Example:高斯概率路径的目标ODE**
>
> 和往常一样，令$p_t(\cdot\vert z)=\mathcal{N}(\alpha_tz,\beta_t^2I_d)$，令$\overset{\cdot}{\alpha}_t=\partial_t\alpha_t,\:\overset{\cdot}{\beta}_t=\partial_t\beta_t$表示$\alpha_t$和$\beta_t$关于时间$t$的导数。则该式的条件高斯概率向量场由下式给出：
> $$
> u_t^{target}(x\vert z)=\left( \overset{\cdot}{\alpha}_t-\frac{\overset{\cdot}{\beta}_t}{\beta_t}\alpha_t \right)z+\frac{\overset{\cdot}{\beta}_t}{\beta_t}x
> $$
> 这是在边缘化技巧基础上的一个有效的条件向量场模型：其ODE轨迹$X_t$满足$X_t\sim p_t(\cdot\vert z)=\mathcal{N}(\alpha_tz,\beta_t^2I_d)$且$X_0\sim\mathcal{N}(0,I_d)$。在上图中，我们通过比较来自条件概率路径（基本真值）的样本与来自该流的模拟ODE轨迹的样本，直观地证实了这一点。现在我们来证明一下：
>
> **证明**：首先通过以下方程构建一个条件流模型$\psi_t^{target}(x\vert z)$：
> $$
> \psi_t^{target}(x\vert z) = \alpha_tz+\beta_tx
> $$
> 如果$X_t$是这个条件流模型的ODE轨迹，那么根据定义有：
> $$
> X_t=\psi_t^{target}(X_0\vert z)=\alpha_tz+\beta_tX_0\sim\mathcal{N}(\alpha_tz,\beta^2I_d)=p_t(\cdot\vert z)
> $$
> 我们得出的结论是，轨迹的分布与条件概率路径相似。剩下的就是从$\psi_t^{target}(x\vert z)$中提取向量场$u_t^{target}(x\vert z)$。根据流的定义，以下方程成立：
> $$
> \begin{aligned}
> \frac{d}{dt}\psi_t^{target}(x\vert z)&=u_{t}^{target}(\psi_t^{target}(x\vert z)\vert z)
> \\[10pt]
> \overset{\cdot}{\alpha}_tz+\overset{\cdot}{\beta}_tx&=u_t^{target}(\alpha_tz+\beta_tx\vert z)
> \\[10pt]
> \overset{\cdot}{\alpha}_tz+\overset{\cdot}{\beta}_t\left( \frac{x-\alpha_tz}{\beta_t} \right)z&=u_t^{target}(x\vert z)
> \\[10pt]
> \left( \overset{\cdot}{\alpha}_t-\frac{\overset{\cdot}{\beta}_t}{\beta_t}\alpha_t \right)z+\frac{\overset{\cdot}{\beta}_t}{\beta_t}x&=u_t^{target}(x\vert z)
> \end{aligned}
> $$
> 其中第一项$\left(  \overset{\cdot}{\alpha}_t-\frac{\overset{\cdot}{\beta}_t}{\beta_t}\alpha_t  \right)z$的意义是朝目标$z$的方向移动，第二项$\frac{\overset{\cdot}{\beta}_t}{\beta_t}x$的意义是根据当前点$x$调整速度（通常$\overset{\cdot}{\beta}_t\lt0$，表示收缩）

本节的剩余部分将通过**连续性方程**证明边缘化技巧的合理性。为了解释它，我们需要使用散度运算符$div$，我们将其定义为：
$$
div(v_t)(x)=\sum_{i=1}^{d}\frac{\partial}{\partial x_i}v_t(x)
$$

> **连续性方程**
>
> 考虑一个有向量空间$u_t^{target}$的流模型，其满足$X_0\sim p_{init}$。那么：
> $$
> \forall t\in[0,1],\;X_t\sim p_t
> \\
> \Updownarrow
> \\
> \forall x\in R^d,t\in[0,1],\; \underbrace{ \partial_tp_t(x)}_{\text{Change of Probability Mass at x}}=\underbrace{-div(p_t u_t)(x)}_{\text{Outflow - Inflow of Probability Mass from u}}
> $$
> 其中$\partial_tp_t=\frac{d}{dt}p_t(x)$表示$p_t(x)$关于时间的偏导数。上式即为**连续性方程**。
>
> 意义：这是质量守恒的物理原理，确保概率密度$p_t(x)$的变化与向量场$u_t^{target}$的“流动”一致。
>
> **证明**：
> $$
> \begin{aligned}
> \partial_tp_t(x)&=\partial_t\int{p_t(x\vert z)p_{data}(z)dz}
> \\[10pt]
> &=\int{\partial_t p_t(x\vert z)p_{data}(z)dz}
> \\[10pt]
> &=\int{-div\left(p_t(\cdot\vert z)u_t(\cdot\vert z)\right)(x)p_{data}(z)dz}
> \\[10pt]
> &=-div\left(\int{p_t(x\vert z)u_t^{target}(x\vert z))(x)p_{data}(z)dz}\right)
> \\[10pt]
> &=-div\left( p_t(x)\int u_t^{target}(x\vert z)\frac{p_t(x\vert z)p_{data}(z)}{p_t(x)}dz \right)(x)
> \\[10pt]
> &=-div(p_tu_t^{target})(x)
> \end{aligned}
> $$

## Conditional and Marginal Score Function

### 分数函数的定义

我们定义$p_t$的**边际分数函数**为$\nabla\log{p_{t}(x)}$，其意义如下：

- 它表示概率密度$p_t(x)$变化最快的方向。
- 在生成模型中，分数函数可以看作“去噪方向”，告诉我们如何从当前点$x$调整以更接近$p_t(x)$的高概率区域。

### 条件分数函数

很显然，条件分数函数由以下公式定义：
$$
\nabla \log{p_{t}(x|z)}
$$

### 边际分数函数

和之前条件向量场和边际向量场之间的转化相似，我们也可以用条件分数函数表示边际分数函数：
$$
\begin{aligned}
\nabla\log{p_t(x)}&=\frac{\nabla p_t(x)}{p_t(x)}
\\[10pt]
&=\frac{\nabla\int{p_t(x|z)p_{data}(z)dz}}{p_t(x)}
\\[10pt]
&=\frac{\int\nabla p_t(x|z)p_{data}(z)dz}{p_t(x)}
\\[10pt]
&=\int\nabla\log{p_t(x|z)}\frac{p_t(x|z)p_{data}(z)}{p_t(x)}dz
\end{aligned}
$$

### 扩展至SDE

我们可以用这种定义把ODE扩展到SDE，将条件向量场$u_{t}^{target}(x\vert z)$和边际向量场$u_{t}^{target}(x)$如之前一样定义，然后定义一个扩散系数$\sigma_t\ge0$，我们可以通过相同的概率路径构造一个SDE：

$$
X_0\sim p_{init},\;dX_t=\left[ u_t^{target}(X_t)+\frac{\sigma_{t}^{2}}{2}\nabla{\log{p_t(X_t)}} \right]dt+\sigma_tdW_t
\\[10pt]
\Rightarrow X_t\sim p_t\;(0\le t\le1)
$$
**直观理解**：

- $u_t^{target}$是确定性漂移，沿概率路径移动。
- $\sigma_tdW_t$是随机扰动，引入噪声。
- $\frac{\sigma_{t}^{2}}{2}\nabla{\log{p_t(X_t)}}$是“去噪修正项”，抵消布朗运动带来的扩散效应，确保$X_t$仍遵循$p_t$

**结果**：轨迹$X_T$的分布仍然是$p_t$。

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250331184817638.png" alt="image-20250331184817638" style="zoom:30%;" />

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250331184831620.png" alt="image-20250331184831620" style="zoom:30%;" />

> **高斯概率路径的分数函数**
>
> 对于高斯路径$p_t(x|z)=\mathcal{N}(x;\alpha_tz,\beta_t^2I_d)$，我们可以利用高斯概率密度来得到：
> $$
> \nabla\log{p_t(x|z)}=\nabla\log{\mathcal{N}(x;\alpha_tz,\beta_t^2I_d)}=-\frac{x-\alpha_tz}{\beta_t^2}
> $$
> 其分数是关于$x$的线性函数，这是高斯分布的一个特殊性质。

## 总结

### Conditional Prob. Path, Vector Field, and Score

|                              | Notation                     | Key Property                                        | Gaussian Example                                             |
| ---------------------------- | ---------------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| Conditional Probability Path | $p_{t}(\cdot|z)$             | Interpolates $p_{\text{init}}$ and a data point $z$ | $\mathcal{N}(\alpha_{t}z,\beta_{t}^{2}I_{d})$                |
| Conditional Vector Field     | $u_{t}^{\text{target}}(x|z)$ | ODE follows conditional path                        | $\left( \overset{\cdot}{\alpha}_t-\frac{\overset{\cdot}{\beta}_t}{\beta_t}\alpha_t \right)z+\frac{\overset{\cdot}{\beta}_t}{\beta_t}x$ |
| Conditional Score Function   | $\nabla{\log{p_{t}(x|z)}}$   | Gradient of log-likelihood                          | $-\frac{x-\alpha_tz}{\beta_t^2}$                             |

### Marginal Prob. Path, Vector Field, and Score

|                           | Notation                   | Key property                                         | Formula                                                      |
| ------------------------- | -------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Marginal Probability Path | $p_{t}$                    | Interpolates $p_{\text{init}}$ and $p_{\text{data}}$ | $p_t(x)=\int{p_t(x\vert z)p_{data}(z)dz}$                    |
| Marginal Vector Field     | $u_{t}^{\text{target}}(x)$ | ODE follows marginal path                            | $u_{t}^{\text{target}}(x)=\int u_t^{\text{target}}(x\vert z)\frac{p_t(x|z)p_{\text{data}}(z)}{p_t(x)}dz$ |
| Marginal Score Function   | $\nabla{\log{p_{t}(x)}}$   | Can be used to convert ODE target to SDE             | $\nabla\log{p_t(x)} = \int{\nabla{\log{p_{t}(x|z)}}\frac{p_{t}(x|z)p_{\text{data}}(z)}{p_{t}(x)}dz}$ |

