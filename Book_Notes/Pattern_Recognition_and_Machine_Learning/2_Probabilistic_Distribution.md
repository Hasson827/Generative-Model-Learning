# Probabilistic Distribution

## 基本概念

### 先验分布

先验分布（Prior Distribution）是你对某个事情的**初始猜测**，是在你看到具体数据之前，根据已有知识或经验给出的概率分布。换句话说，它是你对某个未知量（比如一个参数）的“先入为主”的看法。

**举个例子**：  
假设你在猜一个朋友有多大概率会迟到。你知道他平时挺准时的，所以你可能会猜他有80%的概率会准时，20%的概率会迟到。这个“80%准时、20%迟到”的分布就是你的**先验分布**，它基于你对他的了解，而不是当天的具体情况。

### 似然函数

似然函数（Likelihood Function）是**数据给你提供的证据**，它描述了在不同的假设下，当前观察到的数据有多“合理”。注意，似然函数不是概率分布，它只是告诉你“如果某个假设是真的，看到这些数据的可能性有多大”。

**回到例子**：  
假如你今天约了朋友见面，结果他迟到了。你会想：“如果他是个准时的人（80%准时），他迟到的可能性是多少？如果他是个常迟到的人（比如20%准时），他迟到的可能性又是多少？”  

- 如果他准时概率高（80%），他迟到这件事看起来就不太“合理”，似然值低。  
- 如果他准时概率低（20%），他迟到就显得很“合理”，似然值高。  
这个衡量“合理性”的东西就是**似然函数**，它根据今天的迟到数据，帮你评估不同的假设。

### 什么是后验分布？

后验分布（Posterior Distribution）是你**综合了先验和数据后的更新信念**。它结合了你的初始猜测（先验分布）和新证据（似然函数），告诉你基于目前所有信息，某个假设的概率是多少。

**继续例子**：  
你原本认为朋友80%准时（先验），但今天他迟到了（数据）。通过似然函数，你发现“他准时概率高”这个假设不太能解释今天的迟到，而“他准时概率低”更符合数据。于是，你更新了看法，可能得出结论：他准时概率也许只有50%了。这个新的分布（50%准时、50%迟到）就是**后验分布**。

### 三者之间的关系

可以用一个公式简单表达它们的关系（这就是贝叶斯定理）：
**后验分布 ∝ 先验分布 × 似然函数**
（“∝”表示“正比于”，实际还需要一个归一化常数，但我们先忽略细节）

- **先验分布**是你开始时的信念（80%准时）
- **似然函数**是数据对这些信念的“打分”（迟到这件事更支持哪种假设）
- **后验分布**是你根据数据调整后的新信念（综合后可能是50%准时）  

用生活化的比喻来说：  

- 先验是你对朋友的“人设”判断。  
- 似然是他今天的表现给你的“线索”。  
- 后验是你根据线索修正后的“最新人设”。

### 各自的对应

1. **先验分布**：对应的是**你的主观知识或假设**。  它是起点，反映你对未知量（参数）的预期。
2. **似然函数**：对应的是**观测数据**。  它是桥梁，告诉你数据如何支持或反驳你的假设。
3. **后验分布**：对应的是**更新后的结论**。  它是终点，综合了先验和数据，给你一个更贴近真相的答案。

## 二进制变量（0-1变量）

### 基本形式

考虑一个二进制随机变量 $x \in \left\{ 0,1 \right\}$ ，其取1的概率为 $\mu \in [0,1]$ ，则我们能够给出下式：

$$
\begin{aligned}
p(x=1\,|\,\mu) &= \mu \\
p(x=0\,|\,\mu) &= 1-\mu
\end{aligned}
$$

即在给定 $\mu$ ，也就是已知 $\mu$ 的情况下，我们能够知道 $p(x=1)=\mu$ 且 $p(x=0)=1-\mu$ ，这是十分简单且显然的事情。

于是我们能够将上述两式统一得到**伯努利分布(Bernoulli Distribution)**

$$
Bern(x|\mu) = \mu^{x}(1-\mu)^{1-x}
$$

并且十分容易可证明它是归一化的，且其均值和方差由下式给出：

$$
\begin{aligned}
E[x] &= \mu
\\
var[x] &= \mu (1-\mu)
\end{aligned}
$$

### 似然函数

现在假设我们有一个数据集 $\mathcal{D} = \left\{ x_{1},\dots,x_{N} \right\}$ 并且每个样本之间相互独立，那么我们可以得到其似然函数

$$
p(\mathcal{D} | \mu) = \prod_{n=1}^{N}p(x_{n}|\mu) = \prod_{n=1}^{N}\mu^{x_{n}}(1-\mu)^{x_{n}}
$$

我们要将似然函数最大化，等价于将对数似然函数最大化，因此我们需要最大化以下函数：

$$
\ln p(\mathcal{D} | \mu) =\sum_{n=1}^{N} \ln p(x_{n}|\mu) = \sum_{n=1}^{N} x_{n}\ln\mu \,+\, (1-x_{n}) \ln (1-\mu)
$$

将该函数对 $\mu$ 求导可求出最大似然估计：

$$
\mu_{ML} = \frac{1}{N} \sum_{n=1}^{N} x_{n}
$$

将测试时 $x=1$ 的次数设为 $m$ ，则原式可以化为：

$$
\mu_{ML} = \frac{m}{N}
$$

对于 $m$ 在 $N$ 中的分布，我们能给出二项分布公式：

$$
Bin(m|\mu ,N) = \binom{N}{m} \mu^{m}(1-\mu)^{N-m}
$$

其均值和方差由下式给出：

$$
\begin{aligned}
E[m] &= \sum_{m=0}^{N}mBin(m|\mu,N) = \mu N
\\
var[m] &= \sum_{m=0}^{N}(m-E[m])^{2}Bin(m|\mu,N) = N\mu(1-\mu)
\end{aligned}
$$

### $\beta$ 分布

首先我们在二项分布的基础上引入贝叶斯视角，在贝叶斯统计中，我们不把 $p$ 当成固定的未知数，而是把它看作一个随机变量，拥有自己的概率分布。
这个分布就是“先验分布”。当我们通过试验（比如二项分布的试验）收集到数据后，可以更新这个分布，得到“后验分布”。而 $\beta$ 分布恰好是描述 $p$ 的自然选择。

#### 基本形式

$$
Beta(\mu|a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}
$$

其中

- $\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}$ 是归一化常数，使得该分布的积分为1。
- $a$ 和 $b$ 是超参数，控制概率分布函数的形状，也控制着 $\mu$ 。

它的灵活性很强：通过调整 $a$ 和 $b$ ，可以描述从均匀分布到单峰分布，甚至是 U 形分布。

#### $a$ 和 $b$ 对分布的影响

1. **均值和方差**

$$
\begin{aligned}
\mu &= \frac{a}{a+b}
\\
\sigma^{2} &= \frac{ab}{(a+b)^{2}(a+b+1)}
\end{aligned}
$$

均值随着 $a$ 相对 $b$ 的比例增大而向 1 靠拢，反之向 0 靠拢。方差则在 $a$ 和 $b$ 都较小时较大，随着两者之和增加而减小。

2. **形状特性**

- 当 $a = b$ 时，分布关于 $x = 0.5$ 对称。
  - $a = b = 1$：均匀分布（平坦）。
  - $a = b > 1$：分布呈钟形，峰值在 0.5，$a, b$ 越大越尖锐。
  - $a = b < 1$：U 形分布，两端高中间低。
- 当 $a < b$ 时，分布偏向左侧（峰值靠近 0）。
- 当 $a > b$ 时，分布偏向右侧（峰值靠近 1）。
- 当 $a < 1, b < 1$ 时，分布在 0 和 1 附近有尖峰，形成 U 形。
- 当 $a > 1, b > 1$ 时，分布更集中在中间，呈现单峰。

3. **边界行为**

- 若 $a < 1$，则 $x = 0$ 处趋向无穷（密度激增）。
- 若 $b < 1$，则 $x = 1$ 处趋向无穷。
- 若 $a > 1$，则 $x = 0$ 处密度为 0；若 $b > 1$，则 $x = 1$ 处密度为 0。

4. **特殊情况**

- $a = 1, b = 1$：均匀分布。
- $a = 1, b > 1$：密度从 0 到 1 单调递减。
- $a > 1, b = 1$：密度从 0 到 1 单调递增。
- $a, b \to \infty$（且 $a/b$ 保持常数）：分布趋向正态分布（集中于均值附近）。

5. **直观理解**

- $a$ 可以看作“正向事件”的强度，$b$ 是“反向事件”的强度。比如在贝叶斯统计中，Beta 分布常用于描述成功概率的先验分布，$a$ 和 $b$ 分别对应成功和失败的“伪计数”。
- $a + b$ 越大，分布越集中，表示“信息量”越多；$a + b$ 越小，分布越分散，表示不确定性更高。

总结来说，$a$ 和 $b$ 共同决定了 Beta 分布的对称性、偏态、峰值位置以及边界行为。通过调整这两个参数，可以灵活地塑造分布的形态以适应不同场景。

#### 从二项分布到 $\beta$ 分布的桥梁

假设我们用二项分布做实验，进行了 $n$ 次试验，成功了 $k$ 次。现在我们想用贝叶斯方法推断 $p$ 的分布：

- 先验：假设 $p$ 服从 $\beta$ 分布，比如 $Beta(a,b)$ 这是我们的初始猜测。
- 似然：实验数据服从二项分布，似然函数是 $P(k|n,p) \propto p^{k} (1-p)^{n-k}$
- 后验：根据贝叶斯定理，后验分布正比于先验乘以似然，即 $P(p|k,n) \propto P(p)P(k|n,p)$ 。代入 $\beta$ 分布和二项分布的表达式， $P(p|k,n) \propto p^{a-1}(1-p)^{b-1} p^{k}(1-p)^{n-k}$ ，合并同类项可得：

$$
P(p|k,n) \propto p^{a+k-1} (1-p)^{b+n-k-1} = Beta(a+k,b+n-k)
$$

所以后验分布仍然是一个 $\beta$ 分布，参数从 $(a,b)$ 更新为 $(a+k,b+n-k)$ ，其实从抛硬币的角度来看，就是 $a \to a+head$ 和 $b \to b+tail$

#### 逻辑的直观解释

- $a$ 和 $b$ 的含义：你可以把 $a$ 看作“虚拟的成功次数”，$b$ 看作“虚拟的失败次数”。初始的 $\beta$ 分布 $Beta(a,b)$ 是基于这些“虚拟数据”的猜测。
- 试验的贡献：二项分布提供了真实的试验数据（ $k$ 次成功， $n−k$ 次失败）。这些数据被加到先验的“虚拟计数”上，更新了我们的信念。
- 结果：后验分布 $Beta(a+k,b+n−k)$ 是结合了先验信息和观测数据的自然延续。

#### 为什么这很优雅？

- 共轭性：二项分布和 Beta 分布的数学形式天然匹配，更新后还是 $\beta$ 分布，不需要复杂的数值计算。
- 连续性：二项分布是离散的，描述具体的成功次数；$\beta$ 分布是连续的，描述 $p$ 的不确定性。它们通过贝叶斯更新连接起来。
- 直觉：这就像你在掷硬币时，先凭感觉猜正面的概率，然后根据实际掷的结果调整猜测，最后得到一个更靠谱的概率分布。

## 多元离散变量

### 基本形式

首先考虑多元变量的表示问题，例如如果变量由 $K$ 个可能值？

- 第一种可能是每一个可能值对应 $1 \dots K$ 中的一种，这实际上是不可行的，因为这样很难表示变量之间可能存在的相互关系
- 第二种可能是把每一个样本对应到一个含有 $k$ 个元素的列向量，称为 "1-of-K scheme" ，具体映射方式如下：

现在抛掷了一枚骰子，由于它具有六种可能性，所以每抛一次（一个样本）就对应一个含有六个元素的列向量 $\mathbf{x}$ ，如果一次骰子投出 3 ，那么对应的样本如下：

$$
\mathbf{x} = (0,0,1,0,0,0)^{T}
$$

明确以上表达方式后，我们就能够表达多元离散变量的概率分布

假设 $\mathbf{x} \in \left\{ 0,1,\dots,N \right\}$

$$
\begin{aligned}
p(\mathbf{x | \mu}) &= \prod_{k=1}^{K}(\mu_{k})^{x_{k}}
\\
\mathbf{\mu} &= (\mu_{1},\dots,\mu_{K}) ^{T}
\end{aligned}
$$

其中

- $\mu_{k} \ge 0$
- $\textstyle \sum_{k}\mu_{k} = 1$

其均值可用以下方式表述：

$$
E[\mathbf{x}|\mathbf{\mu}] = \sum_{\mathbf{x}}p(\mathbf{x|\mu})\mathbf{x} = (\mu_{1},\dots,\mu_{M})^{T} = \mathbf{\mu}
$$

### 似然函数

假设现在有一个数据集 $\mathcal{D} = \left\{  x_{1},\dots,x_{N} \right\}$ ，那么我们可以给出其似然函数：

$$
p(\mathcal{D}|\mathbf{\mu}) = \prod_{n=1}^{N} \prod_{k=1}^{K}\mu_{k}^{x_{nk}} = \prod_{k=1}^{K} \mu_{k}^{(\textstyle \sum_{n}x_{nk})} = \prod_{k=1}^{K}\mu_{k}^{m_{k}}
$$

参考以下矩阵可能会好理解一些：

$$
\begin{bmatrix}
x_{11}&x_{12}&\dots&x_{1K}\\
x_{21}&x_{22}&\dots&x_{2K}\\
\vdots&\vdots&\ddots&\vdots\\
x_{N1}&x_{N2}&\dots&x_{NK}
\end{bmatrix}
$$

其中

- 每一行是一个样本点，每一列代表一种可能性
- 可以理解为 $x_{n}$ 是行号，而 $\mu_{k}$ 是列号
- 公式中的 $m_{k}$ 指的是每一列的元素值之和

对似然函数取对数并对 $\mu$ 求偏导，由于此处 $\mu$ 有归一化要求，因此我们采用拉格朗日乘子方法：

$$
\ln p(\mathcal{D}|\mu) = \sum_{k=1}^{K}m_{k} \ln\mu_{k}
$$

最大化以下表达式：

$$
L(\mu,\lambda) = \sum_{k=1}^{K}m_{k} \ln\mu_{k} + \lambda \left( \sum_{k=1}^{K}\mu_{k} \,-\,1 \right)
$$

即：

$$
\forall k \in \left\{ 1,\dots,K \right\} \,\,,\,\,\frac{\partial L(\mu,\lambda)}{\partial \mu_{k}} = \frac{m_{k}}{\mu_{k}}+\lambda = 0
\\[10pt]
\forall k \in \left\{ 1,\dots,K \right\} \,\,,\,\,\mu_{k} = -\frac{m_{k}}{\lambda}
\\[10pt]
1 = \sum_{k=1}^{K}\mu_{k} = -\frac{1}{\lambda}\sum_{k=1}^{K}m_{k} = -\frac{N}{\lambda}
\\[10pt]
\lambda = -N
\\[10pt]
\mu_{k} = \frac{m_{k}}{N}
$$

取第 $k$ 个值的概率也就是第 $k$ 个量出现的比例

### 多项式分布(Multinomial Distribution)

实际上就是前文中 $m_{k}$ 的似然函数

$$
Mult(m_{1},m_{2},\dots,m_{K}|\mu,N) = \binom{N}{m_{1}m_{2}\dots m_{K}}\prod_{k=1}^{K}\mu_{k}^{m_{k}}
$$

其中

$$
\binom{N}{m_{1}m_{2}\dots m_{K}} = \frac{N!}{m_{1}!m_{2}!\dots m_{K}!}
$$

是用于归一化的，后半部分才决定了这个函数的形状

注意，对于 $m_{k}$ ，我们有如下约束：

$$
\sum_{k=1}^{K}m_{k} = N
$$

### 狄利克雷分布(Dirichlet Distribution)

现在，我们为多项式分布的参数 $\mu_{k}$ 引入一系列先验分布。通过检查多项式分布的形式，我们可以看到共轭先验由下式给出：

$$
p(\mathbf{\mu}|\mathbf{\alpha}) \propto \prod_{k=1}^{K}\mu_{k}^{\alpha_{k}-1}\,\,,\,\,\mu_{k} \in [0,1]\,\,,\,\,
\textstyle\sum_{k} \mu_{k}=1
$$

其中 $\alpha_{1},\dots,\alpha_{K}$ 是分布的参数， $\alpha = (\alpha_{1},\dots,\alpha_{K})^{T}$

现在我们加上这个式子的归一化常数，就得到了狄利克雷分布：

$$
Dir(\mathbf{\mu}|\mathbf{\alpha}) = \frac{\Gamma(\alpha_{0})}{\Gamma(\alpha_{1})\cdots\Gamma(\alpha_{K})} \prod_{k=1}^{K}\mu_{k}^{\alpha_{k}-1}
$$

其中

$$
\alpha_{0} = \sum_{k=1}^{K}\alpha_{k}
$$

将先验分布与似然函数相乘可以得到后验分布：

$$
p(\mu |\mathcal{D},\alpha) \propto p(D|\mu)p(\mu|\alpha) = \prod_{k=1}^{K}\mu_{k}^{\alpha_{k}+m_{k}-1}
$$

后验分布再次采用狄利克雷分布的形式，证实了狄利克雷确实是多项式的共轭先验。这允许我们通过与 $Dir(\mu|\alpha)$ 的比较来确定归一化系数，因此：

$$
\begin{aligned}
p(\mu|\mathcal{D},\alpha) &= Dir(\mu|\alpha + \mathbf{m})
\\
&= \frac{\Gamma(\alpha_{0}+N)}{\Gamma(\alpha_{1}+m_{1})\cdots\Gamma(\alpha_{K}+m_{K})}\prod_{k=1}^{K}\mu_{k}^{\alpha_{k}+m_{k}-1}
\end{aligned}
$$

### 狄利克雷分布和贝塔分布的联系

#### 1. **维度上的联系**

- **贝塔分布**是定义在区间 $[0, 1]$ 上的概率分布，适用于描述单一概率变量的分布。
- **狄利克雷分布**是贝塔分布的高维推广，适用于描述多个概率变量的联合分布，且这些变量满足和为1的约束条件。

#### 2. **参数上的联系**

- 贝塔分布有两个参数 $\alpha$ 和 $\beta$，控制分布的形态。
- 狄利克雷分布有多个参数 $\alpha_1, \alpha_2, \dots, \alpha_K$，每个参数对应一个维度。当 $K=2$ 时，狄利克雷分布退化为贝塔分布。

#### 3. **概率密度函数的联系**

- 贝塔分布的概率密度函数为：

     $$
     f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
     $$

- 狄利克雷分布的概率密度函数为：

     $$
     f(\mathbf{x}; \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{i=1}^K x_i^{\alpha_i-1}
     $$

当 $K=2$ 时，狄利克雷分布简化为贝塔分布。

#### 4. **应用场景的联系**

- 贝塔分布常用于二项分布的共轭先验。
- 狄利克雷分布常用于多项分布的共轭先验。

#### 5. **几何直观**

- 贝塔分布描述的是单位区间上的概率分布。
- 狄利克雷分布描述的是单纯形（simplex）上的概率分布，单纯形是高维空间中的几何对象。

## 高斯分布

### 基本形式

$$
\mathcal{N}(x|\mu,\sigma^{2}) = \frac{1}{\sqrt{2\pi \sigma^{2}}}\,\exp\left\{{-\frac{(x-\mu)^2}{2\sigma^{2}}}\right\}
\\[10pt]
\mathcal{N}(\mathbf{x}|\mu,\Sigma) = \frac{1}{(2\pi)^{\frac{D}{2}}} \frac{1}{|\Sigma|^{\frac{1}{2}}}\,\exp\left\{{-\frac{1}{2}(\mathbf{x}-\mu)^{T}\Sigma^{-1}(\mathbf{x}-\mu)}\right\}
$$

其中：

- $D$ 是维度
- $\Sigma$ 是一个 $D\times D$ 的协方差矩阵，代表方差
- $\mu$ 是一个维度为 $D$ 的向量，代表均值
- $|\Sigma|$ 是 $\Sigma$ 的行列式

我们考虑高斯分布的几何形式，高斯分布对 $\mathbf{x}$ 的函数依赖性是通过以下二次形式得到的：

$$
\Delta^{2} = (\mathbf{x}-\mu)^{T}\Sigma^{-1}(\mathbf{x}-\mu)
$$

$\Delta$ 的量称为从 $\mu$ 到 $\mathbf{x}$ 的马氏距离，当 $\Sigma$ 是单位矩阵时，它退化为欧几里得距离。高斯分布在 $\mathbf{x}$ 空间中的表面上是恒定的，而这种二次形式是恒定的。

协方差矩阵 $\Sigma$ 一定是对称矩阵

现在考虑协方差矩阵 $\Sigma$ 的特征向量方程，即：

$$
\Sigma u_{i} = \lambda_{i}u_{i}\,\,\,\,,\,\,\,\,i \in \left\{ 1,2,\dots,D \right\}
$$

由对称矩阵的性质我们可知：

- $\forall\,i \in \left\{ 1,2,\dots,D \right\}\,\,,\,\,\lambda_{i} \in \mathcal{R}$
- 所有的特征向量 $u_{i}$ 组成一个正交基，即 $u_{i}u_{j} = I_{ij}$ ，其中 $I_{ij}$ 是单位矩阵中第 $i$ 行，第 $j$ 列的元素。

因此我们可以得到协方差矩阵 $\Sigma$ 与其特征向量、特征值之间的关系：

$$
\Sigma = \Sigma I \,=\, \Sigma \mathbf{U U^{T}} \,=\, \Sigma \sum_{i=1}^{D}u_{i}u_{i}^{T} \,=\, \sum_{i=1}^{D}\lambda_{i}u_{i}u_{i}^{T}
\\[10pt]
\mathbf{U} = \begin{bmatrix}
u_{1}^{T}\\
u_{2}^{T}\\
\vdots\\
u_{D}^{T}
\end{bmatrix}
\,\,\,,\,\,\,\mathbf{UU^{T}=U^{T}U} = I
$$

类似的，我们能够表达 $\Sigma^{-1}$ ：

$$
\Sigma^{-1} \,=\, \sum_{i=1}^{D}\frac{1}{\lambda_{i}}u_{i}u_{i}^{T}
$$

我们将这些式子代入 $\Delta^{2}$ 的计算公式可以得到：

$$
\Delta^{2} = \sum_{i=1}^{D}\frac{y_{i}^{2}}{\lambda_{i}}
\\[10pt]
y_{i} = u_{i}^{T}(\mathbf{x-\mu})
$$

我们将向量 $y_{i}$ 组合为一个矩阵 $\mathbf{y}$

$$
\mathbf{y} = (y_{1},\dots,y_{D})^{T} = \mathbf{U(x-\mu)}
$$

如果所有特征值 $\lambda_{i}$ 都是正的，那么这些表面代表椭球体，它们的中心在 $\mu$ 上，它们的轴沿 $u_{i}$ 方向，并且比例因子在 $\frac{\lambda_{i}}{2}$ 给出的轴方向上。

<center>
<img src="../imgs/Gaussian_Distribution.png" width="400">
</center>

原本我们是在 $x$ 中讨论高斯分布，现在我们把坐标轴换成 $y$ ，这样我们可以得到一个雅可比矩阵(Jacobian Matrix)

$$
J = \frac{\partial x_{i}}{\partial y_{i}} = U_{ji}
$$

其中 $U_{ji}$ 是矩阵 $\mathbf{U^{T}}$ 中的元素。利用矩阵 $\mathbf{U}$ 的正交性，我们可以得到雅可比矩阵 $\mathbf{J}$ 的行列式的平方。

$$
|\mathbf{J}|^{2} = |\mathbf{U^{T}}|^{2} = |\mathbf{U^{T}}||\mathbf{U}| = |\mathbf{I}| = 1
$$

因此协方差矩阵的行列式为1，同时我们可以将 $\Sigma$ 的行列式表示为其特征值的乘积，即：

$$
\begin{aligned}
|\Sigma| &=  |\mathbf{U\Lambda U^{T}}|
\\
&= \mathbf{|U||\Lambda||U^{T}|}
\\
&= \mathbf{|\Lambda|}
\\
&= \prod_{j=1}^{D}\lambda_{j}
\end{aligned}
$$

因此在 $y$ 坐标轴中，高斯分布变为以下形式：

$$
p(\mathbf{y}) = p(\mathbf{x})|\mathbf{J}| = \prod_{j=1}^{D}\frac{1}{(2\pi \lambda)^{\frac{1}{2}}} \exp\left\{ -\frac{y_{j}^{2}}{2\lambda_{j}} \right\}
$$

这实际上是 $D$ 个相互独立的单变量的高斯分布的乘积

### 条件高斯分布

假设我们有一个联合高斯分布 $\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma})$，其中 $\mathbf{x}$ 可以被分成两个部分 $\mathbf{x}_a$ 和 $\mathbf{x}_b$，对应的均值和协方差矩阵也可以相应地分成：

$$
\boldsymbol{\mu} = \begin{bmatrix} \boldsymbol{\mu}a \ \boldsymbol{\mu}b \end{bmatrix} \quad 
\\[10pt]
\quad \boldsymbol{\Sigma} = \begin{bmatrix} \boldsymbol{\Sigma}{aa} & \boldsymbol{\Sigma}{ab} \\ \boldsymbol{\Sigma}{ba} & \boldsymbol{\Sigma}{bb} \end{bmatrix}
$$
其中 $\boldsymbol{\mu}_a$ 和 $\boldsymbol{\mu}_b$ 分别是 $\mathbf{x}a$ 和 $\mathbf{x}b$ 的均值向量，$\boldsymbol{\Sigma}{aa}$ 和 $\boldsymbol{\Sigma}{bb}$ 分别是 $\mathbf{x}_a$ 和 $\mathbf{x}_b$ 的协方差矩阵，$\boldsymbol{\Sigma}{ab}$ 和 $\boldsymbol{\Sigma}{ba}$ 是 $\mathbf{x}_a$ 和 $\mathbf{x}_b$ 之间的协方差矩阵。

我们感兴趣的是在给定 $\mathbf{x}_b$ 的情况下，$\mathbf{x}_a$ 的条件分布。条件高斯分布的均值和协方差矩阵可以通过以下公式计算得到：

$$
\boldsymbol{\mu}{a|b} = \boldsymbol{\mu}a + \boldsymbol{\Sigma}{ab} \boldsymbol{\Sigma}{bb}^{-1} (\mathbf{x}_b - \boldsymbol{\mu}_b)
$$

$$
\boldsymbol{\Sigma}{a|b} = \boldsymbol{\Sigma}{aa} - \boldsymbol{\Sigma}{ab} \boldsymbol{\Sigma}{bb}^{-1} \boldsymbol{\Sigma}_{ba}
$$

其中，$\boldsymbol{\mu}{a|b}$ 是条件均值，$\boldsymbol{\Sigma}{a|b}$ 是条件协方差矩阵。

因此，条件高斯分布 $\mathbf{x}a | \mathbf{x}b$ 仍然是一个高斯分布，其均值和协方差矩阵分别为 $\boldsymbol{\mu}{a|b}$ 和 $\boldsymbol{\Sigma}{a|b}$。

总结来说，条件高斯分布的关键在于利用联合分布的均值和协方差矩阵，通过矩阵运算得到条件分布的均值和协方差矩阵。这种方法在多元统计分析和机器学习中非常有用，特别是在贝叶斯推理和高斯过程等领域。

### 边际高斯分布

### 高斯变量的贝叶斯定理

### 高斯函数的最大似然

### 序贯估计

### 高斯函数的贝叶斯推理

### 周期变量

### 高斯函数的混合函数

## 指数家族

### 最大似然和充分数据

### 共轭先验

### 非信息性先验

## 非参数方法

### 核密度估计器

### 最近邻方法
