# Lec07 - Normalizing Flow Models

## 对于复杂数据分布的简单先验

对于任何模型分布都有以下两点需求：

1. 可分析的概率密度
2. 容易采样

但是显然概率的真实分布相当复杂，因此衍生出**流模型**，即通过**可逆变换**把简单分布转换成复杂分布。

## 流模型与VAE

流模型和VAE实际上非常相似：

1. 他们从一个简单的隐空间中的先验分布开始：$z\sim\mathcal{N}(0,I)=p(z)$
2. 通过条件概率转化：$p(x|z)=\mathcal{N}(\mu_{\theta}(z),\Sigma_{\theta}(z))$
3. 即使先验分布$p(z)$很简单，但是边际分布$p(x)$很难计算，因为我们需要遍历所有的$z$并且积分
4. 此处开始产生分歧！！！：
   
   - VAE选择插入一个$p(z|x)$并且构造一个利用神经网络逼近的编码器来避免计算边际分布。**注意：**选择该思路的一个优势是$z$和$x$的维度可以不同，$z$的维度可以远低于$x$的维度。
   
   - 流模型选择转化过程变成一个可逆过程，即我们既可以通过$x=f_{\theta}(z)$在隐空间采样后得到数据分布，也可以通过$z=f_{\theta}^{-1}(x)$来进行逆变换，我们学习这个逆变换之后转换到正向变换从而生成数据。**注意：**选择该思路需要隐变量和数据变量的维度相同。

## 变量替换公式

**一维情况**：假设$X=f(Z)$，并且$f(\cdot)$是一个**可逆的单调函数**，即$Z=f^{-1}(X)=h(X)$，则有以下公式：
$$
p_{X}(x)=p_{Z}(h(x))\cdot\left| h'(x) \right|
$$
举例说明：假设$Z\sim\mathcal{U}[0,2],\;X=4Z$，则$Z=h(X)=\frac{X}{4},\;|h'(x)|=\frac{1}{4}$，那么我们可以得到$p_{X}(4)=p_{Z}(1)\cdot\frac{1}{4}=\frac{1}{8}$

**扩展到高维度线性关系**：假设$Z\in[0,1]^{n}$是一个$n$维随机向量，并且有$X=AZ$，其中矩阵$A$是一个$n\times n$的方阵，其逆矩阵为$W=A^{-1}$，则
$$
p_{X}(x)=p_{Z}(Wx)\cdot |\det{W}|=\frac{p_{Z}(Wx)}{|\det{A}|}
$$
实际上分母可以被视为归一化常数，确保$\int{p_{X}(x)dx=1}$

**扩展到高维度非线性关系**：假设$Z\in[0,1]^{n}$是一个$n$维随机向量，并且有$X=f(Z)$，$f(\cdot):R^{n}\mapsto R^{n}$是一个非线性可逆函数，即同样有$Z=f^{-1}(X)$，则我们可以得到$f(\cdot)$的雅可比矩阵的行列式：
$$
p_{X}(x)=p_{Z}\left(f^{-1}(x)\right)\left|\det{\frac{\partial{f^{-1}(x)}}{\partial{x}}}\right|=f_{Z}(z)\left|\det{\frac{\partial{f(z)}}{\partial{z}}}\right|^{-1}
$$
这实际上和一维情况下是等价的

## 归一化流模型

在**归一化流模型**中，$Z$和$X$之间的映射由$f_{\theta}:R^{n}\mapsto R^{n}$给出，其是确定的（即没有随机性）和可逆的，即$X=f_{\theta}(Z),\;Z=f_{\theta}^{-1}(X)$

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250412232826497.png" alt="image-20250412232826497" style="zoom:50%;" />

利用变量替换公式，我们可以得到$X$的边际分布：
$$
p_{X}(x;\theta)=p_{Z}(f_{\theta}^{-1}(x))\left| \det{\left( \frac{\partial{f_{\theta}^{-1}(x)}}{\partial{x}} \right)} \right|
$$
**再次强调：在归一化流模型中，$X$和$Z$的维度是相同的！**

通过变量代换，我们可以得到逆变换后的归一化概率密度。显然，只有一层神经网络层是不可能拟合的，因此需要嵌套多层神经网络
$$
z_{m}=f_{\theta}^{m}\circ\cdots\circ f_{\theta}^{1}(z_0)=f_{\theta}^{m}(f_{\theta}^{m-1}(\cdots(f_{\theta}^{1}(z_{0}))))\triangleq f_{\theta}(z_{0})
$$

- 首先从简单的先验分布中采样得到$z_{0}$
- 利用$m$层可逆变换最终得到$x=z_{M}$，由于每一层的变换都是可逆的，因此整个架构都是可逆的
- 通过变量代换可得：

$$
p_{X}(x;\theta)=p_{Z}(f_{\theta}^{-1}(x))\cdot\prod_{m=1}^{M}{\left| \det{\left( \frac{\partial({f_{\theta}^{m})^{-1}(z_m)}}{\partial{z_m}} \right)} \right|}
$$

下图是一个以高斯分布为先验分布的流模型的变换过程：

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250413004118272.png" alt="image-20250413004118272" style="zoom:25%;" />

## 训练与推理

通过最大化数据集$\mathcal{D}$对数似然函数进行模型训练，即
$$
\max_{\theta}{\log{p_{X}(\mathcal{D};\theta)}}=\sum_{x\in\mathcal{D}}{\log{p_{Z}(f_{\theta}^{-1}(x))}}+\log{\left| \det{\left( \frac{\partial{f_{\theta}^{-1}(x)}}{\partial{x}} \right)} \right|}
$$

- 与变分自编码器不同，我们可以通过逆变换步骤$x\mapsto z$和变量代换公式得到精确的似然估计。

- 在采样过程，我们通过正向变换步骤$z\mapsto x$来得到模型生成的数据，即

$$
z\sim p_{Z}(z),\;\;x=f_{\theta}(z)
$$

- $z$通过逆变换推断出的潜在表示（无需推理网络！）

**难点**：计算一个$n\times n$的雅可比矩阵的行列式一般需要$O(n^{3})$的计算量，在$n$比较大的时候很难计算

**解决方案**：选择一个合适的变换，其雅可比矩阵有特殊结构（例如三角方阵，其行列式为对角线元素之积，即$O(n)$的复杂度）

## 三角雅可比矩阵

$$
\mathbf{x}=(x_1,\cdots,x_n)=\mathbf{f}(\mathbf{z})=(f_1(\mathbf{z}),\cdots,f_n(\mathbf{z})),\;\mathbf{z}=(z_1,\cdots,z_n)
\\[10pt]
J=\frac{\partial{\mathbf{f}}}{\partial{\mathbf{z}}}=
\begin{pmatrix}  
\frac{\partial{f_1}}{\partial{z_1}}&\cdots&\frac{\partial{f_1}}{\partial{z_n}} \\
 \vdots&\vdots&\vdots \\
 \frac{\partial{f_n}}{\partial{z_1}}&\cdots&\frac{\partial{f_n}}{\partial{z_n}}
\end{pmatrix}
$$

假设$x_i=f_i(\mathbf{z})$只和$\mathbf{z}$的前$i$个参数有关，即$\mathbf{z}_{\le i}$，则
$$
J=\frac{\partial{\mathbf{f}}}{\partial{\mathbf{z}}}=
\begin{pmatrix}  
\frac{\partial{f_1}}{\partial{z_1}}&\cdots&0 \\
 \vdots&\vdots&\vdots \\
 \frac{\partial{f_n}}{\partial{z_1}}&\cdots&\frac{\partial{f_n}}{\partial{z_n}}
\end{pmatrix}
$$
就是一个标准的下三角矩阵。这样其行列式就可以以$O(n)$的复杂度计算。类似的，如果$x_i$只和$\mathbf{z}_{\ge i}$有关，则雅可比矩阵就是一个标准的上三角矩阵。

## 设计可逆变换

### NICE (Nonlinear Independent Components Estimation)

**NICE**把两个可逆的变换组合在一起：**加性耦合层**和**重缩放层**

**加性耦合层**：$\forall\;d\in[1,n),\:d\in\mathbf{N}$，我们都把变量$\mathbf{z}$分成两个不相交的部分$\mathbf{z}_{1:d}$和$\mathbf{z}_{d+1:n}$

- 前向匹配：$\mathbf{z\mapsto x}$
  - $\mathbf{x}_{1:d}=\mathbf{z}_{1:d}$（恒等变换）
  - $\mathbf{x}_{d+1:n}=\mathbf{z}_{d+1:n}+m_{\theta}(\mathbf{z}_{1:d})$，其中$m_{\theta}(\cdot)$是一个神经网络层，有$d$个输入单元，对应$\mathbf{z}_{1:d}$，以及$n-d$个输出单元，对应$\mathbf{z}_{d+1:n}$

- 反向匹配：$\mathbf{x\mapsto z}$
  - $\mathbf{z}_{1:d}=\mathbf{x}_{1:d}$（恒等变换）
  - $\mathbf{z}_{d+1:n}=\mathbf{x}_{d+1:n}+m_{\theta}(\mathbf{x}_{1:d})$，

- 前向传播中的雅可比矩阵：

$$
J=\frac{\partial{\mathbf{x}}}{\partial{\mathbf{z}}}=\begin{pmatrix}  
I_d&0\\
\frac{\partial{\mathbf{x}_{d+1:n}}}{\partial{\mathbf{z}_{1:d}}} & I_{n-d}
\end{pmatrix}
\\[10pt]
\det{J}=1
$$

在每一层中，加性耦合层都组合在一起（划分方式任意），而NICE的最后一层利用了**重放缩**：

- 前向匹配：$\mathbf{z\mapsto x}$
  $$
  x_{i}=s_iz_i
  $$
  其中$s_i\gt0$是第$i$个维度上的缩放因子

- 反向匹配：$\mathbf{x\mapsto z}$

$$
z_i=\frac{x_i}{s_i}
$$

- 前向传播中的雅可比矩阵：
  $$
  J=\text{diag}(\mathbf{s})
  \\[10pt]
  \det{J}j=\prod_{i=1}^{n}{s_i}
  $$

### Real-NVP

- 前向匹配：$\mathbf{z\mapsto x}$
  - $\mathbf{x}_{1:d}=\mathbf{z}_{1:d}$（恒等变换）
  - $\mathbf{x}_{d+1:n}=\mathbf{z}_{d+1:n}\odot\exp{(\alpha_{\theta}(\mathbf{z}_{1:d}))}+\mu_{\theta}(\mathbf{z}_{1:d})$，其中$\alpha_{\theta}(\cdot)$和$\mu_{\theta}(\cdot)$是以$\theta$为参数的神经网络层，有$d$个输入单元，对应$\mathbf{z}_{1:d}$，以及$n-d$个输出单元，对应$\mathbf{z}_{d+1:n}$（$\odot$是按元素的积，即每个维度依次相乘）
- 反向匹配：$\mathbf{x\mapsto z}$
  - $\mathbf{z}_{1:d}=\mathbf{x}_{1:d}$（恒等变换）
  - $\mathbf{z}_{d+1:n}=\left(\mathbf{x}_{d+1:n}-\mu_{\theta}(\mathbf{x}_{1:d})\right)\odot(\exp{(-\alpha_{\theta}(\mathbf{x}_{1:d}))})$
- 前向传播中的雅可比矩阵：

$$
J=\frac{\partial{\mathbf{x}}}{\partial{\mathbf{z}}}=\begin{pmatrix}  
I_d&0\\
\frac{\partial{\mathbf{x}_{d+1:n}}}{\partial{\mathbf{z}_{1:d}}} & diag(\exp{(\alpha_{\theta}(\mathbf{z}_{1:d})})
\end{pmatrix}
\\[10pt]
\det{J}=\prod_{i=d+1}^{n}{\exp{(\alpha_{\theta}(\mathbf{z}_{1:d})_{i})}}=\exp\left(\sum_{i=d+1}^{n}{\alpha_{\theta}(\mathbf{z}_{1:d})_{i}}\right)
$$

此处的雅可比矩阵的行列式并不一定为1，因此可能概率分布的“体积”会有所变换，也因此可以有更丰富的表示。

## 将连续自回归模型视为流模型

考虑一个高斯自回归模型：
$$
p(\mathbf{x})=\prod_{i=1}^{n}p(x_i|\mathbf{x}_{\lt i})
\\[10pt]
p(x_i|\mathbf{x}_{\lt i})=\mathcal{N}(\mu_i(x_1,\cdots,x_{i-1}),\exp{(\alpha_i(x_1,\cdots,x_{i-1}))^{2}})
$$
其中$\mu_i$和$\alpha_i$在$i\gt1$的情况下是神经网络层，在$i=1$的情况下是常数。

从模型中进行采样：

- 从标准高斯分布中采样$u_{i}\sim\mathcal{N}(0,1),\;i=1,\cdots,n$
- 令$x_1=\exp{(\alpha_{1})}\cdot u_1+\mu_1$（重参数化技巧），计算$\mu_2(x_1),\alpha_2(x_1)$
- 令$x_2=\exp{(\alpha_{2})}\cdot u_2+\mu_2$（重参数化技巧），计算$\mu_3(x_1,x_2),\alpha_3(x_1,x_2)$
- 如此反复，我们会发现**流模型的另一种解释**：把从标准高斯分布中采样得到的样本$(u_1,u_2,\cdots,u_n)$通过可逆的转换（由$\mu_i(\cdot),\alpha_i(\cdot)$参数化得到），得到模型的输出$(x_1,x_2,\cdots,x_n)$

### 掩码自回归流 (Masked Autoregressive Flow)

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250414000506540.png" alt="image-20250414000506540" style="zoom:30%;" />

上述过程实际上就是掩码自回归流的正向匹配过程$\mathbf{u\mapsto x}$，既然是流模型，肯定也要有一个**反向匹配的过程**：

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250414000759970.png" alt="image-20250414000759970" style="zoom:30%;" />

反向匹配：$\mathbf{x\mapsto u}$

- 计算所有的$\mu_i,\alpha_i$（因为你有了所有的数据，这些可以直接算出来）
- 令$u_1=\frac{x_1-\mu_1}{\exp{(\alpha_1)}}$（缩放+平移）

- 令$u_2=\frac{x_2-\mu_2}{\exp{(\alpha_2)}}$
- 如此反复，我们会发现雅可比矩阵仍然是下三角形矩阵，因此行列式计算十分方便。同时，似然估计容易计算并且可以并行计算

### 逆自回归流 (Inverse Autoregressive Flow)

自回归模型的一个缺点是：采样速度很慢！怎么解决这个问题？由于这是一个可逆的架构，因此我们也可以选择把$\mathbf{x}$看成$\mathbf{u}$，把$\mathbf{u}$看成$\mathbf{x}$，这样子我们就可以并行计算$\mathbf{u\mapsto x}$的所有参数，采样速度大大加快。

<img src="/Users/hongshuo/Library/Application Support/typora-user-images/image-20250416004559186.png" alt="image-20250416004559186" style="zoom:30%;" />

前向匹配$\mathbf{u\mapsto x}$（并行计算）：

- 从标准高斯分布中采样$u_{i}\sim\mathcal{N}(0,1),\;i=1,\cdots,n$
- 计算所有的$\mu_i,\alpha_i$（并行计算）
- 令$x_1=\exp{(\alpha_{1})}\cdot u_1+\mu_1$
- 令$x_2=\exp{(\alpha_{2})}\cdot u_2+\mu_2$......

反向匹配$\mathbf{x\mapsto u}$（序列计算）：

- 令$u_1=\frac{x_1-\mu_1}{\exp{(\alpha_1)}}$，计算$\mu_2(u_1),\alpha_2(u_1)$
- 令$u_2=\frac{x_2-\mu_2}{\exp{(\alpha_2)}}$，计算$\mu_3(u_1,u_2),\alpha_3(u_1,u_2)$

其采样过程很快，但是似然估计过程很慢，也就是训练过程慢。

### MAF vs. IAF

- MAF：似然评估快，采样慢。更适合基于最大似然估计的训练和密度估计。
- IAF：采样快，似然评估慢。更适合实时生成。

问题：**能否兼两者之长？** 有的兄弟！有的！像这样的模型当然是不止一个了！

### 平行波网 (Parallel Wavenet)

包含**教师模型**和**学生模型**的两阶段训练。教师模型通过MAF进行参数化。教师模型可以通过最大似然估计（MLE）高效训练。一旦教师模型训练完成，初始化一个通过 IAF 进行参数化的学生模型。学生模型无法高效地为外部数据点评估密度，但允许高效采样。

**概率密度蒸馏**：“学生模型”的分布的损失函数是“学生模型”的分布和“老师模型”的分布之间的差异，一般用KL散度来衡量，即：
$$
D_{KL}(s,t)=E_{\mathbf{x}\sim s}\left[ \log{s(\mathbf{x})}-\log{t(\mathbf{x})} \right]
$$
评估并优化此目标的蒙特卡罗估计值需要：

- 来自学生模型（IAF）的样本$\mathbf{x}$
- 学生模型分配给$\mathbf{x}$的密度
- 教师模型（MAF）分配给$\mathbf{x}$的密度
