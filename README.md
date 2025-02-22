# Diffusion-Model-Learning

 这个仓库记录了我开始学习扩散模型的学习过程，包括论文资源、学习笔记、代码复现。主要集中于图像生成的扩散模型。

## 学习顺序建议：

### **第一阶段：基础知识准备**

1. **数学基础**
   * **概率论与统计学** ：扩散模型的核心是概率分布和随机过程。重点学习：
   * 马尔可夫链
   * 高斯分布
   * 随机微分方程（SDE）
   * **优化理论** ：理解梯度下降、反向传播等优化方法。
     **推荐资源** ：
   * 《概率论与数理统计》（国内教材）
   * [Khan Academy: Probability and Statistics汗学院：概率与统计](https://www.khanacademy.org/math/statistics-probability)
2. **深度学习基础**
   * **PyTorch** ：确保你对PyTorch的Tensor操作、自动求导、模型定义和训练流程非常熟悉。
   * **生成模型** ：了解生成对抗网络（GANs）、变分自编码器（VAEs）等生成模型的基本原理。
     **推荐资源** ：
   * [PyTorch官方教程](https://pytorch.org/tutorials/)
   * [Deep Learning with PyTorch: A 60 Minute Blitz深度学习与 PyTorch：60 分钟速成](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

---

### **第二阶段：扩散模型入门**

1. **扩散模型的基本概念**
   * 了解扩散模型的核心思想：通过逐步添加噪声将数据分布转化为高斯分布，再通过逆向过程生成数据。
   * 学习关键概念：前向过程（Forward Process）、逆向过程（Reverse Process）、噪声预测网络（Noise Prediction Network）。
     **推荐资源** ：
   * [Denoising Diffusion Probabilistic Models (DDPM) 原始论文去噪扩散概率模型（DDPM）原始论文](https://arxiv.org/abs/2006.11239)
   * [Lilian Weng的博客：What are Diffusion Models?李莉安·王博客：什么是扩散模型？](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
2. **代码实践**
   * 从简单的扩散模型实现开始，理解代码结构。
     **推荐资源** ：
   * [PyTorch实现DDPM  PyTorch 实现 DDPM](https://github.com/hojonathanho/diffusion)
   * [Denoising Diffusion Implicit Models (DDIM) 代码去噪扩散隐式模型（DDIM）代码](https://github.com/ermongroup/ddim)

---

### **第三阶段：深入学习与扩展**

1. **高级扩散模型**
   * 学习改进的扩散模型，如DDIM、Score-Based Generative Models、Latent Diffusion Models (LDM)。学习改进的扩散模型，如 DDIM、基于随机微分方程的得分生成模型、潜在扩散模型（LDM）。
     **推荐资源** ：
   * [DDIM论文](https://arxiv.org/abs/2010.02502)
   * [Score-Based Generative Modeling through Stochastic Differential Equations基于随机微分方程的得分生成建模](https://arxiv.org/abs/2011.13456)
   * [Latent Diffusion Models  潜在扩散模型](https://arxiv.org/abs/2112.10752)
2. **应用领域**
   * 了解扩散模型在图像生成、文本生成、音频生成等领域的应用。
   * **推荐资源** ：
   * [Stable Diffusion: 文本到图像生成](https://github.com/CompVis/stable-diffusion)
   * [Audio Diffusion: 音频生成  音频扩散：音频生成](https://github.com/archinetai/audio-diffusion-pytorch)

---

### **第四阶段：项目实践与优化**

1. **动手实践**
   * 选择一个感兴趣的应用领域（如图像生成），实现一个完整的扩散模型项目。
     **推荐项目** ：
   * 使用扩散模型生成MNIST或CIFAR-10数据集中的图像。
   * 复现Stable Diffusion的简化版本。
2. **性能优化**
   * 学习如何加速扩散模型的推理过程（如DDIM的加速采样）。
   * 探索如何提高生成质量（如改进噪声预测网络）。

---

### **第五阶段：前沿研究与社区参与**

1. **跟踪最新研究**
   * 关注扩散模型的最新论文和进展。
     **推荐资源** ：
   * [arXiv: Diffusion Models](https://arxiv.org/search/?query=diffusion+models&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
   * [Papers with Code: Diffusion Models](https://paperswithcode.com/task/diffusion-models)
2. **参与社区**
   * 加入相关社区，与同行交流。
     **推荐社区** ：
   * [Reddit: Machine Learning](https://www.reddit.com/r/MachineLearning/)
   * [GitHub: 相关开源项目](https://github.com/topics/diffusion-models)

---

### **总结**

1. **学习路线概览** ：

* 基础知识 → 扩散模型入门 → 深入学习 → 项目实践 → 前沿研究。

1. **关键资源** ：

* 论文：DDPM、DDIM、Score-Based Models、Latent Diffusion Models。
* 代码：PyTorch实现、Stable Diffusion、Audio Diffusion。
* 社区：arXiv、Papers with Code、GitHub。

通过以上路线，你可以逐步掌握扩散模型的理论和实践，最终能够独立实现和应用扩散模型。祝你学习顺利！

## 使用方法

1. **克隆仓库**

```bash
git clone https://github.com/Hasson827/UCIML-Example.git
cd UCIML-Example
```

2. **数据集下载说明**

本项目使用了以下数据集：
- [MNIST](http://yann.lecun.com/exdb/mnist/)

3. 下载数据集
-  确保你已安装 `wget` 和 `unzip`。在 Linux/macOS 中可以使用以下命令安装：
   ```bash
   sudo apt update && sudo apt install wget unzip   # Ubuntu/Debian
   brew install wget unzip                          # macOS (使用 Homebrew)

- 运行以下命令来安装数据集

```bash
bash Book_Notes/download-datasets.sh
```

## 贡献

欢迎贡献！如果你有新的数据集代码示范或改进建议，请提交 Pull Request。

1. Fork 这个仓库。
2. 创建你的分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)。
4. 推送到分支 (`git push origin feature/AmazingFeature`)。
5. 打开一个 Pull Request。

## 许可证

这个项目采用 MIT 许可证。详情请参阅 [LICENSE](https://chat.deepseek.com/a/chat/s/LICENSE) 文件。

## 联系方式

如果你有任何问题或建议，请通过以下方式联系我：

* 邮箱: hongshuo.24@intl.zju.edu.cn
* GitHub: Hasson827
