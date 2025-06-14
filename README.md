# Generative-Model-Learning

一个全面的生成模型学习资源库，包含理论基础、前沿论文解读以及实践代码。本项目致力于构建从基础概念到最新研究的完整学习路径。

## 📚 项目概述

本仓库整理了生成模型相关的学习资料，涵盖：
- **理论基础**：概率论、贝叶斯网络、变分推断等数学基础
- **经典模型**：扩散模型(DDPM/DDIM)、流模型、变分自编码器等
- **前沿研究**：最新论文解读和技术分析
- **实践应用**：代码示例和实现细节

## 🗂️ 目录结构

### 📖 课程笔记

#### Stanford CS236 - Deep Generative Models

#### MIT 6.S184 - Computational Aspects of Statistical Learning

### 📄 论文笔记

#### 扩散模型核心论文

- **[扩散模型奠基与DDPM](Paper%20Notes/扩散模型奠基与DDPM/扩散模型奠基与DDPM.md)**: DDPM原理深度解析
  - 前向扩散过程详解
  - 反向去噪过程推导
  - ELBO损失函数推导
  - U-Net架构设计原理
  - 时间嵌入机制
  
- **[去噪扩散隐式模型(DDIM)](Paper%20Notes/去噪扩散隐式模型(DDIM)/Denoising%20Diffusion%20Implicit%20Models.md)**: DDIM加速采样技术
  - 非马尔可夫扩散过程
  - 确定性生成过程
  - 采样速度优化

### 💻 计算机视觉基础

## 🔧 核心概念速查

| 概念 | 描述 | 相关资料 |
|------|------|----------|
| **DDPM** | 去噪扩散概率模型，生成模型的重要突破 | [DDPM详解](Paper%20Notes/扩散模型奠基与DDPM/扩散模型奠基与DDPM.md) |
| **DDIM** | 确定性采样的扩散隐式模型 | [DDIM解析](Paper%20Notes/去噪扩散隐式模型(DDIM)/Denoising%20Diffusion%20Implicit%20Models.md) |
| **Flow Models** | 基于常微分方程的生成模型 | [Flow Models](MIT_6.S184/Lec02%20Flow%20and%20Diffusion%20Models.md#21-流模型) |
| **贝叶斯网络** | 图模型表示的概率分布 | [Background](Stanford_CS236/Lec02%20-%20Background.md#贝叶斯网络) |
| **U-Net** | 扩散模型中的核心网络架构 | [U-Net详解](Paper%20Notes/扩散模型奠基与DDPM/扩散模型奠基与DDPM.md#2-u-net的作用与噪声预测) |

## 🤝 贡献

欢迎贡献！如果你有新的数据集代码示范或改进建议，请提交 Pull Request。

1. Fork 这个仓库
2. 创建你的分支 (`git checkout -b feature/AmazingFeature`)
3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开一个 Pull Request

### 贡献指南

- 📝 **论文笔记**：欢迎添加新的论文解读
- 💻 **代码实现**：欢迎提供相关算法的代码实现
- 🐛 **错误修正**：发现错误请及时提交修正
- 📚 **内容完善**：欢迎补充和完善现有内容

## 📜 许可证

这个项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 📧 联系方式

如果你有任何问题或建议，请通过以下方式联系我：

* 📮 邮箱: hongshuo.24@intl.zju.edu.cn 或 hz108@illinois.edu
* 🐙 GitHub: [Hasson827](https://github.com/Hasson827)

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

*持续更新中...*