# Awesome Deepfake Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

一份精心整理的Deepfake（深度伪造）检测资源列表，涵盖最新论文、开源代码、常用数据集及方法基准。该项目致力于为研究人员和开发者提供一个快速入门和追踪前沿技术的平台。

*A curated list of resources for Deepfake Detection, including papers, code, datasets, and benchmarks. This repository aims to help researchers and developers get started and keep track of the cutting-edge techniques.*

---

## 目录 (Table of Contents)

*   [最新动态 (News)](#最新动态-news)
*   [论文列表 (Papers)](#论文列表-papers)
    *   [范式一：伪造痕迹追寻范式 (Forgery Trace Seeking Paradigm)](#范式一伪造痕迹追寻范式-forgery-trace-seeking-paradigm)
    *   [范式二：真实分布学习范式 (Authentic Distribution Learning Paradigm)](#范式二真实分布学习范式-authentic-distribution-learning-paradigm)
    *   [纯粹的综述、基准与分析研究](#纯粹的综述基准与分析研究)
*   [开源代码 (Code)](#开源代码-code)
*   [数据集 (Datasets)](#数据集-datasets)
*   [方法对比 (Benchmark)](#方法对比-benchmark)
*   [贡献指南 (Contribution)](#贡献指南-contribution)
*   [许可证 (License)](#许可证-license)

---

## 最新动态 (News)

*   **2024-XX-XX:** 项目初始化，并根据两大核心检测范式完成130余篇文献的分类整理。
*   <!-- 在这里添加最新的项目更新 -->

---

## 论文列表 (Papers)

所有检测技术相关的论文根据其核心逻辑，被归类于**“证伪” (Forgery Trace Seeking)** 和 **“存真” (Authentic Distribution Learning)** 两大范式之下。

### **范式一：伪造痕迹追寻范式 (Forgery Trace Seeking Paradigm)**
**核心逻辑：“证伪”。** 主动寻找、增强、解耦和利用生成过程中遗留的各类痕迹作为证据。

#### **Deepfake人脸伪造检测**

##### **1. 频率域伪影 (Frequency-Domain Artifacts)**
*   **HFF (2021, CVPR):** *Luo et al.*
    *   **主要贡献:** 提出一个双流模型，证明利用多尺度高频噪声（SRM）并建模其与RGB特征的交互（DCMA），可以提升人脸伪造检测的泛化能力。
    *   **代码/项目链接:** 未提及 (作者个人实现: [crywang/face-forgery-detection](https://github.com/crywang/face-forgery-detection))
*   **SPSL (2021, CVPR):** *Honggu Liu et al.*
    *   **主要贡献:** 依赖“上采样是多数伪造技术的必要步骤，并在相位谱留下痕迹”的先验，首次利用相位谱并结合浅层网络来检测伪造。
    *   **代码/项目链接:** 未提及
*   **Watch Your Up-Convolution (2020, CVPR):** *Durall, R. 等*
    *   **主要贡献:** 首次系统性地揭示了基于上采样/转置卷积的GAN会产生明显的频谱失真，并利用该缺陷实现了高精度检测。
    *   **代码/项目链接:** [https://github.com/cc-hpc-itwm/UpConv](https://github.com/cc-hpc-itwm/UpConv)
*   **Wavelet-packets (2022, Machine Learning):** *Wolter, M. 等*
    *   **主要贡献:** 首次将多尺度小波包变换用于合成图像分析，揭示了GAN在时频域的均值和标准差异常。
    *   **代码/项目链接:** [https://github.com/v0lta/PyTorch-Wavelet-Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)

##### **2. 混合边界与合成伪影 (Blending Boundary & Synthesis Artifacts)**
*   **SLADD (2022, CVPR):** *Liang Chen et al.*
    *   **主要贡献:** 依赖“一个可泛化的表征应对多样化伪造类型敏感”的先验，通过对抗性增强伪造样本多样性，并用自监督任务提升模型敏感度。
    *   **代码/项目链接:** [https://github.com/liangchen527/SLADD](https://github.com/liangchen527/SLADD)
*   **SBI (2022, CVPR):** *Shiohara & Yamasaki*
    *   **主要贡献:** 提出一种数据合成方法SBI，通过单张图像自混合生成高质量伪造样本，提升模型对未知伪造的泛化能力。
    *   **代码/项目链接:** [https://github.com/mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages)
*   **LSDA (2024, CVPR):** *Zhiyuan Yan et al.*
    *   **主要贡献:** 依赖“扩大伪造空间能学习到更鲁棒的决策边界”的先验，在潜在空间中进行域内和域间增强，以扩展伪造特征的多样性。
    *   **代码/项目链接:** 未提及
*   **Face X-ray (2020, CVPR):** *Li, L. 等*
    *   **主要贡献:** 提出Face X-ray概念，将伪造检测问题转化为寻找图像中的“混合边界”，不依赖特定伪造痕迹，从而提高泛化能力。
    *   **代码/项目链接:** 未提及

##### **3. 解耦伪造特征 (Decoupling Forgery Features)**
*   **UCF (2023, ICCV):** *Zhiyuan Yan et al.*
    *   **主要贡献:** 依赖“伪造特征可分解为方法特定和方法通用的部分”的先验，设计多任务解耦框架，分离内容、特定伪造特征和通用伪造特征。
    *   **代码/项目链接:** 未提及
*   **IID (2023, CVPR):** *Huang, B. 等*
    *   **主要贡献:** 提出通过伪造人脸的“显式身份”（源人脸）和“隐式身份”（目标人脸）之间的不一致性来检测人脸交换。
    *   **代码/项目链接:** 未提及
*   **CADDM (2023, CVPR):** *Dong, S. 等*
    *   **主要贡献:** 发现并验证了Deepfake检测中的“隐式身份泄露”问题，并设计了伪影检测模块来强制模型关注局部伪影。
    *   **代码/项目链接:** [https://github.com/megvii-research/CADDM](https://github.com/megvii-research/CADDM)

##### **4. 内部一致性与物理实验 (Internal Consistency & Physical Priors)**
*   **Bi-LIG + TCC-ViT (2024, IEEE TIFS):** *Jiang, P. 等*
    *   **主要贡献:** 提出双层不一致性生成器(Bi-LIG)，使模型同时学习“外在不一致性”和“内在不一致性”。
    *   **代码/项目链接:** 未提及
*   **GrDT (2021, WACVW):** *Xie, H., 等*
    *   **主要贡献:** 结合面部关键点的几何分布（图注意力网络处理）和局部纹理特征（灰度共生矩阵GLCM提取）。
    *   **代码/项目链接:** [https://github.com/SIPLab24/GrDT](https://github.com/SIPLab24/GrDT)

#### **通用合成图像检测**

##### **1. 频率与统计指纹 (Frequency & Statistical Fingerprints)**
*   **Synthbuster (2024, IEEE OJSP):** *Bammey, Q.*
    *   **贡献:** 提出通过简单的交叉差分高通滤波器在单张图像上即可有效揭示并检测扩散模型的频域伪影。
    *   **代码/项目链接:** [数据集: https://zenodo.org/records/10066460](https://zenodo.org/records/10066460)
*   **UGAD (2024, CIKM):** *Alam, I. 等*
    *   **贡献:** 提出一种多模态频域检测方法，结合径向积分操作(RIO)和空间傅里叶提取(SFE)。
    *   **代码/项目链接:** 未提及
*   **MaskSim (2024, CVPRW):** *Li, Y. 等*
    *   **贡献:** 提出MaskSim，一种半白盒方法，通过学习频谱掩模和参考模式来识别特定生成器的频率指纹。
    *   **代码/项目链接:** [https://github.com/li-yanhao/masksim](https://github.com/li-yanhao/masksim)
*   **DCT-based Classifier (2024, arXiv):** *Pontorno, O. 等*
    *   **贡献:** 对GAN和DM图像的DCT系数进行统计分析，发现特定（尤其是低频）系数子集对检测更鲁棒。
    *   **代码/项目链接:** [github.com/opontorno/dcts_analysisdeepfakes](https://github.com/opontorno/dcts_analysisdeepfakes)
*   **BiHPF (2022, WACV):** *Jeong, Y. 等*
    *   **贡献:** 提出BiHPF，一种双边高通滤波器，通过放大高频和背景区域的伪影来提升鲁棒检测能力。
    *   **代码/项目链接:** 未提及
*   **Detecting and Simulating... (AutoGAN) (2019, WIFS):** *Zhang, X. 等*
    *   **贡献:** 发现上采样在频域中产生频谱复制的伪影，并提出AutoGAN模拟器，使训练不再依赖目标GAN。
    *   **代码/项目链接:** [https://github.com/ColumbiaDVMM/AutoGAN](https://github.com/ColumbiaDVMM/AutoGAN)
*   **FourierSpectrum (2020, NeurIPS):** *Dzanic, T. 等*
    *   **贡献:** 首次发现真实与生成图像在高频傅立葉谱衰减率上存在系统性差异。
    *   **代码/项目链接:** 未提及
*   **Benford’s Law (2020, ArXiv):** *Bonettini, N. 等*
    *   **贡献:** 证明了GAN生成的图像不符合本福特定律，并基于DCT系数的偏差设计检测器。
    *   **代码/项目链接:** [https://github.com/polimi-ispl/icpr-benford-gan](https://github.com/polimi-ispl/icpr-benford-gan)
*   **Discovering Transferable Forensic Features... (2022, arXiv):** *Chandrasegaran, K. 等*
    *   **贡献:** 提出FF-RS方法来发现和量化可迁移法证特征（T-FF），并揭示了**颜色**是一个被忽视但至关重要的T-FF。
    *   **代码/项目链接:** 未提及

##### **2. 结构与梯度伪影 (Structural & Gradient Artifacts)**
*   **NPR (2024, CVPR):** *Tan, C. 等*
    *   **贡献:** 提出NPR（邻近像素关系）作为一种简单、通用的伪影表示，捕获上采样操作引起的局部结构伪影。
    *   **代码/项目链接:** [https://github.com/chuangchuangtan/NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
*   **SFLD (2025, arXiv):** *Gye et al.*
    *   **贡献:** 提出SFLD，通过多尺度PatchShuffle融合高层语义与低层纹理，减少内容偏见。
    *   **代码/项目链接:** [数据集: https://huggingface.co/datasets/koooooooook/TwinSynths](https://huggingface.co/datasets/koooooooook/TwinSynths)
*   **Rich/Poor Texture Contrast (2023, arXiv):** *Zhong, N. 等*
    *   **贡献:** 提出基于图像贫富纹理区域间像素相关性对比的通用伪造指纹。
    *   **代码/项目链接:** [https://fdmas.github.io/AIGCDetect/](https://fdmas.github.io/AIGCDetect/)
*   **LGrad (2023, CVPR):** *Tan, C. 等*
    *   **贡献:** 提出LGrad框架，利用预训练CNN模型提取的梯度作为广义伪影表示，提升泛化性。
    *   **代码/项目链接:** [https://github.com/chuangchuangtan/LGrad](https://github.com/chuangchuangtan/LGrad)
*   **Detecting GAN generated... (Co-occurrence) (2019, arXiv):** *Nataraj, L. 等*
    *   **贡献:** 提出一种基于像素共生矩阵和CNN的GAN图像检测方法。
    *   **代码/项目链接:** 未提及
*   **Detection, Attribution and Localization... (2020, arXiv):** *Goebel, M. 等*
    *   **贡献:** 提出一种基于多方向共生矩阵和XceptionNet的检测、归因、定位框架。
    *   **代码/项目链接:** 未提及
*   **DIO (2024, arXiv):** *Tan, C. 等*
    *   **贡献:** 提出数据无关算子(DIO)框架，使用固定的、无需训练的滤波器作为伪影提取器。
    *   **代码/项目链接:** [https://github.com/chuangchuangtan/Data-Independent-Operator](https://github.com/chuangchuangtan/Data-Independent-Operator)
*   **Critical Analysis... (no-down variant) (2021, ICME):** *Gragnaniello, D. 等*
    *   **贡献:** 证明取消CNN模型早期的下采样操作能更好地保留高频伪影，是提升GAN图像检测器泛化能力的关键。
    *   **代码/项目链接:** [https://github.com/grip-unina/GANimageDetection](https://github.com/grip-unina/GANimageDetection)

#### **服务于“证伪”的支撑技术**

##### **1. 增强伪造痕迹的训练策略 (Training Strategies for Enhancing Forgery Traces)**
*   **Adv. Augment (2024, MAD):** 通过循环对抗攻击生成更难的伪造痕迹样本，以增强模型。
*   **AFSL (2024, MAD):** 通过对抗特征相似性学习，强制模型关注更鲁棒的伪造痕迹。
*   **Frequency Masking (2025, ICASSP):** 在训练中对输入图像进行频率掩码，作为一种数据增强方法来强制模型关注非掩码区域的痕迹。
*   **VPE (2024, IEEE Access):** 提出视觉提示工程(VPE)作为预处理模块，增强系统对GAN规避攻击的鲁棒性。
*   **APN (2024, arXiv):** 提出一个伪影纯化网络(APN)，通过解耦和提纯伪影特征，让模型更专注于学习核心伪造痕迹。
*   **DF-UDetector (2025, Neural Networks):** 在特征空间进行“伪影恢复”以增强退化场景（痕迹被破坏）下的鲁棒性。

##### **2. 旨在融合多种伪造痕迹的架构 (Architectures for Fusing Forgery Traces)**
*   **A Hybrid Model for Generalizable... (2025, SCID):** 提出一个结合伪造边界、语义和通用伪影的三合一混合检测模型。
*   **LightweightViT (2025, IEEE Access):** 提出一种轻量级ViT模型，利用patch embedding和自注意力机制高效检测图像篡改。
*   **Fine-grained deepfake detection... (2023, NCA):** 提出了一个基于跨模态注意力的细粒度检测网络，融合RGB、频率和纹理特征。
*   **FreqCross (2025, arXiv):** 提出FreqCross三分支网络，融合空间、频谱和径向能量分布特征。
*   **Fusing Global and Local Features (2022, arXiv):** 提出一个双分支网络，通过注意力机制融合全局特征和关键局部补丁特征。

##### **3. 针对伪造痕迹的对抗攻防 (Adversarial Attacks & Defenses on Forgery Traces)**
*   **TAG-WM (2025, arXiv):** 提出一种篡改感知的生成式水印方法，本质上是主动嵌入一种可追溯的“痕迹”。 ([代码](https://github.com/Suchenl/TAG-WM))
*   **Adversarial Robustness in DeepFake Detection (2024, ICICyTA):** 展示了检测模型在对抗攻击下的脆弱性，并验证了防御策略的有效性。
*   **Counter-Forensic (2024, ICVGIP):** 将DM重构真实图像定义为反取证攻击（抹除痕迹），并构建了识别框架。
*   **SpectralGAN (Attack) (2022, CVPR):** 证明了基于频谱伪影的检测器并非鲁棒，其依赖的伪影可被移除。 ([代码](https://www.comp.polyu.edu.hk/~csajaykr/deepdeepfake.htm))

---

### **范式二：真实分布学习范式 (Authentic Distribution Learning Paradigm)**
**核心逻辑：“存真”。** 致力于学习真实图像的本质分布，将任何偏离该分布的样本视为异常。

#### **Deepfake人脸伪造检测**

##### **1. 单类分类与边界学习 (One-Class Classification & Boundary Learning)**
*   **Stay-Positive (2025, ICML):** *Anirudh Sundara Rajan et al.*
    *   **主要贡献:** 通过约束分类器最后一层权重为非负，强制模型只关注伪造特征（即偏离真实分布的部分），忽略真实图像特征。
    *   **代码/项目链接:** [https://github.com/AniSundar18/AlignedForensics](https://github.com/AniSundar18/AlignedForensics)
*   **SeeABLE (2023, ICCV):** *Larue et al.*
    *   **主要贡献:** 将Deepfake检测构建为单类异常检测任务，通过学习定位人工植入的“软差异”来训练模型，不依赖伪造样本。
    *   **代码/项目链接:** [https://github.com/anonymous-author-sub/seeable](https://github.com/anonymous-author-sub/seeable)
*   **RECCE (2022, CVPR):** *Cao et al.*
    *   **主要贡献:** 提出RECCE框架，通过仅在真实人脸上进行重建学习来建模正样本（真实）分布。
    *   **代码/项目链接:** 未提及
*   **SIGMA-DF (2023, ICMR):** *Han, B., 等*
    *   **主要贡献:** 提出新颖的集成元学习框架，通过模拟多重跨域场景和挖掘难例样本，学习更泛化的“真实”决策边界。
    *   **代码/项目链接:** 未提及
*   **ID³ (2022, IEEE TMM):** *Yin, Z., 等*
    *   **主要贡献:** 将不变风险最小化（IRM）范式引入Deepfake检测，强制模型学习跨域不变的“真实”特征。
    *   **代码/项目链接:** [https://github.com/Yzx835/InvariantDomainorientedDeepfakeDetection](https://github.com/Yzx835/InvariantDomainorientedDeepfakeDetection)
*   **Detecting Generated Images... (LNP) (2023, arXiv):** *Bi, X. 等*
    *   **主要贡献:** 提出仅使用真实图像训练的“真实分布学习”新范式，通过单类分类检测各类生成图像。
    *   **代码/项目链接:** 未提及

#### **通用合成图像检测**

##### **1. 基于生成模型重建/反演 (Generative Model Reconstruction/Inversion)**
*   **FIRE (2025, arXiv):** *Beilin Chu et al.*
    *   **主要贡献:** 首次将频率分解思想融入基于重建的检测方法，利用DM难以重建真实图像中频信息的先验进行检测。
    *   **代码/项目链接:** [https://github.com/Chuchad/FIRE](https://github.com/Chuchad/FIRE)
*   **STRE (2025, ICASSP):** *Chengji Shen et al.*
    *   **主要贡献:** 依赖“由扩散模型生成的图像比真实图像更容易被任何扩散模型重建”的先验，利用整个时间序列的重建误差（TRE）作为特征。
    *   **代码/项目链接:** 未提及
*   **AEROBLADE (2024, CVPR):** *Ricker, J. 等*
    *   **主要贡献:** 提出AEROBLADE，一种无需训练的LDM图像检测方法，揭示了LDM的AE对生成图像的重建误差远低于真实图像。
    *   **代码/项目链接:** [https://github.com/jonasricker/aeroblade](https://github.com/jonasricker/aeroblade)
*   **DRCT (2024, ICML):** *Baoying Chen 等*
    *   **主要贡献:** 提出通用训练框架DRCT，通过扩散模型重建真实图像生成高质量难样本，并结合对比学习，提升对“真实”分布的辨别力。
    *   **代码/项目链接:** [https://github.com/beibuwandeluori/DRCT](https://github.com/beibuwandeluori/DRCT)
*   **FakeInversion (2024, Preprint):** *George Cazenavette 等*
    *   **主要贡献:** 提出FakeInversion检测器，利用从固定预训练的Stable Diffusion模型中反演出的特征（噪声图和重建图）来泛化检测。
    *   **代码/项目链接:** [https://fake-inversion.github.io/](https://fake-inversion.github.io/)
*   **LaRE² (2025, Arxiv):** *Luo et al.*
    *   **主要贡献:** 提出LaRE(隐空间重建误差)作为高效的伪造特征,并在EGRE模块中引导图像特征进行优化。
    *   **代码/项目链接:** 未提及
*   **DIRE (2023, arXiv):** *Wang, Z. 等*
    *   **主要贡献:** 提出DIRE，一种基于扩散模型重建误差的通用表示方法，用于检测偏离真实分布的合成图像。
    *   **代码/项目链接:** [https://github.com/ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE)
*   **DistilDIRE (2024, Preprint):** *Yewon Lim 等*
    *   **主要贡献:** 通过知识蒸馏，创建了一个轻量、快速的DIRE版本。
    *   **代码/项目链接:** [https://github.com/miraflow/DistilDIRE](https://github.com/miraflow/DistilDIRE)
*   **Implicit Detector (2023, ECCVW):** *Xi Wang, Vicky Kalogeiton*
    *   **主要贡献:** 提出利用预训练的扩散模型作为特征提取器，通过分析模型对噪声的响应来区分真实与伪造分布。
    *   **代码/项目链接:** [https://www.lix.polytechnique.fr/vista/projects/2024_detector_wang](https://www.lix.polytechnique.fr/vista/projects/2024_detector_wang)
*   **SeDID (2023, ICML Workshop):** *Ruipeng Ma 等*
    *   **贡献:** 利用扩散模型前向和后向过程中的 stepwise error 作为区分真实与生成图像的特征。
    *   **代码/项目链接:** 未提及
*   **Beyond the Spectrum (2021, arXiv):** *Yang He et al.*
    *   **贡献:** 提出基于重合成的伪影检测框架, 通过比较真实/伪造图像与重合成图像的残差进行检测。
    *   **代码/项目链接:** [https://github.com/SSAW14/BeyondtheSpectrum](https://github.com/SSAW14/BeyondtheSpectrum)

##### **2. 基于基础模型的方法 (Foundation Model-based Methods)**
*   **Effort (2025, ICML):** *Zhiyuan Yan et al.*
    *   **主要贡献:** 通过SVD将VFM特征正交分解为“保留预训练知识（真实分布）”和“学习伪造模式”的子空间。
    *   **代码/项目链接:** [https://github.com/YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection)
*   **Wavelet-CLIP (2025, arXiv):** *Baru, L. B. 等*
    *   **主要贡献:** 提出将CLIP的强泛化视觉特征与小波变换相结合，学习更细粒度的伪影。
    *   **代码/项目链接:** [https://github.com/lalithbharadwajbaru/wavelet-clip](https://github.com/lalithbharadwajbaru/wavelet-clip)
*   **SIDA (2025, arXiv):** *Huang et al.*
    *   **主要贡献:** 提出SIDA框架，利用大语言模型实现对社交媒体图像的检测、定位和解释。
    *   **代码/项目链接:** [https://hzlsaber.github.io/projects/SIDA/](https://hzlsaber.github.io/projects/SIDA/)
*   **UniFD (2023, CVPR):** *Ojha, U. 等*
    *   **主要贡献:** 提出使用大型预训练模型（CLIP）的冻结特征空间，配合简单的分类器，以OOD（分布外）检测的思路实现泛化。
    *   **代码/项目链接:** [https://github.com/Yuheng-Li/UniversalFakeDetect](https://github.com/Yuheng-Li/UniversalFakeDetect)
*   **GASE-Net (2022, IEEE SPL):** *Li, W. 等*
    *   **主要贡献:** 将GAN检测问题建模为伪造伪影与已知伪影参考集的相似度估计问题。
    *   **代码/项目链接:** 未提及

##### **3. 基于内在统计先验 (Intrinsic Statistical Priors)**
*   **AI-Synthesized Image Detection: Source Camera Fingerprinting (2025, IEEE Access):** *Manisha, 等*
    *   **贡献:** 将用于源相机识别的鲁棒全局指纹技术（真实图像的内在统计特性）成功应用于AI合成图像检测。
    *   **代码/项目链接:** 未提及
*   **SPAI (2025, CVPR):** *Karageorgiou et al.*
    *   **贡献:** 提出将真实图像的频谱分布作为不变性先验，通过掩码频谱学习进行建模。
    *   **代码/项目链接:** [https://mever-team.github.io/spai](https://mever-team.github.io/spai)
*   **B-Free (2025, CVPR):** *Guillaro, F. 等*
    *   **贡献:** 提出一种无偏训练范式，通过自条件生成和内容增强（修复）技术，强制模型学习生成过程的伪影，而非内容偏差。
    *   **代码/项目链接:** [https://github.com/grip-unina/B-Free](https://github.com/grip-unina/B-Free)
*   **Beyond Generation (2025, CVPR):** *Zhong, N. 等*
    *   **贡献:** 提出一种基于扩散模型的低层特征提取器, 通过自监督预训练任务学习真实图像的内在特征分布。
    *   **代码/项目链接:** 未提及

#### **服务于“存真”的支撑技术**

##### **1. 旨在学习更纯粹“真实”分布的训练策略**
*   **QC-Sampling (2023, MAD):** 提出一种基于质量的训练样本采样策略，通过筛选高质量伪造图像（更接近真实分布边界的负样本）进行训练，以学习更精确的决策边界。
*   **Combating Dataset Misalignment (2024, WDC'25):** 证明了数据集未对齐是影响学习真实分布的关键原因，在对齐的数据集上训练能显著提升鲁棒性。
*   **T-GD (2020, ICML):** 使用教师-学生自训练框架，将从一个已知“真实vs伪造”分布中学到的知识，迁移到新的分布上。

##### **2. 持续更新对“真实”分布认知的策略**
*   **HIDD (2025, ICME):** 以人类感知为中心的增量学习框架，持续更新模型对新出现伪造类型的认知边界。
*   **Continuous fake media detection (2024, CVIU):** 使用KD/EWC的持续学习框架来适配和学习新的伪造技术，不断调整对“真实”的定义。

##### **3. 轻量化与高效实现“存真”范式的方法**
*   **AOT-PixelNet (2025, Applied Soft Computing):** 自适应正交变换+极简PixelNet的轻量通用检测。
*   **LAID (2025, arXiv):** 提出轻量级AIGC检测的Benchmark，并评估了多种轻量化模型。
*   **Lightweight CNN for DFDC (2025, ComTech):** MTCNN预处理+轻量CNN的人脸区域检测。

##### **4. 基于“真实”分布的特殊应用**
*   **Art or Artifact? (2025, WDC'25):** 检测+分割多任务框架，在判断真伪的同时，定位偏离“真实艺术”分布的区域。
*   **M‑Task‑SS (2024, ICT Express):** 多任务自监督（仅真实图像）提升跨库检测。
*   **Forensic Self-Descriptions (2023, arXiv):** 仅需真实图像训练，通过对预测滤波器的残差建模实现零样本检测和开集溯源。

---

### **纯粹的综述、基准与分析研究**
这些论文不提出新的检测方法，而是对领域进行总结、评估或提供工具。

*   **SIDBench (2024, MAD):** 提出了一个模块化的基准测试框架（SIDBench）。 ([代码](https://github.com/mever-team/sidbench))
*   **Online Detector (2023, ICCVW):** 在模拟模型发布顺序的“在线”设置中研究AIGC检测。 ([项目主页](https://richzhang.github.io/OnlineGenAIDetection/))
*   **Community Forensics (2025, arXiv):** 创建了包含4803个生成器的超大规模数据集Community Forensics。 ([代码/数据](https://jespark.net/projects/2024/community_forensics))
*   **DE-FAKE (2023, CCS):** 对文生图模型进行系统的检测与溯源研究，并提出了结合图像和文本prompt的混合检测器。 ([代码](https://github.com/zeyangsha/De-Fake))
*   **CNNDetection (2020, CVPR):** 证明了在单个GAN(ProGAN)上训练的简单CNN分类器，经过数据增强，可以泛化到多种未见过的生成器。 ([代码](https://github.com/peterwang512/CNNDetection/))
*   **Robust Deepfake Detection...: A Short Survey (2024, MIS):** 全面综述了鲁棒深度伪造检测的研究现状。
*   **A Review of Deepfake Techniques... (2024, IEEE Access):** 对Deepfake检测领域的关键挑战、近期成功和未来研究方向进行了全面的元文献综述。
*   **Deepfake Generation and Detection: Case Study... (2023, IEEE Access):** 全面综述了Deepfake的生成与检测技术。
*   **Deepfake_Detection_Analyzing_Model_Generalization... (2023, IEEE Access):** 全面对比了CNN和Transformer在Deepfake检测泛化性上的表现。
*   **Towards Generalization in Deepfake Detection (Keynote) (2022, IH&MMSec):** Keynote演讲摘要，强调了域泛化（domain generalization）的重要性。

---

## 开源代码 (Code)

我们强烈建议在 `[论文列表 (Papers)](#论文列表-papers)` 部分为每篇论文附上其官方或高质量的实现代码链接。此部分可用于汇总一些特别有价值的工具库或框架。

*   **[项目名称]**: [项目简介]，[链接到GitHub仓库]

---

## 数据集 (Datasets)

以下是本领域常用的一些公开数据集。

| 数据集名称 | 主要特点 | 官网/下载链接 |
| :--- | :--- | :--- |
| **FaceForensics++ (FF++)** | 包含Deepfakes, Face2Face, FaceSwap, NeuralTextures四种主流伪造方法 | [Link](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF (v2)** | 高质量、少伪影的Deepfake视频，更具挑战性 | [Link](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC (Deepfake Detection Challenge)** | Kaggle竞赛数据集，规模巨大，包含多种混淆和攻击 | [Link](https://ai.facebook.com/datasets/dfdc/) |
| **WildDeepfake** | 从互联网收集的真实场景下的Deepfake视频 | [Link](https://www.di.ens.fr/willows/research/wilddeepfake/) |
| **GenImage** | 包含多种GAN和Diffusion模型的通用AIGC图像数据集 | [Link](https://github.com/GenImage-Dataset/GenImage) |
| **Community Forensics** | 包含4800+生成器和270万张图像的超大规模数据集 | [Link](https://jespark.net/projects/2024/community_forensics) |
| **SIDBench** | 用于可靠评估合成图像检测方法的Python框架及相关数据集 | [Link](httpsgithcom/mever-team/sidbench) |

---

## 方法对比 (Benchmark)

以下是一些代表性方法在主流数据集上的公开评测结果。欢迎提交您的结果！

| 方法名称 | 主干网络 | 数据集 | 评价指标 (ACC % / AUC %) | 对应论文 | 代码链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XceptionNet | Xception | FF++ (c23) | 99.65 / 99.8 | *MesoNet* | [Link] |
| UniFD | CLIP ViT-L/14 | ProGAN(Train) -> Diffusion(Test) | 81.38 / - | *UniFD* | [Link](https://github.com/Yuheng-Li/UniversalFakeDetect) |
| DIRE | ResNet-50 | ADM(T) -> 7 DMs(Test) | 99.9 / - | *DIRE* | [Link](https://github.com/ZhendongWang6/DIRE) |
| ... | ... | ... | ... | ... | ... |

---

## 贡献指南 (Contribution)

我们非常欢迎任何形式的贡献！如果您有好的建议、发现了新的论文/代码/数据集，或者发现了错误，请随时通过以下方式贡献：

1.  **Fork** 本项目
2.  创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4.  将更改推送到分支 (`git push origin feature/AmazingFeature`)
5.  提交一个 **Pull Request**

您也可以直接开启一个 [Issue](https://github.com/YOUR_USERNAME/Awesome-Deepfake-Detection/issues) 来进行讨论。

---

## 许可证 (License)

本项目采用 MIT 许可证。详情请见 `LICENSE` 文件。

Copyright (c) 2024 [Your Name or Organization]
