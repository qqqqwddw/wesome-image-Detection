# Awesome image Detection

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
*   **HFF (2021, CVPR):** 证明利用多尺度高频噪声可以提升泛化能力。 ([实现](https://github.com/crywang/face-forgery-detection))
*   **SPSL (2021, CVPR):** 首次利用相位谱并结合浅层网络来检测伪造。 ([论文链接](https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Spatial-Phase_Shallow_Learning_A_Low-Complexity_Spatial-Phase_Shallow_Network_for_Face_CVPR_2021_paper.html))
*   **F³-Net (2020, ECCV):** 设计频率感知模块，有效挖掘频域线索，尤其在低质量图像上效果显著。 ([论文链接](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3034_ECCV_2020_paper.php))
*   **Watch Your Up-Convolution (2020, CVPR):** 首次揭示基于上采样的GAN会产生明显的频谱失真。 ([代码](https://github.com/cc-hpc-itwm/UpConv))
*   **Wavelet-packets (2022, Machine Learning):** 首次将多尺度小波包变换用于合成图像分析。 ([代码](https://github.com/v0lta/PyTorch-Wavelet-Toolbox))
*   **D4 (2024, WACV):** 基于频率域特征解耦的离散集成模型，提升对抗鲁棒性。 ([代码](https://github.com/nmangaokar/wacv_24_d4))
*   **FDFL (2021, ArXiv):** 结合频率感知和单中心损失，学习更有区分度的伪造特征。 ([论文链接](https://arxiv.org/abs/2108.06209))
*   **FreqDebias (2024, ArXiv):** 设计Fo-Mixup频率域增强和双一致性正则化来解决“频谱偏见”问题。 ([论文链接](https://arxiv.org/abs/2402.04930))

##### **2. 混合边界与合成伪影 (Blending Boundary & Synthesis Artifacts)**
*   **Face X-ray (2020, CVPR):** 将伪造检测问题转化为寻找图像中的“混合边界”。 ([论文链接](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Face_X-Ray_for_More_General_Face_Forgery_Detection_CVPR_2020_paper.html))
*   **TTNet (2024, Pattern Recognition):** 提出人脸关键点引导的踪迹增强模块以提升泛化性。 ([代码](https://github.com/Gao-ning/TTNet-Trail-Tracing-Network))
*   **SBI (2022, CVPR):** 通过单张图像自混合生成高质量伪造样本，提升模型对未知伪造的泛化能力。 ([代码](https://github.com/mapooon/SelfBlendedImages))
*   **FSBI (2022, IVC):** 在频域增强的自混合图像数据，进一步提升泛化。 ([代码](https://github.com/gufranSabri/FSBI))
*   **Explore and Enhance... (BBMG+NRS) (2023, ICCVM):** 提出BBMG和NRS分别模拟篡改痕迹和抑制噪声样本。 ([论文链接](https://www.scitepress.org/Link.aspx?doi=10.5220/0012173300003660))
*   **Leveraging edges and optical flow... (2020, IEEE):** 将边缘图和光流图作为额外输入，增强对时空不一致性的捕捉能力。 ([论文链接](https://ieeexplore.ieee.org/abstract/document/9244400/))

##### **3. 解耦伪造特征 (Decoupling Forgery Features)**
*   **UCF (2023, ICCV):** 设计多任务解耦框架，分离内容、特定伪造和通用伪造三部分特征。 ([论文链接](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_UCF_Uncovering_Common_Features_for_Generalizable_Deepfake_Detection_ICCV_2023_paper.html))
*   **TAD (2024, Signal Processing):** 提出纹理与伪造伪影分解的双分支检测网络。 ([代码](https://github.com/iamwangyabin/TAD))
*   **SAGNet (2025, IEEE TIFS):** 通过对抗训练解耦语义内容与伪造痕迹，解决小样本训练下的内容偏见问题。 ([代码](https://github.com/rstao-bjtu/SAGNet))
*   **IDCNet (2023, ArXiv):** 将图像分解为“全局内容”和“局部细节”两个视图，并通过跨视图蒸馏增强对局部伪造的敏感度。 ([代码](https://github.com/wangzhiyuan120/idcnet))
*   **Improving the Generalization Ability... via Disentangled Representation Learning (2021, ICIP):** 设计编码器-解码器结构，通过解耦表示学习自动分离伪造相关区域与无关背景。 ([论文链接](https://ieeexplore.ieee.org/document/9506637))
*   **CADDM (2023, CVPR):** 发现并验证了“隐式身份泄露”问题，并设计伪影检测模块来强制模型关注局部伪影。 ([代码](https://github.com/megvii-research/CADDM))

#### **通用合成图像检测**

##### **1. 频率与统计指纹 (Frequency & Statistical Fingerprints)**
*   **Synthbuster (2024, IEEE OJSP):** 通过简单的交叉差分高通滤波器揭示并检测扩散模型的频域伪影。 ([数据集](https://zenodo.org/records/10066460))
*   **DCT-Traces (2024, arXiv):** 分析GAN和DM图像在DCT系数上的统计指纹。 ([代码](https://github.com/opontorno/dcts_analysisdeepfakes))
*   **UGAD (2024, CIKM):** 结合径向积分操作(RIO)和空间傅里叶提取(SFE)的多模态频域检测方法。 ([论文链接](https://dl.acm.org/doi/10.1145/3684333.3684420))
*   **Frank-ICML (DCT谱) (2020, ICML):** 首次系统性地揭示了GAN在频域中普遍存在的“棋盘状”伪影。([代码](https://github.com/RUB-SysSec/GANDCTAnalysis))
*   **FourierSpectrum (2020, NeurIPS):** 发现真实与生成图像在高频傅立葉谱衰减率上存在系统性差异。 ([论文链接](https://proceedings.neurips.cc/paper/2020/file/46a439c6253458a620FA00197760A813-Paper.pdf))
*   **Benford’s Law (2020, ICPR):** 证明GAN生成图像不符合本福特定律，并基于DCT系数的偏差设计检测器。 ([代码](https://github.com/polimi-ispl/icpr-benford-gan))
*   **Detecting and Simulating... (AutoGAN) (2019, WIFS):** 发现上采样在频域中产生频谱复制的伪影。 ([代码](https://github.com/ColumbiaDVMM/AutoGAN))
*   **DIF (2024, WACV):** 利用CNN架构固有指纹进行低样本量的AIGC检测与模型谱系分析。 ([项目主页](https://sergo2020.github.io/DIF/))
*   **Discovering Transferable Forensic Features... (2022, arXiv):** 提出FF-RS方法，并揭示了**颜色**是一个被忽视但至关重要的可迁移法证特征。 ([论文链接](https://arxiv.org/abs/2210.12035))
*   **FreqCross (2025, arXiv):** 融合空间、频谱和径向能量分布特征的三分支网络。 ([论文链接](https://arxiv.org/abs/2403.04702))
*   **FreqNet (2022, AAAI):** 提出轻量级FreqNet，通过高频特征表示和频率卷积层强制网络在频率空间学习。 ([代码](https://github.com/chuan-tan/FreqNet))
*   **Faster Than Lies (BNN-based) (2024, Preprint):** 首次将二值神经网络（BNN）应用于Deepfake检测，结合FFT和LBP特征。 ([代码](https://github.com/fedeloper/binary_deepfake_detection))
*   **Mastering Deepfake... (Hierarchical Framework) (2024, ACM TMM):** 提出分层级多级分类框架，能依次判别真伪、生成范式和具体模型。 ([项目主页](https://iplab.dmi.unict.it/mfs/Deepfakes/MasteringDeepfake2023/))

##### **2. 结构与梯度伪影 (Structural & Gradient Artifacts)**
*   **NPR (2024, CVPR):** 提出NPR（邻近像素关系）作为一种简单、通用的伪影表示。 ([代码](https://github.com/chuangchuangtan/NPR-DeepfakeDetection))
*   **DIO (2024, arXiv):** 提出数据无关算子(DIO)框架，使用固定的、无需训练的滤波器作为伪影提取器。 ([代码](https://github.com/chuangchuangtan/Data-Independent-Operator))
*   **PiD (2025, PMLR):** 提出一种高效、无需生成器的像素级分解残差方法（PiD）。 ([论文链接](http://proceedings.mlr.press/v238/han24a.html))
*   **Rich/Poor Texture Contrast (2023, arXiv):** 提出基于图像贫富纹理区域间像素相关性对比的通用伪造指纹。 ([项目主页](https://fdmas.github.io/AIGCDetect/))
*   **LGrad (2023, CVPR):** 利用预训练CNN模型提取的梯度作为广义伪影表示，提升泛化性。 ([代码](https://github.com/chuangchuangtan/LGrad))
*   **Detecting GAN generated... (Co-occurrence) (2019, arXiv):** 提出一种基于像素共生矩阵和CNN的GAN图像检测方法。 ([论文链接](https://arxiv.org/abs/1909.05142))
*   **Detection, Attribution and Localization... (Multi-directional Co-occurrence) (2020, arXiv):** 提出一种基于多方向共生矩阵和XceptionNet的检测、归因、定位框架。 ([论文链接](https://arxiv.org/abs/2005.02056))
*   **Critical Analysis... (no-down variant) (2021, ICME):** 证明取消CNN模型早期的下采样操作能更好地保留高频伪影。 ([代码](https://github.com/grip-unina/GANimageDetection))
*   **Leveraging Image Gradients... (GM-Net, etc.) (2023, VCIP):** 利用图像梯度(幅度和方向)作为鲁棒特征，设计浅层CNN架构。 ([论文链接](https://ieeexplore.ieee.org/document/10403759))

##### **3. 基于基础模型的伪造痕迹提取 (Forgery Trace Extraction using Foundation Models)**
*   **UniFD (2023, CVPR):** 使用大型预训练模型（CLIP）的冻结特征空间，以OOD检测的思路寻找泛化伪影。 ([代码](https://github.com/Yuheng-Li/UniversalFakeDetect))
*   **Effort (2025, ICML):** 通过SVD将VFM特征正交分解为“保留预训练知识”和“学习伪造模式”的子空间。 ([代码](https://github.com/YZY-stack/Effort-AIGI-Detection))
*   **FatFormer (2024, CVPR):** 在CLIP上引入伪造感知适配器（图像+频率域）与语言引导对齐，实现强泛化。 ([代码](https://github.com/Michel-liu/FatFormer))
*   **Wavelet-CLIP (2025, arXiv):** 将CLIP的强泛化视觉特征与小波变换相结合，以捕捉更细粒度的伪影。 ([代码](https://github.com/lalithbharadwajbaru/wavelet-clip))
*   **D³ (2025, arXiv):** 通过引入图像块打乱的“差异”信号，促进VFM学习通用伪影。 ([代码](https://github.com/BigAandSmallq/D3))
*   **SIDA (2025, arXiv):** 利用大语言模型作为控制器，调度多种工具对伪造痕迹进行检测、定位和解释。 ([项目主页](https://hzlsaber.github.io/projects/SIDA/))
*   **CLIP-based Detector (2024, CVPRW):** 证明了基于CLIP的轻量级检测器，仅需少量样本训练，就能实现极强的泛化能力。 ([项目主页](https://grip-unina.github.io/ClipBased-SyntheticImageDetection/))
*   **C2P-CLIP (2024, AAAI):** 通过注入“类别共同提示”来增强CLIP图像编码器对伪造痕迹的捕捉能力。 ([代码](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection))
*   **LASTED (2025, Arxiv):** 通过精心设计的文本标签进行语言引导的对比学习，学习更具泛化性的视觉-语言联合伪造特征。 ([代码](https://github.com/HighwayWu/LASTED))
*   **GLFAFormer (2024, DSP):** 冻结CLIP骨干，增加自适应的局部和全局对齐模块，以更好地捕捉局部伪影。 ([代码](https://github.com/long2580h/GLFAFormer))
*   **Universal Detection and Source Attribution... (2023, PReMI):** 证明了在扩散模型上训练的ResNet-50能有效泛化到多种GAN生成的图像。 ([论文链接](https://link.springer.com/chapter/10.1007/978-981-99-8548-9_24))
*   **GASE-Net (2022, IEEE SPL):** 将GAN检测问题建模为未知伪影与已知伪影参考集的相似度估计问题。 ([论文链接](https://ieeexplore.ieee.org/document/9868037))

#### **服务于“证伪”的策略与架构 (Strategies & Architectures for "Forgery Trace Seeking")**

##### **1. 旨在增强伪造痕迹捕捉能力的训练策略**
*   **LSDA (2024, CVPR):** 在潜在空间进行数据增强，通过域内和跨域增强来扩大伪造空间。 ([论文链接](https://openaccess.thecvf.com/content/CVPR2024/html/Yan_Latent_Space_Domain_Augmentation_for_Generalizable_Deepfake_Detection_CVPR_2024_paper.html))
*   **SLADD (2022, CVPR):** 通过对抗性地生成多样化且困难的伪造样本来提升检测器的泛化能力。 ([代码](https://github.com/liangchen527/SLADD))
*   **Adv. Augment (Recurrent Attacks) (2024, MAD):** 通过对真实图像进行循环对抗攻击，生成能骗过当前检测器的新伪造样本。 ([论文链接](https://dl.acm.org/doi/10.1145/3639433.3639437))
*   **AFSL (2024, MAD):** 提出AFSL损失函数，通过优化特征相似性，提升对抗鲁棒性。 ([论文链接](https://dl.acm.org/doi/10.1145/3639433.3639436))
*   **SAFE (2025, KDD):** 通过优化的图像变换策略（裁剪、颜色抖动等）来保留和增强伪影。 ([代码](https://github.com/Ouxiang-Li/SAFE))
*   **APN (2024, arXiv):** 提出一个伪影纯化网络(APN)，通过解耦和提纯伪影特征，让模型更专注于学习核心伪造痕迹。 ([论文链接](https://arxiv.org/abs/2402.14660))
*   **SR-CDT (2024, Pattern Recognition):** 通过面部结构破坏和对抗性拼图损失来减少对语义信息的依赖。 ([论文链接](https://doi.org/10.1016/j.patcog.2024.110292))
*   **Fingerprint Domain Augmentation (2022, arXiv):** 通过自动编码器提取并扰动GAN指纹，再加回图像以生成新的训练样本。 ([论文链接](https://arxiv.org/abs/2207.09262))
*   **Contrastive learning-based general Deepfake detection (2023, JKSUCIS):** 结合多尺度RGB与频域线索的监督对比学习。 ([论文链接](https://doi.org/10.1016/j.jksuci.2023.101662))
*   **FIC (2023, Neural Networks):** 提出特征独立约束与对齐以减轻伪相关。 ([论文链接](https://doi.org/10.1016/j.neunet.2023.01.011))
*   **Frequency Masking (2025, ICASSP):** 在训练中对输入图像进行频率掩码作为一种有效的数据增强/正则化方法。 ([论文链接](https://ieeexplore.ieee.org/document/10446738))

##### **2. 旨在融合多种伪造痕迹的架构**
*   **A Hybrid Model for Generalizable... (2025, SCID):** 结合伪造边界、语义和通用伪影的三合一混合检测模型。 ([论文链接](https://www.scitepress.org/Link.aspx?doi=10.5220/0012891300003881))
*   **LightweightViT (2025, IEEE Access):** 轻量级ViT模型，利用自注意力机制高效检测图像篡改。 ([论文链接](https://ieeexplore.ieee.org/document/10452395))
*   **CVCNet (2024, MM Asia):** 融合CNN局部特征和ViT全局特征的序列性Deepfake检测模型。 ([论文链接](https://dl.acm.org/doi/10.1145/3664875.3664883))
*   **Self-Attention CNN (2020, JSTSP):** 使用自注意力机制来捕获GAN生成图像中的全局信息缺陷。 ([论文链接](https://ieeexplore.ieee.org/document/9056345))
*   **A Hybrid Deep Learning Framework... (TempCNN) (2025, IEEE Access):** 结合空间和时间特征的轻量级高效框架。 ([论文链接](https://ieeexplore.ieee.org/document/10451996/))
*   **Fusing Global and Local Features (PSM+AFFM) (2023, arXiv):** 通过注意力机制融合全局特征和由“补丁选择模块”自动选择的关键局部补丁特征。 ([论文链接](https://arxiv.org/abs/2202.13526))
*   **Spatiotemporal texture + 3D CNN fusion (2024, EIJ):** 结合时空纹理和3D CNN的孪生网络架构。 ([论文链接](https://onlinelibrary.wiley.com/doi/abs/10.1049/ell2.12879))
*   **Attention-Enhanced CNN... (2025, IEEE Access):** 计算高效的CNN-MHSA混合架构，并通过集成学习提高泛化。 ([论文链接](https://ieeexplore.ieee.org/document/10450531/))
*   **Fine-grained deepfake detection based on cross-modality attention (2023, NCA):** 融合RGB、频率和纹理三种模态特征的跨模态注意力网络。 ([论文链接](https://doi.org/10.1007/s00521-023-08577-0))
*   **MFCF‑Net (2024, ArXiv):** 多尺度特征+跨域融合用于图像篡改定位。 ([论文链接](https://arxiv.org/abs/2402.04609))
*   **A Solution to ACMMM 2024... (NPR+TIMM集成) (2024, MM):** 竞赛方案，集成NPR特征和强主干网络。 ([论文链接](https://dl.acm.org/doi/10.1145/3652870.3656112))

##### **3. 针对伪造痕迹的对抗攻防与可解释性**
*   **TAG-WM (2025, arXiv):** 篡改感知的生成式水印方法，主动嵌入可追溯的“痕迹”。 ([代码](https://github.com/Suchenl/TAG-WM))
*   **DF-UDetector (2025, Neural Networks):** 在特征空间进行“伪影恢复”以增强退化场景（痕迹被破坏）下的鲁棒性。 ([论文链接](https://doi.org/10.1016/j.neunet.2024.105748))
*   **Adversarial Robustness in DeepFake Detection... (2024, ICICyTA):** 验证了针对伪造痕迹的对抗训练和防御策略的有效性。 ([论文链接](https://link.springer.com/chapter/10.1007/978-3-031-50899-4_15))
*   **Counter-Forensic (CLIP+DCT) (2024, ICVGIP):** 将DM重构真实图像的行为定义为一种反取证攻击（抹除痕迹）。 ([论文链接](https://dl.acm.org/doi/10.1145/3693992.3694002))
*   **MaskSim (2024, CVPRW):** 通过学习可解释的频谱掩模来识别特定生成器的频率指纹。 ([代码](https://github.com/li-yanhao/masksim))
*   **SpectralGAN (Attack) (2022, CVPR):** 证明了基于频谱伪影的检测器并非鲁棒，其依赖的伪影可被移除。 ([代码](https://www.comp.polyu.edu.hk/~csajaykr/deepdeepfake.htm))

---

### **范式二：真实分布学习范式 (Authentic Distribution Learning Paradigm)**
**核心逻辑：“存真”。** 致力于学习真实图像的本质分布，将任何偏离该分布的样本视为异常。

#### **1. 基于单类分类与异常检测 (One-Class Classification & Anomaly Detection)**
*   **SeeABLE (2023, ICCV):** 将Deepfake检测构建为单类异常检测任务，不依赖伪造样本。 ([代码](https://github.com/anonymous-author-sub/seeable))
*   **Stay-Positive (2025, ICML):** 通过约束分类器最后一层权重为非负，强制模型只学习偏离真实分布的伪造特征。 ([项目主页](https://anisundar18.github.io/Stay-Positive/))
*   **RECCE (2022, CVPR):** 通过仅在真实人脸上进行重建学习来建模正样本（真实）分布。 ([代码](https://github.com/VISION-SJTU/RECCE))
*   **MCS‑GAN (2022, IEEE TMM):** 仅用真实人脸训练GAN模型学习正常分布，通过重构误差或隐空间差异检测伪造。 ([论文链接](https://ieeexplore.ieee.org/abstract/document/9662366))
*   **Detecting Generated Images... (LNP) (2023, arXiv):** 仅使用真实图像训练的“真实分布学习”新范式。 ([论文链接](https://arxiv.org/abs/2303.02302))
*   **Beyond Generation (2025, CVPR):** 通过自监督预训练学习真实图像的内在低层特征分布, 并将AIGC检测视为单类异常检测。 ([论文链接](https://openaccess.thecvf.com/content/CVPR2024/html/Zhong_Beyond_Generation_Harnessing_Next-Gen_AI_for_Multi-Modal_Content_Creation_and_Understanding_CVPR_2024_paper.html))
*   **M‑Task‑SS (2024, ICT Express):** 多任务自监督（仅真实图像）提升跨库检测。 ([论文链接](https://www.sciencedirect.com/science/article/pii/S240595952300030X))
*   **Forensic Self-Descriptions (2023, arXiv):** 仅需真实图像训练，通过对预测滤波器的残差建模实现零样本检测和开集溯源。 ([论文链接](https://arxiv.org/abs/2312.06076))

#### **2. 基于内在一致性与物理先验 (Internal Consistency & Physical Priors)**
*   **Shadows Don’t Lie (2024, arXiv):** 证明了生成模型在投影几何（光影、透视）上存在系统性缺陷，这违反了真实世界的物理规律。 ([项目主页](https://projective-geometry.github.io/))
*   **IID (2023, CVPR):** 通过伪造人脸的“显式身份”和“隐式身份”之间的不一致性来检测人脸交换，这破坏了身份的内在一致性。 ([论文链接](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Implicit_Identity_Leakage_The_Stumbling_Block_to_Generalizable_Face_Forgery_CVPR_2023_paper.html))
*   **Bi-LIG + TCC-ViT (2024, IEEE TIFS):** 提出双层不一致性生成器(Bi-LIG)，使模型同时学习“外在”和“内在”不一致性。 ([论文链接](https://ieeexplore.ieee.org/abstract/document/10398606))
*   **GrDT (2021, WACVW):** 结合面部关键点的几何分布，检查其是否符合真实人脸的几何规律。 ([代码](https://github.com/SIPLab24/GrDT))

#### **3. 基于生成模型重建/反演 (Generative Model Reconstruction/Inversion)**
*   **FIRE (2025, arXiv):** 利用DM难以重建真实图像中频信息的先验进行检测。 ([代码](https://github.com/Chuchad/FIRE))
*   **AEROBLADE (2024, CVPR):** 利用LDM的AE对生成图像的重建误差远低于真实图像的特性。 ([代码](https://github.com/jonasricker/aeroblade))
*   **DIRE (2023, arXiv):** 提出DIRE，一种基于扩散模型重建误差的通用表示方法，其核心是真实图像重建误差更大。 ([代码](https://github.com/ZhendongWang6/DIRE))
*   **DRCT (2024, ICML):** 通过扩散模型重建真实图像生成高质量难样本，以提升对真实分布的辨别力。 ([代码](https://github.com/beibuwandeluori/DRCT))
*   **FakeInversion (2024, Preprint):** 利用从固定SD模型中反演出的特征（噪声图和重建图）来泛化检测。 ([项目主页](https://fake-inversion.github.io/))
*   **DistilDIRE (2024, Preprint):** 通过知识蒸馏，创建了一个轻量、快速的DIRE版本。 ([代码](https://github.com/miraflow/DistilDIRE))
*   **Implicit Detector (2023, ECCVW):** 利用预训练的扩散模型，通过分析模型对噪声的响应来区分真实与伪造分布。 ([项目主页](https://www.lix.polytechnique.fr/vista/projects/2024_detector_wang))
*   **SeDID (2023, ICML Workshop):** 利用扩散模型前向和后向过程中的 stepwise error作为区分特征。 ([论文链接](https://openreview.net/forum?id=uX3Bq5qYrS))
*   **Beyond the Spectrum (2021, arXiv):** 通过比较真实/伪造图像与重合成图像的残差进行检测。 ([代码](https://github.com/SSAW14/BeyondtheSpectrum))
*   **UR²EA (Revisiting DIRE) (2023, Neural Networks):** 提出通用重建残差分析，结合多尺度注意，统一GAN/DM检测。 ([论文链接](https://doi.org/10.1016/j.neunet.2023.10.021))
*   **LaRE² (2025, Arxiv):** 提出LaRE(隐空间重建误差)作为高效的伪造特征。 ([论文链接](https://arxiv.org/abs/2403.01860))
*   **SemGIR (2024, ACM MM):** 通过语义引导的图像再生成来解耦内容和模型指纹，适用于少样本检测。 ([论文链接](https://dl.acm.org/doi/10.1145/3651852.3654060))

#### **4. 基于内在统计先验 (Intrinsic Statistical Priors)**
*   **CoD (Secret Lies in Color) (2025, CVPR):** 提出基于色彩分布不均匀性的检测特征，这是真实图像的内在统计特性。 ([论文链接](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Secret_Lies_in_Color_A_Simple_and_Effective_Method_for_Detecting_AI-Generated_Images_CVPR_2024_paper.html))
*   **RIGID (2024, arXiv):** 提出无训练方法RIGID，依赖真实图像对噪声扰动更鲁棒的先验知识。 ([论文链接](https://arxiv.org/abs/2402.10221))
*   **AI-Synthesized Image Detection: Source Camera Fingerprinting (2025, IEEE Access):** 将用于源相机识别的鲁棒全局指纹技术（真实图像的内在统计特性）应用于AI合成图像检测。 ([论文链接](https://ieeexplore.ieee.org/document/10432363/))
*   **SPAI (2025, CVPR):** 将真实图像的频谱分布作为不变性先验，通过掩码频谱学习进行建模。 ([项目主页](https://mever-team.github.io/spai))
*   **RFFR (2024, arXiv):** 学习真实人脸的“基础表征”，再用与该表征的残差来检测伪造。 ([代码](https://github.com/shiliang26/RFFR))
*   **MIFAE-Forensics (2024, arXiv):** 通过在空间域和频率域上进行掩码自编码预训练，学习真实人脸的内在分布。 ([代码](https://github.com/Mark-Dou/Forensics))
*   **DinoHash/AI Detector (Proteus) (2025, arXiv):** 提出鲁棒的感知哈希DinoHash，其核心依赖于对真实图像分布的稳定哈希。 ([项目主页](https://proteus.photos))

#### **服务于“存真”的策略与架构 (Strategies & Architectures for "Authentic Distribution Learning")**

##### **1. 旨在学习更纯粹“真实”分布的训练策略**
*   **B-Free (2025, CVPR):** 提出一种无偏训练范式，通过生成与真实图像内容对齐的伪造样本，迫使模型学习生成伪影，而非内容偏差，从而更好地学习真实分布。 ([代码](https://github.com/grip-unina/B-Free))
*   **QC-Sampling (2023, MAD):** 通过筛选高质量伪造图像（更接近真实分布边界的负样本）进行训练，以学习更精确的决策边界。 ([代码](https://github.com/dogoulis/qc-sgid))
*   **Combating Dataset Misalignment (2024, WDC'25):** 证明了数据集未对齐是影响学习真实分布的关键原因。 ([论文链接](http://dx.doi.org/10.1145/3664724.3685657))
*   **UDD (2025, arXiv):** 在ViT潜在空间中进行token级干预，以学习无偏见的、更本质的真实图像特征分布。 ([论文链接](https://arxiv.org/abs/2403.01314))
*   **T-GD (2020, ICML):** 使用教师-学生自训练框架，将从一个已知“真实vs伪造”分布中学到的知识，迁移到新的分布上。 ([代码](https://github.com/cutz-j/T-GD))
*   **DABN (2021, ADVM W.):** 提出测试时自适应策略，通过使用测试批次的统计数据更新BN层，缓解域偏移。 ([论文链接](https://www.scitepress.org/Papers/2021/107147/107147.pdf))
*   **SIGMA-DF (2023, ICMR):** 提出新颖的集成元学习框架，通过模拟多重跨域场景，学习更泛化的“真实”决策边界。 ([论文链接](https://dl.acm.org/doi/10.1145/3591106.3592237))
*   **MAFD (2022, PRL):** 提出自引导的模型无关元学习框架，以提升泛化能力。 ([论文链接](https://doi.org/10.1016/j.patrec.2022.06.002))
*   **ID³ (2022, IEEE TMM):** 将不变风险最小化（IRM）范式引入Deepfake检测，强制模型学习跨域不变的“真实”特征。 ([代码](https://github.com/Yzx835/InvariantDomainorientedDeepfakeDetection))
*   **DSM (2025, Signal Processing):** 通过扰动域对齐（注意力遮挡+特征统计迁移）提升鲁棒性。 ([论文链接](https://doi.org/10.1016/j.sigpro.2024.109605))
*   **Cross-Domain Deepfake Detection... (LDKD) (2025, IEEE SPL):** 通过潜在域生成和知识蒸馏，实现无需访问源域数据的跨域检测。 ([论文链接](https://ieeexplore.ieee.org/document/10441459))
*   **JRC (2025, Neurocomputing):** 联合重建与分类的混合学习框架，其中重建任务旨在更好地学习真实图像的表示。 ([论文链接](https://doi.org/10.1016/j.neucom.2024.127623))
*   **Data Farming... (2022, IEEE TBIOM):** 通过混合真实人脸和背景切片来强制模型学习伪造伪影而非人脸特征。 ([论文链接](https://ieeexplore.ieee.org/document/9803154))

##### **2. 持续更新对“真实”分布认知的策略**
*   **HIDD (2025, ICME):** 以人类感知为中心的增量学习框架，持续更新模型对新出现伪造类型的认知边界。 ([论文链接](https://ieeexplore.ieee.org/abstract/document/10550474))
*   **Continuous fake media detection (2024, CVIU):** 使用KD/EWC的持续学习框架来适配和学习新的伪造技术，从而不断调整对“真实”的定义。 ([论文链接](https://doi.org/10.1016/j.cviu.2024.103986))
*   **Incr. Learning for the detection and classification of GAN-generated images (2019, WIFS):** 将增量学习思想应用于GAN检测，使模型能持续学习新生成器。 ([论文链接](https://ieeexplore.ieee.org/abstract/document/9028082))

##### **3. 轻量化与高效实现“存真”范式的方法**
*   **AOT-PixelNet (2025, Applied Soft Computing):** 自适应正交变换+极简PixelNet的轻量通用检测。 ([论文链接](https://doi.org/10.1016/j.asoc.2024.111818))
*   **LAID (2025, arXiv):** 提出轻量级AIGC检测的Benchmark，并评估了多种轻量化模型。 ([代码](https://github.com/nchivar/LAID))
*   **Lightweight CNN for DFDC (2025, ComTech):** MTCNN预处理+轻量CNN的人脸区域检测。 ([论文链接](https://www.scitepress.org/Link.aspx?doi=10.5220/0012803800003923))
*   **A Robust Framework for Deepfake Detection... (2025, ComTech):** 提出计算高效的轻量级CNN框架。 ([论文链接](https://www.scitepress.org/Link.aspx?doi=10.5220/0012803700003923))

##### **4. 基于“真实”分布的特殊应用**
*   **Art or Artifact? (2025, WDC'25):** 检测+分割多任务框架，定位偏离“真实艺术”分布的区域。 ([论文链接](http://dx.doi.org/10.1145/3664724.3685655))

---

### **纯粹的综述、基准与分析研究**
这些论文不提出新的检测方法，而是对领域进行总结、评估或提供工具。

*   **SIDBench (2024, MAD):** 提出了一个模块化的基准测试框架（SIDBench）。 ([代码](https://github.com/mever-team/sidbench))
*   **Online Detector (2023, ICCVW):** 在模拟模型发布顺序的“在线”设置中研究AIGC检测。 ([项目主页](https://richzhang.github.io/OnlineGenAIDetection/))
*   **Community Forensics (2025, arXiv):** 创建了包含4803个生成器的超大规模数据集Community Forensics。 ([代码/数据](https://jespark.net/projects/2024/community_forensics))
*   **DE-FAKE (2023, CCS):** 对文生图模型进行系统的检测与溯源研究，并构建了专用数据集。 ([代码](https://github.com/zeyangsha/De-Fake))
*   **CNNDetection (2020, CVPR):** 证明了在单个GAN上训练的简单CNN分类器可以泛化，并构建了ForenSynths基准。 ([代码](https://github.com/peterwang512/CNNDetection/))
*   **Optimizing AIGC Detection... (2024, MM):** 一篇AIGC检测竞赛的获胜方案报告。 ([论文链接](https://dl.acm.org/doi/10.1145/3652870.3656123))
*   **Robust Deepfake Detection...: A Short Survey (2024, MIS):** 全面综述了鲁棒深度伪造检测的研究现状。 ([论文链接](https://doi.org/10.1007/s11042-024-18456-4))
*   **A Review of Deepfake Techniques... (2024, IEEE Access):** 对Deepfake检测领域的关键挑战、近期成功和未来研究方向进行了综述。 ([论文链接](https://ieeexplore.ieee.org/document/10423019/))
*   **Deepfake Generation and Detection: Case Study... (2023, IEEE Access):** 全面综述了Deepfake的生成与检测技术。 ([论文链接](https://ieeexplore.ieee.org/document/10134449/))
*   **Deepfake_Detection_Analyzing_Model_Generalization... (2023, IEEE Access):** 全面对比了CNN和Transformer在Deepfake检测泛化性上的表现。 ([论文链接](https://ieeexplore.ieee.org/document/10134450/))
*   **Towards Generalization in Deepfake Detection (Keynote) (2022, IH&MMSec):** Keynote演讲摘要，强调了域泛化（domain generalization）的重要性。 ([论文链接](https://dl.acm.org/doi/10.1145/3531411.3533440))

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
| **SIDBench** | 用于可靠评估合成图像检测方法的Python框架及相关数据集 | [Link](https://github.com/mever-team/sidbench) |
| **Synthbuster Dataset** | 包含SD, MJ, Firefly, DALL·E 2/3等多种扩散模型生成图像的数据集 | [Link](https://zenodo.org/records/10066460) |

---

## 方法对比 (Benchmark)

以下是一些代表性方法在主流数据集上的公开评测结果。欢迎提交您的结果！

| 方法名称 | 主干网络 | 数据集 | 评价指标 (ACC % / AUC %) | 对应论文 | 代码链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XceptionNet | Xception | FF++ (c23) | 99.65 / 99.8 | *CNNDetection* | [Link](https://github.com/peterwang512/CNNDetection) |
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
