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

*   **2024-XX-XX:** 项目根据新的文献表格进行了全面重构和更新，完整收录超过160篇最新文献。
*   <!-- 在这里添加最新的项目更新 -->

---

## 论文列表 (Papers)

所有检测技术相关的论文根据其核心逻辑，被归类于**“证伪” (Forgery Trace Seeking)** 和 **“存真” (Authentic Distribution Learning)** 两大范式之下。

### **范式一：伪造痕迹追寻范式 (Forgery Trace Seeking Paradigm)**
**核心逻辑：“证伪”。** 主动寻找、增强、解耦和利用生成过程中遗留的各类痕迹（如频率伪影、混合边界、像素依赖关系等）作为检测证据。

#### **1. 频率域伪影 (Frequency-Domain Artifacts)**
*   **Detecting and Simulating... (Zhang et al., WIFS 2019):** 发现上采样在频域中产生频谱复制的伪影，并提出AutoGAN模拟器，使检测器训练不再依赖目标GAN模型。 ([代码](https://github.com/ColumbiaDVMM/AutoGAN))
*   **Watch Your Up-Convolution (Durall et al., CVPR 2020):** 首次系统性地揭示了基于上采样/转置卷积的GAN会产生明显的频谱失真，并利用该缺陷实现了高精度检测。 ([代码](https://github.com/cc-hpc-itwm/UpConv))
*   **Frank et al. (ICML 2020):** 首次系统性地揭示了GAN生成图像在频域中普遍存在严重的、可识别的“棋盘状”伪影,并证明其源于上采样操作。 ([代码](https://github.com/RUB-SysSec/GANDCTAnalysis))
*   **F³-Net (Qian et al., ECCV 2020):** 提出FAD和LFS两种频率感知模块，并设计双流协作学习框架，有效挖掘频域中的伪造线索，尤其在低质量压缩图像上效果显著。 ([论文搜索](https://scholar.google.com/scholar?q=Thinking+in+Frequency:+Face+Forgery+Detection+by+Mining+Frequency-aware+Clues))
*   **FourierSpectrum (Dzanic et al., NeurIPS 2020):** 首次发现真实图像与深度网络生成图像在高频傅立葉谱的衰减率上存在系统性差异，并基于此提出仅需少量样本的检测器。 ([论文搜索](https://scholar.google.com/scholar?q=Fourier+Spectrum+Discrepancies+in+Deep+Network+Generated+Images))
*   **FDFL (Li et al., CVPR 2021):** 提出单中心损失(SCL)以学习更有区分度的特征空间，并设计自适应频率特征生成模块(AFFGM)以数据驱动方式挖掘频率线索。 ([论文搜索](https://scholar.google.com/scholar?q=Frequency-Aware+Discriminative+Feature+Learning+Supervised+by+Single-Center+Loss+for+Face+Forgery+Detection))
*   **SPSL (Liu et al., CVPR 2021):** 依赖“上采样是多数伪造技术的必要步骤，并在相位谱留下痕迹”的先验，首次利用相位谱并结合浅层网络来检测伪造。 ([论文搜索](https://scholar.google.com/scholar?q=Spatial-Phase+Shallow+Learning:+Rethinking+Face+Forgery+Detection+in+Frequency+Domain))
*   **HFF (Luo et al., CVPR 2021):** 提出一个双流模型，证明利用多尺度高频噪声（SRM）并建模其与RGB特征的交互（DCMA），可以提升人脸伪造检测的泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Generalizing+Face+Forgery+Detection+With+High-frequency+Features))
*   **Wavelet-packets (Wolter et al., Machine Learning 2022):** 首次将多尺度小波包变换用于合成图像分析，揭示了GAN在时频域的均值和标准差异常，并构建了高效的分类器。 ([代码](https://github.com/v0lta/PyTorch-Wavelet-Toolbox))
*   **BiHPF (Jeong et al., WACV 2022):** 提出BiHPF，一种双边高通滤波器，通过放大高频和背景区域的伪影来提升对各类生成图像的鲁棒检测能力。 ([代码](https://github.com/SamsungSDS-Team9/BiHPF))
*   **FingerprintNet (Jeong et al., ECCV 2022):** 提出一种自监督框架，仅用真实图像训练一个“指纹生成器”来合成多样的伪造指纹，从而训练一个对未见GAN具有泛化能力的检测器。 ([论文搜索](https://scholar.google.com/scholar?q=FingerprintNet:+Synthesized+Fingerprints+for+Generated+Image+Detection))
*   **Synthbuster (Bammey, OJSP 2023):** 提出通过简单的交叉差分高通滤波器在单张图像上即可有效揭示并检测扩散模型的频域伪影。 ([论文搜索](https://scholar.google.com/scholar?q=Synthbuster:+A+simple+and+efficient+method+for+detecting+diffusion+models))
*   **D4 (Hooda et al., WACV 2024):** 提出D4，一个基于频率域特征解耦的离散集成模型，通过在不同频率子集上训练模型来提升对抗鲁棒性。 ([论文搜索](https://scholar.google.com/scholar?q=D4:+Detection+of+Adversarial+Diffusion+Deepfakes+Using+Disjoint+Ensembles))
*   **DCT-based Classifier (Pontorno et al., arXiv 2024):** 对GAN和DM图像的DCT系数（β_AC）进行统计分析，发现特定（尤其是低频）系数子集对检测更鲁棒和有效。 ([代码](https://github.com/opontorno/dcts_analysisdeepfakes))
*   **MaskSim (Li et al., CVPRW 2024):** 提出MaskSim，一种半白盒方法，通过学习频谱掩模和参考模式来识别特定生成器的频率指纹，结果可解释。 ([代码](https://github.com/li-yanhao/masksim))
*   **FreqNet (Tan et al., AAAI 2024):** 提出轻量级FreqNet，通过高频特征表示(HFRF)和频率卷积层(FCL)强制网络在频率空间学习，以增强泛化性。 ([代码](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection))
*   **FreqCross (Yang, arXiv 2025):** 提出FreqCross三分支网络，融合空间(RGB)、频谱(2D FFT)和新颖的径向能量分布特征，专门检测高级扩散模型。 ([论文搜索](https://scholar.google.com/scholar?q=FreqCross:+A+Multi-Modal+Frequency-Spatial+Fusion+Network+for+Advanced+Diffusion-Based+Deepfake+Detection))
*   **FreqDebias (Kashiani et al., CVPR 2025):** 发现并提出“频谱偏见”问题，设计Fo-Mixup频率域增强和双一致性正则化来提升检测器泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=FreqDebias:+Towards+Generalizable+Deepfake+Detection+via+Consistency-Driven+Frequency+Debiasing))
*   **DIO (Tan et al., arXiv 2024):** 提出数据无关算子(DIO)框架, 使用固定的、无需训练的滤波器(如随机初始化的卷积)作为伪影提取器, 实现了卓越的泛化性能。 ([代码](https://github.com/chuangchuangtan/Data-Independent-Operator))
*   **Benford’s Law (Bonettini et al., ArXiv 2020):** 证明了GAN生成的图像不符合本福特定律，并基于DCT系数与该定律的偏差设计了简单高效的特征和检测器。 ([论文搜索](https://scholar.google.com/scholar?q=On+the+use+of+Benford’s+law+to+detect+GAN-generated+images))

#### **2. 结构与梯度伪影 (Structural & Gradient Artifacts)**
*   **Detection of GAN-Generated... (Marra et al., MIPR 2018):** 对当时多种伪造检测器在GAN生成图像（特别是CycleGAN）上的性能进行了系统性评测，并率先研究了社交媒体压缩的影响。 ([论文搜索](https://scholar.google.com/scholar?q=Detection+of+GAN-Generated+Fake+Images+over+Social+Networks))
*   **Detecting GAN generated... (Nataraj et al., arXiv 2019):** 提出一种基于像素共生矩阵和CNN的GAN图像检测方法，在当时展现了优秀的性能和泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Detecting+GAN+generated+Fake+Images+using+Co-occurrence+Matrices))
*   **Detection, Attribution and Localization... (Goebel et al., arXiv 2020):** 提出一种基于多方向共生矩阵和XceptionNet的检测、归因、定位框架，并进行了大规模GAN数据集的评估。 ([论文搜索](https://scholar.google.com/scholar?q=Detection,+Attribution+and+Localization+of+GAN+Generated+Images))
*   **CNNDetection (Wang et al., CVPR 2020):** 证明了在单个GAN(ProGAN)上训练的简单CNN分类器, 经过仔细的数据增强, 可以惊人地泛化到多种未见过的CNN生成器。 ([代码](https://github.com/peterwang512/CNNDetection/))
*   **Critical Analysis... (Gragnaniello et al., ICME 2021):** 通过实验证明，取消CNN模型早期的下采样操作能更好地保留高频伪影，是提升GAN图像检测器泛化能力和鲁棒性的关键。 ([代码](https://github.com/grip-unina/GANimageDetection))
*   **LGrad (Tan et al., CVPR 2023):** 提出LGrad框架，利用预训练CNN模型提取的梯度作为广义伪影表示，将数据依赖问题转化为模型依赖问题，提升泛化性。 ([代码](https://github.com/chuangchuangtan/LGrad))
*   **Rich/Poor Texture Contrast (Zhong et al., arXiv 2023):** 提出基于图像贫富纹理区域间像素相关性对比的通用伪造指纹，并构建了综合性AIGC检测基准。 ([项目主页](https://fdmas.github.io/AIGCDetect/))
*   **Ghosh & Naskar (VCIP 2023):** 利用图像梯度(幅度和方向)作为鲁棒特征，设计了四种浅层CNN架构，在多种后处理和压缩攻击下保持了高检测精度。 ([论文搜索](https://scholar.google.com/scholar?q=Leveraging+Image+Gradients+for+Robust+GAN-Generated+Image+Detection+in+OSN+context))
*   **NPR (Tan et al., CVPR 2024):** 提出NPR（邻近像素关系）作为一种简单、通用的伪影表示，它捕获了生成模型中上采样操作引起的局部结构伪影。 ([代码](https://github.com/chuangchuangtan/NPR-DeepfakeDetection))
*   **SFLD (Gye et al., arXiv 2025):** 提出SFLD，通过多尺度PatchShuffle融合高层语义与低层纹理，减少内容偏见，并构建了高质量基准TwinSynths。 ([数据集](https://huggingface.co/datasets/koooooooook/TwinSynths))
*   **PPL (Yang et al., arXiv 2025):** 识别并解决了检测器中的“少数块偏差”，提出PPL框架强制模型学习全局伪影，其依据是生成伪影普遍存在于图像所有区域的先验知识。 ([论文搜索](https://scholar.google.com/scholar?q=All+Patches+Matter,+More+Patches+Better:+Enhance+AI-Generated+Image+Detection+via+Panoptic+Patch+Learning))
*   **APN (Meng et al., CVIU 2024):** 提出一个伪影纯化网络(APN)，通过解耦和提纯空域和频域中的伪影特征，提升了对跨生成器和跨场景（内容）的泛化检测能力。 ([论文搜索](https://scholar.google.com/scholar?q=Artifact+feature+purification+for+cross-domain+detection+of+AI-generated+images))

#### **3. 混合边界与时空伪影 (Blending Boundary & Spatiotemporal Artifacts)**
*   **Leveraging edges and optical flow (Chintha et al., IEEE 2020):** 将边缘图和光流图作为额外输入，与RGB帧信息融合，增强模型对时空不一致性的捕捉能力。 ([论文搜索](https://scholar.google.com/scholar?q=Leveraging+edges+and+optical+flow+on+faces+for+deepfake+detection))
*   **Face X-ray (Li et al., CVPR 2020):** 提出Face X-ray概念，将伪造检测问题转化为寻找图像中的“混合边界”，不依赖特定伪造痕迹，从而提高泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Face+X-Ray+for+More+General+Face+Forgery+Detection))
*   **T-GD (Jeon et al., ICML 2020):** 使用教师-学生自训练框架和L2-SP正则化，将知识从源GAN模型迁移到目标GAN模型，依赖于伪造图像共享可迁移特征的先验。 ([代码](https://github.com/cutz-j/T-GD))
*   **SLADD (Chen et al., CVPR 2022):** 通过对抗性增强伪造样本多样性，并用自监督任务（预测伪造配置）提升模型敏感度，以学习更可泛化的表征。 ([代码](https://github.com/liangchen527/SLADD))
*   **SBI (Shiohara & Yamasaki, CVPR 2022):** 提出一种数据合成方法SBI，通过单张图像自混合生成高质量伪造样本，提升模型对未知伪造的泛化能力。 ([代码](https://github.com/mapooon/SelfBlendedImages))
*   **Explore and Enhance... (Wang et al., ICCVM 2024):** 系统性分析了伪造样本生成的关键因素，并提出BBMG和NRS分别模拟篡改痕迹和抑制噪声样本。 ([论文搜索](https://scholar.google.com/scholar?q=Explore+and+Enhance+the+Generalization+of+Anomaly+DeepFake+Detection))
*   **A Hybrid Deep Learning Framework... (Zafar et al., IEEE Access 2025):** 提出了一个结合空间（EfficientNet）和时间（TempCNN）特征的轻量级高效框架，在性能和计算成本之间取得了良好平衡。 ([论文搜索](https://scholar.google.com/scholar?q=A+Hybrid+Deep+Learning+Framework+for+Highly-Efficient+Deepfake+Video+Detection))
*   **A Robust Framework... (Chandrasekaran et al., ComTech 2025):** 提出一个计算高效的轻量级CNN框架（13M参数），在DFDC数据集上达到了高准确率。 ([论文搜索](https://scholar.google.com/scholar?q=A+Robust+Framework+for+Deepfake+Detection+Using+Computationally+Efficient+Lightweight+CNN))

#### **4. 解耦伪造特征 (Decoupling Forgery Features)**
*   **Improving the Generalization Ability... (Hu et al., ICIP 2021):** 设计编码器-解码器结构，通过解耦表示学习自动分离伪造相关区域与无关背景，从而聚焦于伪造痕迹。 ([论文搜索](https://scholar.google.com/scholar?q=Improving+the+Generalization+Ability+of+Deepfake+Detection+via+Disentangled+Representation+Learning))
*   **CADDM (Dong et al., CVPR 2023):** 发现并验证了Deepfake检测中的“隐式身份泄露”问题，并设计了伪影检测模块来强制模型关注局部伪影。 ([代码](https://github.com/megvii-research/CADDM))
*   **IID (Huang et al., CVPR 2023):** 提出通过伪造人脸的“显式身份”（源人脸）和“隐式身份”（目标人脸）之间的不一致性来检测人脸交换。 ([论文搜索](https://scholar.google.com/scholar?q=Implicit+Identity+Driven+Deepfake+Face+Swapping+Detection))
*   **UCF (Yan et al., ICCV 2023):** 提出一个多任务解耦框架，将图像特征分解为内容、方法特定伪造和通用伪造三部分，仅使用通用伪造特征进行检测。 ([论文搜索](https://scholar.google.com/scholar?q=UCF:+Uncovering+Common+Features+for+Generalizable+Deepfake+Detection))
*   **LSDA (Yan et al., CVPR 2024):** 提出在潜在空间进行数据增强，通过域内(插值困难样本)和跨域(Mixup)增强来扩大伪造空间，从而学习更泛化的决策边界。 ([论文搜索](https://scholar.google.com/scholar?q=Transcending+Forgery+Specificity+with+Latent+Space+Augmentation+for+Generalizable+Deepfake+Detection))
*   **B-Free (Guillaro et al., CVPR 2025):** 提出一种创新的“无偏见”训练方法，通过从真实图像生成配对的伪造图像，迫使模型学习生成伪影而非内容，极大地提高了泛化能力。 ([代码](https://github.com/grip-unina/B-Free))
*   **UDD (Fu et al., AAAI 2025):** 识别出检测器中的位置偏见和内容偏见，并提出在ViT潜在空间中进行token级干预，从而学习无偏见的伪造特征。 ([论文搜索](https://scholar.google.com/scholar?q=Unbiased+Deepfake+Detection+via+Token-level+Intervention))
*   **FreqDebias (Kashiani et al., CVPR 2025):** 发现并提出“频谱偏见”问题，设计Fo-Mixup频率域增强和双一致性正则化来提升检测器泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=FreqDebias:+Towards+Generalizable+Deepfake+Detection+via+Consistency-Driven+Frequency+Debiasing))
*   **IDCNet (Wang et al., IEEE TIFS 2025):** 提出将图像分解为“全局内容”和“局部细节”两个视图，并通过跨视图蒸馏增强了对局部伪造的敏感度。 ([代码](https://github.com/wangzhiyuan120/idcnet))
*   **Advancing Generalization... (Son & Kim, IEEE Access 2025):** 提出一种基于监督对比学习的双流时空特征模型，显著提升了跨域泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Advancing+Generalization+in+Deepfake+Detection:+Supervised+Contrastive+Representation+Learning+With+Dual+Stream+Spatio-Temporal+Features))
*   **Enhancing Generalization... (Usmani et al., ICVGIP 2024):** 提出一种结合持续学习和ViT的Deepfake检测模型(CLEViT)，以适应不断出现的伪造技术。 ([论文搜索](https://scholar.google.com/scholar?q=Enhancing+Generalization+Ability+in+Deepfake+Detection+via+Continual+Learning))

#### **5. 基于基础模型的方法 (Methods using Foundation Models)**
*   **GASE-Net (Li et al., IEEE SPL 2022):** 将GAN检测问题建模为伪影相似度估计问题，并使用关系网络（Relation Network）进行检测，提升了对未见GAN的泛化性。 ([论文搜索](https://scholar.google.com/scholar?q=Detection+of+GAN-Generated+Images+by+Estimating+Artifact+Similarity))
*   **UniFD (Ojha et al., CVPR 2023):** 提出使用大型预训练模型（CLIP）的冻结特征空间，配合简单的分类器，能实现更好的跨生成族泛化。 ([代码](https://github.com/Yuheng-Li/UniversalFakeDetect))
*   **DE-FAKE (Sha et al., ACM SIGSAC 2023):** 首次对文生图模型进行系统的检测与溯源研究, 并提出了结合图像和文本prompt的混合检测器。 ([代码](https://github.com/zeyangsha/De-Fake))
*   **Robust DeepFake Detection Method... (Ha et al., SAC '23):** 提出CNN和ViT的集成方法，结合了CNN的高精度和ViT对噪声和新数据集的高泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Robust+DeepFake+Detection+Method+using+CNN+and+Vision+Transformer+Ensemble))
*   **Universal Detection and Source Attribution... (Das et al., PReMI 2023):** 提出一个基于ResNet-50的通用检测器，证明了在扩散模型上训练的模型能有效泛化到多种GAN生成的图像。 ([论文搜索](https://scholar.google.com/scholar?q=Universal+Detection+and+Source+Attribution+of+AI-Generated+Images))
*   **FatFormer (Liu et al., CVPR 2024):** 提出FatFormer，证明了对预训练模型（CLIP）进行伪造感知自适应（图像+频率域）和语言引导对齐，比冻结范式具有更好的泛化性。 ([论文搜索](https://scholar.google.com/scholar?q=Forgery-aware+Adaptive+Transformer+for+Generalizable+Synthetic+Image+Detection))
*   **C2P-CLIP (Tan et al., arXiv 2024):** 提出C2P-CLIP, 通过解码CLIP特征发现其通过匹配“概念”进行检测, 进而通过注入“类别共同提示”来增强CLIP图像编码器的检测能力。 ([代码](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection))
*   **Cozzolino et al. (CVPRW 2024):** 证明了基于CLIP的轻量级检测器, 仅需少量样本训练, 就能实现极强的泛化能力和鲁棒性。 ([项目主页](https://grip-unina.github.io/ClipBased-SyntheticImageDetection/))
*   **SIDA (Huang et al., arXiv 2025):** 提出SIDA框架，利用大语言模型实现对社交媒体图像的检测、定位和解释，并构建了首个大规模社交媒体伪造数据集SID-Set。 ([项目主页](https://hzlsaber.github.io/projects/SIDA/))
*   **Effort (Yan et al., ICML 2025):** 通过SVD将VFM特征正交分解为“保留预训练知识”和“学习伪造模式”的子空间，维持特征空间高秩以提升泛化性。 ([代码](https://github.com/YZY-stack/Effort-AIGI-Detection))
*   **LASTED (Wu et al., arXiv 2025):** 提出LASTED框架,通过精心设计的文本标签进行语言引导的对比学习,从而学习更具泛化性的视觉-语言联合取证特征。 ([代码](https://github.com/HighwayWu/LASTED))
*   **D³ (Yang et al., arXiv 2025):** 提出D³框架, 通过引入图像块打乱的“差异”信号, 促进通用伪造的学习，以平衡ID和OOD性能。 ([代码](https://github.com/BigAandSmallq/D3))
*   **MaskCLIP (Wang et al., arXiv 2025):** 定义OpenSDI挑战并构建专用数据集，提出SPM框架和MaskCLIP模型，协同CLIP和MAE进行扩散模型检测与定位。 ([代码](https://github.com/iamwangyabin/OpenSDI))
*   **Wavelet-CLIP (Baru et al., arXiv 2025):** 提出将CLIP的强泛化视觉特征与小波变换相结合，利用小波分类头处理CLIP特征，以学习更细粒度的时空伪影。 ([代码](https://github.com/lalithbharadwajbaru/wavelet-clip))
*   **Community Forensics (Park & Owens, arXiv 2025):** 创建了包含4803个生成器的超大规模数据集，证明了训练数据的生成器多样性是提升检测器泛化能力的关键。 ([代码/数据](https://jespark.net/projects/2024/community_forensics))
*   **Advanced_Detection... (Lamichhane, IEEE Access 2025):** 将Vision Transformer应用于AI合成图像检测，并通过消融研究分析了ViT不同组件对性能的影响。 ([论文搜索](https://scholar.google.com/scholar?q=Advanced+Detection+of+AI-Generated+Images+Through+Vision+Transformers))
*   **Detection of AI-Generated Images... (Saskoro et al., IEEE Access 2024):** 提出一种门控专家网络，能有效结合多个针对特定生成器的专家模型，提升了对多源AIGC图像的检测泛化能力和数据效率。 ([论文搜索](https://scholar.google.com/scholar?q=Detection+of+AI-Generated+Images+From+Various+Generators+Using+Gated+Expert+Convolutional+Neural+Network))
*   **VPE (Gupta et al., IEEE Access 2024):** 提出VPE作为FRS的预处理模块，增强系统对GAN规避攻击的鲁棒性。 ([论文搜索](https://scholar.google.com/scholar?q=Visual+Prompt+Engineering+for+Enhancing+Facial+Recognition+Systems+Robustness+Against+Evasion+Attacks))

#### **6. 多模态融合与集成架构 (Multi-modal Fusion & Ensemble Architectures)**
*   **JSTSP '20 (Mi et al.):** 提出使用自注意力机制来捕获GAN生成图像中的全局信息缺陷，该缺陷被假设源于上采样过程。 ([论文搜索](https://scholar.google.com/scholar?q=GAN-Generated+Image+Detection+With+Self-Attention+Mechanism+and+Auxiliary+Classifier))
*   **Fusing Global and Local Features (Ju et al., arXiv 2022):** 提出一个双分支网络，通过注意力机制融合全局特征和由“补丁选择模块”(PSM)自动选择的关键局部补丁特征。 ([代码](https://github.com/littlejuyan/FusingGlobalandLocal))
*   **Fine-grained detection... (Zhao et al., NCA 2023):** 提出了一个基于跨模态注意力的细粒度检测网络，通过融合RGB、频率和纹理三种模态的特征来提升检测性能。 ([论文搜索](https://scholar.google.com/scholar?q=Fine-grained+deepfake+detection+based+on+cross-modality+attention))
*   **Attention-Enhanced CNN... (Dasgupta et al., IEEE Access 2025):** 提出一种计算高效的CNN-MHSA混合架构，并提出集成学习策略，显著提高了跨数据集泛化性能。 ([论文搜索](https://scholar.google.com/scholar?q=Attention-Enhanced+CNN+for+High-Performance+Deepfake+Detection:+A+Multi-Dataset+Study))
*   **LightweightViT (Aryan et al., IEEE Access 2025):** 提出了一种轻量级ViT模型（69.7K参数），利用patch embedding和自注意力机制高效检测图像篡改。 ([论文搜索](https://scholar.google.com/scholar?q=Lightweight+End-to-End+Patch-Based+Self-Attention+Network+for+Robust+Image+Forgery+Detection))
*   **A Hybrid Model... (Le-Phan et al., SCID 2025):** 提出一个结合伪造边界、语义和通用伪影的三合一混合检测模型，并设计了两阶段训练策略以保留各组件的专长。 ([论文搜索](https://scholar.google.com/scholar?q=A+Hybrid+Model+for+Generalizable+Deepfake+Detection+via+Blending,+Semantic,+and+General+Artifacts))
*   **CVCNet (Dong et al., MMASIA 2024):** 针对序列性Deepfake检测，提出了融合CNN局部特征和ViT全局特征的CVCNet模型。 ([论文搜索](https://scholar.google.com/scholar?q=Improving+Sequential+DeepFake+Detection+with+Local+information+enhancement))
*   **Multi-domain Multi-scale... (Liu et al., EIECC 2024):** 融合空间域的多尺度特征和频率域特征，并使用PSO算法优化超参数，以提升模型在跨压缩、跨数据集场景下的泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Multi-domain+Multi-scale+DeepFake+Detection+for+Generalization))

#### **7. 服务于“证伪”的策略 (Strategies for "Forgery Trace Seeking")**
*   **Incr. Learning (Marra et al., WIFS 2019):** 将增量学习思想应用于GAN检测，使模型能在不遗忘旧知识的情况下，持续学习并分类新的GAN生成器。 ([论文搜索](https://scholar.google.com/scholar?q=Incremental+learning+for+the+detection+and+classification+of+GAN-generated+images))
*   **Data Farming (Korshunov & Marcel, TBIOM 2022):** 提出"数据农场"策略，通过混合真实人脸和背景切片来强制模型学习伪造伪影而非人脸特征。 ([代码](https://gitlab.idiap.ch/bob/bob.paper.deepfakes_generalization))
*   **General GAN-generated Image Detection... (Wang et al., arXiv 2023):** 提出在指纹域进行数据增强的新方法：通过自动编码器提取并扰动GAN指纹（残差），再加回图像以生成新的训练样本。 ([论文搜索](https://scholar.google.com/scholar?q=General+GAN-generated+Image+Detection+by+Data+Augmentation+in+Fingerprint+Domain))
*   **Adv. Augment (Stanciu & Ionescu, MAD 2024):** 提出一种数据增强框架，通过对真实图像进行循环对抗攻击，生成能骗过当前检测器的新伪造样本，以扩充训练集。 ([论文搜索](https://scholar.google.com/scholar?q=Improving+Generalization+in+Deepfake+Detection+via+Augmentation+with+Recurrent+Adversarial+Attacks))
*   **Frequency Masking (Doloriel et al., ICASSP 2024):** 首次将掩码图像建模思想用于通用伪造检测，提出在训练中对输入图像进行频率掩码作为一种有效的数据增强方法。 ([代码](https://github.com/chandlerbing65nm/FakeImageDetection.git))
*   **Improving Generalization... (Guan et al., IEEE TIFS 2024):** 提出梯度正则化项，在训练中降低模型对伪造纹理模式的敏感度，从而提升泛化能力，且推理阶段无额外开销。 ([论文搜索](https://scholar.google.com/scholar?q=Improving+Generalization+of+Deepfake+Detectors+by+Imposing+Gradient+Regularization))
*   **AFSL (Khan et al., Springer 2024):** 提出AFSL损失函数，通过优化特征相似性，显著提升了检测器对多种对抗性攻击的鲁棒性。 ([论文搜索](https://scholar.google.com/scholar?q=Adversarially+Robust+Deepfake+Detection+via+Adversarial+Feature+Similarity+Learning))
*   **SAFE (Li et al., KDD 2025):** 识别并解决了当前检测器训练范式中的两大偏见，通过简单的图像变换（裁剪、颜色抖动等）显著提升了泛化性。 ([代码](https://github.com/Ouxiang-Li/SAFE))
*   **PDA-RDD (Lu et al., ICASSP 2025):** 提出通过扰动域对齐（PIMW模块）来提升模型对压缩、噪声等多种真实世界扰动的鲁棒性。 ([论文搜索](https://scholar.google.com/scholar?q=Robust+Deepfake+Detection+via+Perturbation+Domain+Alignment))

#### **8. 攻防、溯源与可解释性 (Attack, Defense, Attribution & Interpretability)**
*   **SpectralGAN (Dong et al., CVPR 2022):** (攻击方法) 证明了基于频谱伪影的GAN图像检测器并非鲁棒，其依赖的频率伪影可通过对抗训练等方法被有效移除。 ([代码](https://www.comp.polyu.edu.hk/~csajaykr/deepdeepfake.htm))
*   **Counter-Forensic (Herur et al., ICVGIP 2024):** (攻防研究) 首次将DM自编码器重构真实图像的行为定义为一种反取证攻击，并构建了一个三分类框架来识别此类被操控的图像。 ([论文搜索](https://scholar.google.com/scholar?q=Addressing+Diffusion+Model+Based+Counter-Forensic+Image+Manipulation+for+Synthetic+Image+Detection))
*   **Adversarial Robustness... (N et al., ICICyTA 2024):** (攻防研究) 展示了Deepfake检测模型在对抗攻击下的极端脆弱性，并验证了对抗训练等防御策略的有效性。 ([论文搜索](https://scholar.google.com/scholar?q=Adversarial+Robustness+in+DeepFake+Detection:+Enhancing+Model+Resilience+with+Defensive+Strategies))
*   **TAG-WM (Chen et al., arXiv 2025):** (主动水印) 提出一种篡改感知的生成式水印方法，利用扩散逆向敏感性定位篡改区域，并指导版权信息解码。 ([代码](https://github.com/Suchenl/TAG-WM))

---

### **范式二：真实分布学习范式 (Authentic Distribution Learning Paradigm)**
**核心逻辑：“存真”。** 致力于学习真实图像的内在分布和物理规律，将任何偏离该分布的样本视为异常或伪造。

#### **1. 基于单类分类与异常检测 (One-Class Classification & Anomaly Detection)**
*   **RECCE (Cao et al., CVPR 2022):** 提出RECCE框架，通过仅在真实人脸上进行重建学习来建模正样本分布，从而检测伪造。 ([论文搜索](https://scholar.google.com/scholar?q=End-to-End+Reconstruction-Classification+Learning+for+Face+Forgery+Detection))
*   **SeeABLE (Larue et al., ICCV 2023):** 将Deepfake检测构建为单类异常检测任务，通过学习定位人工植入的“软差异”来训练模型，不依赖伪造样本。 ([代码](https://github.com/anonymous-author-sub/seeable))
*   **Detecting Generated Images... (Bi et al., arXiv 2023):** 提出仅使用真实图像训练的“真实分布学习”新范式，通过单类分类检测各类生成图像，泛化能力强。 ([论文搜索](https://scholar.google.com/scholar?q=Detecting+Generated+Images+by+Real+Images+Only))
*   **QC Sampling (Dogoulis et al., MAD 2023):** 提出一种基于质量分数的训练样本采样策略，证明了使用更高质量的伪造图像进行训练能提升模型在跨概念场景下的泛化能力。 ([代码](https://github.com/dogoulis/qc-sgid))
*   **MCS-GAN (Xiao et al., IEEE TMM 2024):** 将Deepfake检测视为异常检测问题，仅用真实人脸训练GAN模型学习正常分布，通过重构误差或隐空间差异检测伪造。 ([论文搜索](https://scholar.google.com/scholar?q=MCS-GAN:+A+Different+Understanding+for+Generalization+of+Deep+Forgery+Detection))
*   **Stay-Positive (Rajan et al., ICML 2025):** 通过约束分类器最后一层权重为非负，强制模型只关注伪造特征，忽略真实图像特征，减少虚假关联。 ([项目主页](https://anisundar18.github.io/Stay-Positive/))
*   **SIGMA-DF (Han et al., ICMR 2023):** 提出新颖的集成元学习框架SIGMA-DF，通过模拟多重跨域场景和挖掘难例样本，显著提升了Deepfake检测的泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=SIGMA-DF:+Single-Side+Guided+Meta-Learning+for+Generalizable+Deepfake+Detection))

#### **2. 基于生成模型重建/反演 (Generative Model Reconstruction/Inversion)**
*   **Beyond the Spectrum (He et al., arXiv 2021):** 提出基于重合成的伪影检测框架, 通过比较真实/伪造图像与重合成图像的残差进行检测。 ([代码](https://github.com/SSAW14/BeyondtheSpectrum))
*   **DIRE (Wang et al., arXiv 2023):** 提出DIRE，一种基于扩散模型重建误差的通用表示方法，用于检测扩散模型生成图像，并构建了大规模基准库DiffusionForensics。 ([代码](https://github.com/ZhendongWang6/DIRE))
*   **SeDID (Ma et al., ICML Workshop 2023):** 利用扩散模型前向和后向过程中的 stepwise error 作为区分真实与生成图像的特征。 ([论文搜索](https://scholar.google.com/scholar?q=Exposing+the+Fake:+Effective+Diffusion-Generated+Images+Detection))
*   **AEROBLADE (Ricker et al., CVPR 2024):** 提出AEROBLADE，揭示了LDM的AE对生成图像的重建误差远低于真实图像，并利用此特性进行无需训练的检测和定位。 ([代码](https://github.com/jonasricker/aeroblade))
*   **Implicit Detector (Wang & Kalogeiton, ECCVW 2024):** 提出一种新颖的检测方法，利用预训练的扩散模型作为特征提取器，通过分析模型对噪声的响应来检测伪造，具有很强的扰动鲁棒性。 ([项目主页](https://www.lix.polytechnique.fr/vista/projects/2024_detector_wang))
*   **FakeInversion (Cazenavette et al., Preprint 2024):** 提出了FakeInversion检测器，利用从固定预训练的Stable Diffusion模型中反演出的特征（噪声图和重建图）来泛化检测未见过的生成器。 ([项目主页](https://fake-inversion.github.io))
*   **DistilDIRE (Lim et al., Preprint 2024):** 通过知识蒸馏，创建了一个轻量、快速的DIRE版本，显著降低了计算需求。 ([代码](https://github.com/miraflow/DistilDIRE))
*   **DRCT (Chen et al., ICML 2024):** 提出通用训练框架DRCT，通过扩散模型重建真实图像生成高质量难样本，并结合对比学习，显著提升检测器的泛化能力。 ([代码](https://github.com/beibuwandeluori/DRCT))
*   **FIRE (Chu et al., CVPR 2024):** 首次将频率分解思想融入基于重建的检测方法，通过端到端学习，利用DM难以重建真实图像中频信息的先验进行检测。 ([论文搜索](https://scholar.google.com/scholar?q=FIRE:+Robust+Detection+of+Diffusion-Generated+Images+via+Frequency-Guided+Reconstruction+Error))
*   **STRE (Shen et al., ICASSP 2025):** 依赖“由扩散模型生成的图像比真实图像更容易被任何扩散模型重建”的先验，利用整个时间序列的重建误差（TRE）作为特征进行检测。 ([论文搜索](https://scholar.google.com/scholar?q=Spatial-Temporal+Reconstruction+Error+for+AIGC-based+Forgery+Image+Detection))
*   **SemGIR (Anonymous Authors, ACM MM 2024):** 提出SemGIR，通过语义引导的图像再生成来解耦内容和模型指纹，适用于少样本检测与溯源。 ([论文搜索](https://scholar.google.com/scholar?q=SemGIR:+Semantic-Guided+Image+Regeneration+for+Few-Shot+AI-Generated+Image+Detection))
*   **LaRE² (Luo et al., arXiv 2025):** 提出LaRE(隐空间重建误差)作为高效的伪造特征,并在EGRE模块中引导图像特征进行空间和通道维度的优化。 ([论文搜索](https://scholar.google.com/scholar?q=LaRE²:+Latent+Reconstruction+Error+Based+Method+for+Diffusion-Generated+Image+Detection))
*   **Enhancing the Generalization... (Javaheri et al., MVIP 2024):** 证明了深层检测模型（如DIRE中使用的ResNet-50）可能因过拟合而丧失泛化能力，转而使用更浅的网络作为特征提取器，能更好地学习通用伪影。 ([论文搜索](https://scholar.google.com/scholar?q=Enhancing+the+Generalization+of+Synthetic+Image+Detection+Models+through+the+Exploration+of+Features+in+Deep+Detection+Models))

#### **3. 基于内在统计、物理与一致性先验 (Intrinsic Statistical, Physical & Consistency Priors)**
*   **IID (Huang et al., CVPR 2023):** 提出通过伪造人脸的“显式身份”（源人脸）和“隐式身份”（目标人脸）之间的不一致性来检测人脸交换。 ([论文搜索](https://scholar.google.com/scholar?q=Implicit+Identity+Driven+Deepfake+Face+Swapping+Detection))
*   **Shadows Don't Lie (Sarkar et al., arXiv 2024):** 证明了生成模型在投影几何（光影、透视）上存在系统性缺陷，并以此作为独立于像素信号的检测依据。 ([项目主页](https://projective-geometry.github.io/))
*   **Bi-LIG (Jiang et al., IEEE TIFS 2024):** 提出双层不一致性生成器(Bi-LIG)，通过混合真实和伪造样本来合成训练数据，使模型同时学习“外在不一致性”和“内在不一致性”。 ([论文搜索](https://scholar.google.com/scholar?q=Exploring+Bi-Level+Inconsistency+via+Blended+Images+for+Generalizable+Face+Forgery+Detection))
*   **RIGID (He et al., arXiv 2024):** 提出无训练方法RIGID，依赖真实图像对噪声扰动更鲁棒的先验知识。 ([论文搜索](https://scholar.google.com/scholar?q=RIGID:+A+Simple+and+Efficient+Method+for+AI-generated+Image+Detection))
*   **GrDT (Xie et al., WACVW 2025):** 结合面部关键点的几何分布（图注意力网络处理）和局部纹理特征（灰度共生矩阵GLCM提取），提出一个双分支模型。 ([代码](https://github.com/SIPLab24/GrDT))
*   **CoD (Jia et al., CVPR 2025):** 提出基于色彩分布不均匀性的检测特征CoD，并构建了跨域检测基准FakeART。 ([论文搜索](https://scholar.google.com/scholar?q=Secret+Lies+in+Color:+Enhancing+AI-Generated+Images+Detection+with+Color+Distribution))
*   **SPAI (Karageorgiou et al., CVPR 2025):** 提出将真实图像的频谱分布作为不变性先验，通过掩码频谱学习进行建模，实现对任意分辨率AIGC图像的检测。 ([项目主页](https://mever-team.github.io/spai))
*   **DinoHash (Singhi et al., arXiv 2025):** 提出鲁棒的感知哈希DinoHash，其核心依赖于对真实图像分布的稳定哈希，可用于来源追溯。 ([项目主页](https://proteus.photos))
*   **Forensic Self-Descriptions (Nguyen et al., arXiv 2025):** 提出“法医自描述”概念，仅需真实图像训练，通过对预测滤波器的残差进行建模，实现零样本检测和开集溯源。 ([论文搜索](https://scholar.google.com/scholar?q=Forensic+Self-Descriptions+Are+All+You+Need+for+Zero-Shot+Detection))
*   **MIFAE-Forensics (Wang et al., ICASSP 2025):** 将Deepfake检测视为OOD问题，通过在空间域和频率域上进行掩码自编码预训练，学习真实人脸的内在分布。 ([代码](https://github.com/Mark-Dou/Forensics))
*   **Beyond Generation (Zhong et al., CVPR 2025):** 提出一种基于扩散模型的低层特征提取器，通过自监督预训练任务学习真实图像的内在特征分布。 ([论文搜索](https://scholar.google.com/scholar?q=Beyond+Generation:+A+Diffusion-based+Low-level+Feature+Extractor+for+Detecting+AI-generated+Images))
*   **AI-Synthesized Image Detection... (Manisha et al., IEEE Access 2025):** 将用于源相机识别的鲁棒全局指纹技术成功应用于AI合成图像检测，对后处理操作具有极强的鲁棒性。 ([论文搜索](https://scholar.google.com/scholar?q=AI-Synthesized+Image+Detection:+Source+Camera+Fingerprinting+to+Discern+the+Authenticity+of+Digital+Images))
*   **ID³ (Yin et al., IEEE TMM 2024):** 将不变风险最小化（IRM）范式引入Deepfake检测，强制模型学习跨域不变的伪造特征，显著提升了泛化性。 ([代码](https://github.com/Yzx835/InvariantDomainorientedDeepfakeDetection))
*   **PiD (Fu et al., PMLR 2025):** 提出一种高效、无需生成器的像素级分解残差方法（PiD），通过简单的像素级变换和量化提取高度泛化的伪造痕迹。 ([论文搜索](https://scholar.google.com/scholar?q=PiD:+Generalized+AI-Generated+Images+Detection+with+Pixelwise+Decomposition))

#### **4. 服务于“存真”的策略与架构 (Strategies & Architectures for "Authentic Distribution Learning")**
*   **DABN (Yin et al., ADVM 2021):** 提出DABN，一种测试时自适应策略，通过使用测试批次的统计数据更新BN层，缓解域偏移问题。 ([论文搜索](https://scholar.google.com/scholar?q=Improving+Generalization+of+Deepfake+Detection+with+Domain+Adaptive+Batch+Normalization))
*   **Combating Dataset Misalignment (Choi et al., WDC 2025):** 证明了数据集未对齐是导致模型在真实世界中表现不佳的关键原因，并表明在对齐的数据集上训练能显著提升鲁棒性和泛化性。 ([论文搜索](https://scholar.google.com/scholar?q=Combating+Dataset+Misalignment+for+Robust+AI-Generated+Image+Detection+in+the+Real+World))
*   **LAID (Chivaran & Ni, arXiv 2025):** 首次提出轻量级AIGC检测的Benchmark(LAID)，证明了轻量化模型在保持高精度的同时能大幅降低计算和内存成本。 ([代码](https://github.com/nchivar/LAID))
*   **HIDD (Ma et al., ICME 2024):** 提出一种以人类感知为中心的增量学习框架，利用人类标注的显著性图来指导模型关注关键伪影，并结合知识蒸馏和样本回放策略。 ([论文搜索](https://scholar.google.com/scholar?q=HIDD:+Human-perception-centric+Incremental+Deepfake+Detection))
*   **MEViT (Tran et al., IEEE OJCS 2025):** 结合元学习框架和ViT，模拟域迁移来提升对未见伪造类型的泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=MEViT:+Generalization+of+Deepfake+Detection+With+Meta-Learning+EfficientNet+Vision+Transformer))
*   **Cross-Domain Deepfake Detection... (Wang et al., IEEE SPL 2025):** 提出一个无需访问源域数据的跨域Deepfake检测框架(LDKD)，通过潜在域生成和知识蒸馏，实现了SOTA的泛化性能。 ([论文搜索](https://scholar.google.com/scholar?q=Cross-Domain+Deepfake+Detection+Based+on+Latent+Domain+Knowledge+Distillation))
*   **Art or Artifact? (Zheng et al., WDC 2025):** 提出一个用于AI生成艺术品检测的多任务学习框架，证明了增加像素级分割监督可以同时提升模型在分布内准确率和分布外（OOD）泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Art+or+Artifact?+Segmenting+AI-Generated+Images+for+Deeper+Detection))
*   **Faster Than Lies (BNN-based) (Lanzino et al., Preprint 2024):** 首次将二值神经网络（BNN）应用于Deepfake检测，结合FFT和LBP特征，实现了高达20倍的FLOPs降低。 ([代码](https://github.com/fedeloper/binary_deepfake_detection))

---

### **纯粹的综述、基准与分析研究**
这些论文不提出新的检测方法，而是对领域进行总结、评估、提供工具或进行专门的分析。

*   **Discovering Transferable Forensic Features... (Chandrasegaran et al., arXiv 2022):** 首次提出FF-RS方法来发现和量化CNN检测器中的可迁移法证特征（T-FF），并揭示了**颜色**是一个被忽视但至关重要的T-FF。 ([项目主页](https://keshik6.github.io/transferable-forensic-features/))
*   **Towards Generalization... (Verdoliva, IH&MMSec 2022):** Keynote演讲摘要，强调了Deepfake检测领域中域泛化（domain generalization）的重要性。 ([论文搜索](https://scholar.google.com/scholar?q=Towards+Generalization+in+Deepfake+Detection))
*   **DE-FAKE (Sha et al., ACM SIGSAC 2023):** 首次对文生图模型进行系统的检测与溯源研究, 并提出了结合图像和文本prompt的混合检测器。 ([代码](https://github.com/zeyangsha/De-Fake))
*   **Online Detector (Epstein et al., ICCVW 2023):** 在模拟模型发布顺序的“在线”设置中研究AIGC检测，并证明使用CutMix可在无像素级标注时训练inpainting检测器。 ([项目主页](https://richzhang.github.io/OnlineGenAIDetection/))
*   **Deepfake_Detection_Analyzing_Model_Generalization... (Khan & Dang-Nguyen, IEEE Access 2023):** 首次全面对比了CNN和Transformer在Deepfake检测泛化性上的表现，并揭示了更具挑战性的数据集能带来更好的泛化能力。 ([论文搜索](https://scholar.google.com/scholar?q=Deepfake+Detection:+Analyzing+Model+Generalization+Across+Architectures,+Datasets,+and+Pre-Training+Paradigms))
*   **SIDBench (Schinas & Papadopoulos, MAD 2024):** 提出了一个模块化的基准测试框架（SIDBench），用于标准化和可靠地评估合成图像检测（SID）方法。 ([代码](https://github.com/mever-team/sidbench))
*   **Solution to ACMMM 2024... (Li et al., MM 2024):** 提出了一个有效的竞赛方案，通过数据均衡策略、NPR特征提取和强主干网络集成，取得了优异的泛化表现。 ([论文搜索](https://scholar.google.com/scholar?q=A+Solution+to+ACMMM+2024+on+Artificial+Intelligence+Generated+Image+Detection))
*   **Optimizing AIGC... (Fu, MM 2024):** 在一个AIGC检测竞赛中，成功应用邻域像素关系(NPR)模块与ResNet-50结合，并通过特定的数据增强和训练策略取得了第一名的成绩。 ([论文搜索](https://scholar.google.com/scholar?q=Optimizing+AIGC+Image+Detection:+Strategies+in+Data+Augmentation+and+Model+Architecture))

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
| **DFDC** | Kaggle竞赛数据集，规模巨大，包含多种混淆和攻击 | [Link](https://ai.facebook.com/datasets/dfdc/) |
| **Community Forensics** | 包含4800+生成器和270万张图像的超大规模数据集 | [Link](https://jespark.net/projects/2024/community_forensics) |
| **GenImage** | 包含多种GAN和Diffusion模型的通用AIGC图像数据集 | [Link](https://github.com/GenImage-Dataset/GenImage) |
| **SIDBench** | 用于可靠评估合成图像检测方法的Python框架及相关数据集 | [Link](https://github.com/mever-team/sidbench) |
| **Synthbuster Dataset** | 包含多种扩散模型生成图像的数据集 | [Link](https://zenodo.org/records/10066460) |

---

## 方法对比 (Benchmark)

以下是一些代表性方法在主流数据集上的公开评测结果。欢迎提交您的结果！

| 方法名称 | 主干网络 | 数据集 | 评价指标 (ACC % / AUC %) | 对应论文 | 代码链接 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| XceptionNet | Xception | FF++ (c23) | 99.65 / 99.8 | *CNNDetection* | [Link](https://github.com/peterwang512/CNNDetection) |
| UniFD | CLIP ViT-L/14 | ProGAN(T) -> Diffusion(Test) | 81.38 (Acc) | *UniFD* | [Link](https://github.com/Yuheng-Li/UniversalFakeDetect) |
| DIRE | ResNet-50 | ADM(T) -> 7 DMs(Test) | 99.9 (Acc) | *DIRE* | [Link](https://github.com/ZhendongWang6/DIRE) |
| B-Free | DINOv2+reg | 多个未见模型/真实世界数据集 | 96.3 (bAcc) | *B-Free* | [Link](https://github.com/grip-unina/B-Free) |
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
