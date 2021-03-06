<a name="contents"></a>
# contents
- [incremental SfM](#incremental_SfM)
- [global SfM](#global_SfM)
- [hierarchica SfM](#hierarchical_SfM)
- [multi-stage SfM](#multi-stage_SfM)
- [graph-based](#graph-based)
- [factor graph](#factor-graph)
- [depth](#depth)
- [feature](#feature)
- [outlier](#outlier)
- [bundle adjustment](#bundle-adjustment)
- [localization](#localization)
- [calibration](#calibration)
- [motion](#motion)
- [non-rigid](#non-rigid)
- [distortion](#distortion)
- [parallel](#parallel)
- [disambiguation](#disambiguation)
- [camera model](#camera-model)
- [segmentation](#segmentation)
- [fundamental matrix](#fundamental-matrix)
- [factorization](#factorization)
- [optimization](#optimization)
- [mesh](#mesh)  
- [stereo](#stereo)  
- [privacy](#privacy)
- [tips](#tips)
- [Multi-View Stereo](#Multi-View_Stereo)
- [SLAM](#SLAM)
- [robot grasp](#robot_grasp)



# keywords

> **SfM** | **三维重建** | **graph-based** | **factor-graph** | **深度估计** | **feature keypoint** | **feature dectect** | **feature description** | **AR** | **feature descriptor** | **feature tracking** | **feature matching** | **outlier** | **bundle adjustment** | **localization** | **motion** | **non-rigid** | **distortion** | **parallel** | **camera model** | **fisheye** | **segmentation** | **fundamental matrix** | **factorization** | **optimization** | **mesh** | **stereo** | **RGB-D** | **dehazing** | **Multi-View Stereo** | **point cloud** | **surface** | **单目** | **多目** | **TSDF** | **dense map** | **semi-dense map** | **增大视野** | **visual odometry** | **visual-inertial odometry** | **LiDAR** | **3D结构光** | **直接法** | **特征点法** | **半直接法** | **DNN** | **uncertainty estimation** | **pose** | **GAN** |



# papers

<a name="incremental_SfM"></a>
## incremental SfM

["Structure-from-Motion Revisited"](https://demuc.de/papers/schoenberger2016sfm.pdf), Johannes L.Schonberger, Jan-Michael Frahm, 2016 CVPR



["Structure-from-Motion Revisited"](https://demuc.de/papers/schoenberger2016sfm.pdf) 2016 CVPR

> 增量式SfM用于三维重建
>
> **SfM** | **三维重建** |

+ scene graph augmentation
    + 估计基本矩阵F，当$inliers >= N_F$, image pair 几何匹配
    + 根据决定H的inliers数目NH划分H
        
        GRIC法
        
        如果 $N_H / N_F < \epsilon_{HF}$, camera 在同一个场景移动
        
    + 对于标定好的相机
    
        E 本质矩阵，$N_E / N_F > \epsilon_{EF}$, 说明标定正确
        
        分解E、三角测量、median triangulation angle $\alpha_m$, 可区分pure rotation、planar scenes
        
    + 辨别 WTF（watermarks、timestamps frames）不加入 scene graph
    + valid pairs
    
+ nest best view selection
    
    目标：minimize reconstruction error
    
    使用 uncertainty-driven 方法
    
    PnP 相机姿态估计的精度，取决于观察点的数量、及其在images上的分布；
    
    没有（有误）相机标定下，估计内参
    
    充分地评估会比较耗时、所以近似评估
    
    数目越多、分布越均匀 -> score 越高
    
    按固定大小网格离散化图片，统计score
    
+ robust and efficient triangulation
    
    cheriality constraint
    
+ BA

    global BA
    
    柯西方程 -> loss function
    
    solver 分为 sparse direct solver 和 PCG ceres solver
    
    radial 畸变参数 -> pure self-calibration
    
+ Re-trangulation

    pre-BA RT，post-BA RT
    
    不增加三角 threshold，而是 track （error 低于 filter阈值）
    
+ Redundant View Mining

    problem -> submaps
    
    camera grouping
    
    images 和 points 分为 2 个 set：affected 和 unaffected
    
["Towards Linear-time Incremental Structure from Motion"](http://ccwu.me/vsfm/vsfm.pdf) 2013 3DV

> 增量式
>
> **SfM** | **三维重建** |

+ preemptive feature matching
+ preconditioned conjugate gradient
+ reduce scene graph


<a name="global_SfM"></a>
## global SfM

["Global Structure-from-Motion by Similarity Averaging"](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cui_Global_Structure-From-Motion_by_ICCV_2015_paper.pdf) 2015 ICCV

> 已知：images，由5点算法计算的E，EG graph；求：朝向R和相机中心c
>
> **SfM** | **三维重建** |

![globalSfM](https://github.com/Brook1886/SfM-Notebook/blob/main/image/2015%20Global%20Structure-from-Motion%20by%20Similarity%20Averaging.png)

+ EG graph

+ depth images

+ optional：1）local BA；2）missing correspondence分析

+ 相似度平均
    
    register cameras
    
    rotation averaging -> 相机朝向
    
    scale averaging -> global scale of depth images
    
    scale-aware translation averaging -> basline length, camera postion
    
+ 多视图三角测量

+ final BA：camera、3D points
    
["Robust Camera Location Estimation by Convex Programming"](https://web.math.princeton.edu/~amits/publications/CVPR2015-SfM-Published.pdf) 2015 IEEE

> Location Estimation
>
> **SfM** | **三维重建** |

+ 估计 camera location 和 orientation

+ LUD least unsquared deviations

["Robust Global Translations with 1DSfM"](http://www.cs.cornell.edu/projects/1dsfm/docs/1DSfM_ECCV14.pdf) 2014 ECCV

> 1DSfM
>
> **SfM** | **三维重建** |

+ 使用 L1-IELS 算法，根据 relative rotation 估计 global rotation

> BA 效果与 init 强相关

+ outliers removal，全局SfM主要挑战：outliers，求global camera rotations，再求 translations

    投影到 1-dim 上，根据outliers在某方向满足顺序约束，某方向不满足来筛选outliers
    
    minimum feedback arc set（MFAS）算法，最小反馈边集合
    
+ slove global translation

    设置目标函数，使用弦距离（chordal distance）
   
["Efficient and Robust Large-Scale Rotation Averaging"](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Chatterjee_Efficient_and_Robust_2013_ICCV_paper.pdf) 2013 ICCV

> Rotation Averaging
>
> **SfM** | **三维重建** |

+ Lie-Algebraic Relative Rotation Averaging 算法

    robust rotation averaging 看作回归问题或 M-estimator
    
+ Iteratively Reweighted Least Squares（IRLS）算法

+ L1-IRLS 算法


<a name="hierarchical_SfM"></a>
## hierarchical SfM

["Hierarchical structure-and-motion recovery from uncalibrated images"](https://arxiv.org/pdf/1506.00395) 2015 Jun

> dubbed SAMANTHA
>
> **SfM** | **三维重建** |


<a name="multi-stage_SfM"></a>
## multi-stage SfM

["HSfM: Hybrid Structure-from-Motion"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_HSfM_Hybrid_Structure-from-Motion_CVPR_2017_paper.pdf) 2017 ICCV

> incremental 鲁棒、精度高，但是效率较低；global 对outliers敏感；accuracy + robustness + efficiency 都要？
>
> **SfM** | **三维重建** |

![pipeline](https://github.com/Brook1886/SfM-Notebook/blob/main/image/2017%20HSfM%20Hybrid%20Structure-from-Motion.png)

+ an adaptive community-based rotation averaging -> camera rotations

    + Community
    
        Community detection -> divide a graph (epipolar geometry graph, EG graph) into groups with denser connections inside and sparser connections outside
    
        定义一个 modularity indicator Q，尝试合并两个节点，如果 Q 增加大于一定阈值（0.4）的分为两个 community
        
    + rotation averaging
        
        使用 "Efficient and robust largescale rotation averaging" 方法分别对每个 community 进行 rotation averaging，每个 community 有不同坐标系
        
        a voting scheme （transformation 具有最多的 inliers） -> 合并每个 commutity 的结果
        
        community 之间连接边加权，weight是最好的 transformation 的 inliers 占比
        
        community 构成的 graph -> construct maximal spanning tree (MST)
    
+ camera rotations -> camera centers (incremental way)

    + init，决定 incremental 起始位置
        
        5-point algorithm -> the inlier number of feature matchings; angles 表征 length of baseline; neighbors of the camera 表征 camera poses accuracy
        
        recalculate its relative rotations Rij ->  relative translation tij, 使用 RANSAC
        
    + Camera Registration
    
        RANSAC -> find best camera center (has the largest number of visible scene points inliers)
        
        visible scene points 多，但camera center误差还大，说明 camera rotations 就估计的不准，使用 P3P 直接计算一个 camera pose
        
    + 基于 RANSAC 的 triangulation
        
        每轮迭代，随机选两个views，如果angle between two projection rays 小于3度，直接用DLT；检查 cheirality；重投影误差小，接受
        
+ 其他技巧：Huber function；Re-Triangulation + BA


## 基于学习 SfM

["Supervising the new with the old: learning SFM from SFM"]() 2018

> 
>
> **SfM** | **DNN** |

<a name="graph-based"></a>
## graph-based

["NeuRoRA: Neural Robust Rotation Averaging"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690137.pdf) 2020 ECCV 

> robust cost functions 非线性、且基于噪声和outliers的分布假设
>
> **SfM** | **三维重建** | **graph-based** |

+ view-graph cleaning network（noise、outliers）+ fine-tune network
+ 用合成graphs训练

1. graph-based network可以用于其他graph-based geometric problems，像pose-graph optimization

["Graph-Based Parallel Large Scale Structure from Motion"](https://arxiv.org/pdf/1912.10659v2.pdf) 2019 Dec

> 将大规模SfM问题，看作graph问题；
> 特征匹配后，使用 cluster，并行体现在不同聚类可以同时 SfM
> 
> **SfM** | **三维重建** | **graph-based** |

+ 特征提取、匹配，filter outliers，images cluster
    
    graph cut，weak connection
    
    最大生成树expand图片，增加连接性
    
+ local reconstruction（incremental SfM）in parallel，merge local map，re-triangulation，bundle adjestment

+ 最小生成树获取精确的相似变换

+ 最小高度树找到合适的anchor node，减少误差累计

["GraphMatch: Efficient Large-Scale Graph Construction for Structure from Motion"](https://arxiv.org/pdf/1710.01602) 2017 Oct

> 对比 vocabulary trees 方法，如 BRIAD
> 
> **SfM** | **三维重建** | **graph-based** |

+ 使用 fisher distance
+ sample-and-propagate 机制

["Graph-Based Consistent Matching for Structure-from-Motion"](http://www.eccv2016.org/files/posters/P-2A-19.pdf) 2016 ECCV

> unordered images
> 
> **SfM** | **三维重建** | **graph-based** |

+ visual- similarity-based minimum spanning tree 最小生成树

+ global community-based graph 算法


<a name="factor-graph"></a>
## factor graph

["miniSAM: A Flexible Factor Graph Non-linear Least Squares Optimization Framework"](https://arxiv.org/pdf/1909.00903v1.pdf) 2019 Sep

> factor graphs；非线性最小二乘优化问题
> 
> **factor-graph** |

+ an open-source C++/Python framework
+ a wide list of sparse linear solvers


<a name="depth"></a>
## depth

["SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments"](https://arxiv.org/pdf/2011.04408v1.pdf) 2020 Nov

> 季节、光照变换，缺乏数据集和基准
> 
> **深度估计** |

["RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty"](https://arxiv.org/pdf/2011.10359v1.pdf) 2020 Nov

> 同时估计dense depth map和camera poses；indoor
> 
> **深度估计** |

传统分两步：1）相机估计；2）MVS

+ deep net -> "depth-planes" 线性组合 -> depth map
+ altered BA -> poses、dense reconstructions
+ high-quality sparse keypoint matches -> 优化：1）前一帧 "depth-planes" 线性组合；2）相机姿态

["DeepV2D: Video to Depth with Differentiable Structure from Motion"](https://openreview.net/pdf?id=HJeO7RNKPr) 2020 ICLR

> 输出 motion 和 depth
> 
> **深度估计** |

["Consistent Video Depth Estimation"](https://arxiv.org/pdf/2004.15021.pdf) 2020 Aug

> a monocular video -> depth map
> 
> **深度估计** |

+ 传统 SfM -> video 里的几何约束

    ad-hoc 先验改为学习先验
    
    用 CNN 输出 single-image depth estimation
    
+ fine-tune 网络 满足几何约束

["Self-Supervised 3D Keypoint Learning for Ego-motion Estimation"](https://arxiv.org/pdf/1912.03426v3.pdf) 2019 Dec

> detect and match viewpoint-invariant keypoint；基于学习的方法：keypoint 经单应到synthetic views，该方法对非共面且有光照变化的场景不泛化
> 
> **深度估计** |

+ video -> 自监督学习具有深度信息的关键点
+ 可微分SfM模型
+ 外观+几何匹配 -> 学习关键点、深度估计

["Single Image Depth Estimation Trained via Depth from Defocus Cues"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gur_Single_Image_Depth_Estimation_Trained_via_Depth_From_Defocus_Cues_CVPR_2019_paper.pdf) 2019 CVPR

> defocus cues 散焦视差
> 
> **深度估计** |

+ Point Spread Function：conv layer、散光圈（Circle-Of-Confusion）

KITTI and Make3D 数据集

["MegaDepth: Learning Single-View Depth Prediction from Internet Photos"](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.pdf) 2018 CVPR

> 基于 deep learning 的单视图深度预测： 困难，没有有效的（available）训练数据；NYU 只有室内；Make3D 种类少；KITTI sparse sample
> 
> **深度估计** |

+ 找 Internet photos -> SfM MVS -> Mega Depth (大量 depth 数据集）

+ 难点：MVS -> data，由于noise、不可重建物体存在
    
    解决：data cleaning，auto augment，如 semantic segmentation

["Geometry meets semantics for semi-supervised monocular depth estimation"](https://arxiv.org/pdf/1810.04093v2.pdf) 2018 Oct

> single image 缺乏几何信息；encoder-decoder依赖有效的特征表达
> 
> **深度估计** |

+ semantics -> 改善深度的估计
+ image warping loss
+ semi-supervised，semantic使用GT
+ cross-task loss

KITTI

["Digging into self-supervised monocular depth estimation"](https://arxiv.org/pdf/1806.01260.pdf) 2019

>  MonoDepth2
> 
> **深度估计** |

["Unsupervised monocular depth estimation with left-right consistency"]() 2016

> 
> 
> **深度估计** |

["Deeper depth prediction with fully convolutional residual networks"]() 2016

> 监督学习
> 
> **深度估计** |

["Semi-supervised deep learning for monocular depth map prediction"]() 2017

> 
> 
> **深度估计** |

["Unsupervised learning of depth and ego-motion from video"](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf) 2017

> 
> 
> **深度估计** | **pose** |

["Depth map prediction from a single image using a multi-scale deep network"]() 2014

> 监督学习 CNN
> 
> **深度估计** | 

["Depth and surface normal estimation from monocular images using regression on deep features and hierarchical CRFs"]() 2015

> 监督学习
> 
> **深度估计** |

["Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture"]() 2015

> CNN
> 
> **深度估计** |

["Depth from videos in the wild: Unsupervised monocular depth learning from unknown cameras"]() 2019

>
>
> **深度估计** |

["Unsupervised learning of depth and ego-motion from monocular video using 3d geometric constraints"]() 2018

>
>
> **深度估计** |

["Learning depth from monocular videos using direct methods"]() 2018

>
>
> **深度估计** |

["GeoNet: Unsupervised learning of dense depth, optical flow and camera pose"]() 2018

>
>
> **深度估计** |

["Unsupervised learning of monocular depth estimation and visual odometry with deep feature reconstruction"]() 2018

>
>
> **深度估计** |

["Spatial transformer networks"]() 2017

> adopting differentiable interpolation
>
> **深度估计** |

["Neural RGB→D Sensing: Depth and Uncertainty from a Video Camera"](https://arxiv.org/abs/1901.02571) 2019

> **深度估计** |

训练三个CNN，输入视频流中的相邻5帧，输出depth及其对应的置信度

## uncertainty estimation

["What uncertainties do we need in bayesian deep learning for computer vision?"]() 2017

>
>
> **uncertainty estimation** |

["Multi-task learning using uncertainty to weigh losses for scene geometry and semantics"]() 2018

>
>
> **uncertainty estimation** |



<a name="feature"></a>
## feature

["SEKD: Self-Evolving Keypoint Detection and Description"](https://arxiv.org/pdf/2006.05077v1.pdf) 2020 Jun

> 
> 
> **feature keypoint** |

+ local feature detector 与 descriptor 交互
+ 自监督，unlabeled natural images
+ training strategies

1. homography estimation, relative pose estimation, and structure-from-motion tasks

["Neural Outlier Rejection for Self-Supervised Keypoint Learning"](https://openreview.net/pdf?id=Skx82ySYPH) 2020 ICLR

> 学习keypoint的方法有效，训练数据保证兴趣点准确较为困难
> 
> **feature keypoint** |

+ IO-Net (i.e. InlierOutlierNet)
+ 自监督：keypoint检测、描述、匹配
sample inlier和outlier set

+ KeyPointNet -> keypoint检测、描述
+ spatial discretizations
+ subpixel conv，上采用descriptor fmap分辨率，改善描述

["Multi-View Optimization of Local Feature Geometry"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460647.pdf) 2020 ECCV

> local feature detect；单视图提取关键点来提取特征不准，能否多视图
> 
> **feature dectect** |

+ 拿不准的match先估计局部几何变换，利用多视图优化关键点位置，最小二乘

1. 能改善三角测量和相机定位
 
["LiFF: Light Field Features in Scale and Depth"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dansereau_LiFF_Light_Field_Features_in_Scale_and_Depth_CVPR_2019_paper.pdf) 2019 CVPR

> 解决特征检测、描述；场景中light问题：部分遮挡、低对比度、表面反射、折射
> 
> **feature dectect** |

+ 光场相机
+ 4D光场
+ 尺度不变、4D光场检测特征、对透视变换鲁棒

能提高SfM效果

["LandscapeAR: Large Scale Outdoor Augmented Reality by Matching Photographs with Terrain Models Using Learned Descriptors"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf) 2020 ECCV

> 大规模AR
> 
> **feature description** | **AR** |

+ textured Digital Elevation Models (DEMs)

+ 用 SfM 重建出训练数据，训练 a cross-domain feature descriptor

+ 可在移动设备上用

["LF-Net: Learning Local Features from Images"](http://papers.nips.cc/paper/7861-lf-net-learning-local-features-from-images.pdf) 2018 NeurIPS

> 
> 
> **feature description** |

+ 利用depth和motion在一个图片里定一个虚拟目标，网络输出用于其他图片
+ 两个分支，限制一个，让另一个可微
+ 在indoor（depth 来自3D传感器）和outdoor（depth来自SfM解的估计） datasets下训练

1. 无监督学习
2. 60fps QVGA图片（240×320）

["GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.pdf) 2018 ECCV

> 整合多视图重建时的几何约束来学习局部特征
> 
> **feature description** |

["Integration of the 3D Environment for UAV Onboard Visual Object Tracking"](https://arxiv.org/pdf/2008.02834v3.pdf) 2020 Aug

> 困难：object occlusion, small-scale objects, background clutter, and abrupt camera motion
> 
> **feature tracking** |

+ 在三维重建的场景中，检测和跟踪
+ a model-free visual object tracker, a sparse 3D reconstruction, and a state estimator
+ representing the position of the target in 3D space rather than in image space

low-altitude oblique view，image sequences

1. 比 plain visual cues 和 image-space-based state estimations 效果好

todo：SfM结合去动态物体算法

["Fast connected components computation in large graphs by vertex pruning"](http://for.unipi.it/alessandro_lulli/files/2015/07/J002_FastConnectedComponentsComputationInLargeGraphsByVertexPruning.pdf) 2016 Jul

> 一种图算法
> 
> **feature tracking** |

+ 提出 iterative Map Reduce 算法
+ CRACKER

["ENFT: Efficient Non-Consecutive Feature Tracking for Robust Structure-from-Motion"](https://arxiv.org/pdf/1510.08012v2.pdf) 2015 Oct

> 非连续特征Tracking；SfM很依赖feature tracking； 对于图片序列：对象移动、偶然遮挡、图片噪声如果处理不好，SfM效果不好；大规模重建越明显
> 
> **feature tracking** |

ENFT用于match被打断的tracking（不同图片子序列，甚至不同video）
解决无明显特征、具有噪声、图片畸变等特征跟丢问题，快速再此tracking

+ 一种基于分割的SfM

["AdaLAM: Revisiting Handcrafted Outlier Detection"](https://arxiv.org/pdf/2006.04250v1.pdf) 2020 Jun

> Local feature matching；匹配包含outliers
> 
> **feature matching** |

+ a hierarchical pipeline for effective outlier detection
+ 并行计算，fast

["Robust Line Segments Matching via Graph Convolution Networks"](https://arxiv.org/pdf/2004.04993v2.pdf) 2020 Apr

> 直线匹配；纹理较少、重复结构场景，直线匹配更为重要（对SfM，SLAM）
> 
> **feature matching** |

+ GCN -> 匹配两图中的直线
+ 直线匹配转换为图的优化问题


<a name="outlier"></a>
## outlier

["Efficient Outlier Removal in Large Scale Global Structure-from-Motion"](https://arxiv.org/pdf/1808.03041v4.pdf) 2018 Aug

> global outlier removal
> 
> **outlier** |

+ a convex relaxed l_1 minimization，可用线性规划（LP）解
+ ideal l_0 minimization，可用an iteratively reweighted method解


<a name="bundle-adjustment"></a>
## bundle adjustment

["DeepSFM: Structure From Motion Via Deep Bundle Adjustment"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460222.pdf) 2020 ECCV

> 
> 
> **bundle adjustment** |

["RPBA -- Robust Parallel Bundle Adjustment Based on Covariance Information"](https://arxiv.org/pdf/1910.08138v1.pdf) 2019 Oct

> 并行BA
> 
> **bundle adjustment** |

consensus-based optimization methods

+ adjustment 3d point -> covariance information

["BA-Net: Dense Bundle Adjustment Network"](https://arxiv.org/pdf/1806.04807v3.pdf) 2018 Jun

> dense SfM，基于feature度量的BA
> 
> **bundle adjustment** | **SfM** |

+ 多视图几何约束 -> feature-metric error
+ depth parameterization 恢复深度
+ image -> 几个basis depth maps -> 线性组合 -> final depth （via feature-metric BA）

["BAD SLAM: Bundle adjusted direct RGB-D SLAM"]() 2019

>
>
> **bundle adjustment** |


<a name="localization"></a>
## localization

["Reference Pose Generation for Long-term Visual Localization via Learned Features and View Synthesis"](https://arxiv.org/pdf/2005.05179v3.pdf) 2020 May

> 视觉定位；SfM依赖局部特征，外部环境变化易失败；手工标注特征对应可能不准确
> 
> **localization** |

+ 学习到的特征（3D模型和图片之间匹配到的特征）-> pose
+ 半自动

Aachen Day-Night dataset 有47%提升

["Cascaded Parallel Filtering for Memory-Efficient Image-Based Localization"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Cascaded_Parallel_Filtering_for_Memory-Efficient_Image-Based_Localization_ICCV_2019_paper.pdf) 2019 ICCV

> Image-based localization (IBL) ：估计 camera poses，规模大SfM如何节省memory
> 
> **localization** |

+ cascaded parallel filtering：利用feature, visibility and geometry information来filter wrong matches


<a name="calibration"></a>
## calibration

["Calibration-free Structure-from-Motion with Calibrated Radial Trifocal Tensors"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500375.pdf) 2020 ECCV

> 
> 
> **calibration** |

["Infrastructure-based Multi-Camera Calibration using Radial Projections"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610324.pdf) 2020 ECCV

> 多相机系统，已知相机内参，估计外参
> 
> **calibration** |

using 3D map

假设畸变主要为径向

+ 先初步估计每个相机的外参，再求内参和精确的外参

1. 比先估计内参和姿态、再求外参的方法鲁棒


<a name="motion"></a>
## motion

["Relative Pose from Deep Learned Depth and a Single Affine Correspondence"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf) 2020 ECCV

> 
> 
> **motion** |

+ 结合 non-metric monocular depth + affine correspondences -> 从single correspondence 估计 relative pose
+ 1-point RANSAC approaches
+ 1AC+D solver

使用 global SfM 在 1DSfM 数据集上验证

["Resultant Based Incremental Recovery of Camera Pose from Pairwise Matches"](https://arxiv.org/pdf/1901.09364v1.pdf) 2019 Jan

> 
> 
> **motion** |

+ six-point online algorithm 恢复外参，incremental 到第n张图片
+ Dixon resultant？
+ Bernstein's theorem？证明了复解个数的严格上下界

["Flow-Motion and Depth Network for Monocular Stereo and Beyond"](https://arxiv.org/pdf/1909.05452v1.pdf) 2019 Sep

> 
> 
> **motion** |

+ 2 images、intrinsic -> pose、depth map
+ network估计 光流和相机姿态
+ 三角测量层encode光流和相机姿态
+ target images的depth -> network -> 估计source image的depth
+ 提供给网络训练的数据集

["Trifocal Relative Pose from Lines at Points and its Efficient Solution"](https://arxiv.org/pdf/1903.09755v3.pdf) 2019 Mar

> relative pose estimation
> 
> **motion** |

mixed point、line correspondences、three views

+ 最小化问题：1）3points+1line；2）2points+2lines
+ a suitable homotopy continuation technique

解决2views失败的重建

["DeepVO: Towards end-to-end visual odometry with deep recurrent convolutional neural networks"]() 2017

> supervised
>
> **visual odometry** | **DNN** | **pose** |

["UnDeepVO: Monocular visual odometry through unsupervised deep learning"]() 2017

> unsupervised
>
> **visual odometry** | **DNN** | **pose** |



<a name="non-rigid"></a>
## non-rigid

["Deep NRSfM++: Towards 3D Reconstruction in the Wild"](https://arxiv.org/pdf/2001.10090v1.pdf) 2020 Jan

> 2D landmarks stemming？Classical NRSfM 方法不能 handle 大规模图集且只能处理少数 shape；当前仍然有的问题：不能 handle missing/occluded points；仅仅弱透视相机模型
> 
> **non-rigid** |

["C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion"](https://arxiv.org/pdf/1909.02533.pdf) 2019 Oct

> deformable object（2d key point in images）-> extract -> 3D models
> 
> **non-rigid** |

+ 同时考虑：1）partial occlusion；2）viewpoint change；3）object deformation

+ 正则化

+ partial occlusion，成功的先觉条件：重建shape具有确切的 canonicalization function

+ 不需要 GT 来监督

["DefSLAM: Tracking and Mapping of Deforming Scenes from Monocular Sequences"](https://arxiv.org/pdf/1908.08918v2.pdf) 2019 Aug

> 解决deforming scenes
> 
> **non-rigid** |

+ Shape-from-Template (SfT) + Non-Rigid Structure-from-Motion (NRSfM)
+ deformation tracking thread -> poses、deformation

["Deep Interpretable Non-Rigid Structure from Motion"](https://arxiv.org/pdf/1902.10840v1.pdf) 2019 Feb

> NRSfM问题困难：图片数量；shape变化的handle程度
> 
> **non-rigid** | **SfM** |

+ DNN -> camera poses、3D点（2D坐标系生成？）
+ DNN 可解释为多层稀疏字典学习问题
+ 基于权重提出一种评估方法，需要多少GT能确保所需置信度

["Structure from Recurrent Motion: From Rigidity to Recurrency"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Structure_From_Recurrent_CVPR_2018_paper.pdf) 2018 CVPR

> 解决Non-Rigid Structure-from-Motion (NRSfM)
> 
> **non-rigid** | **SfM** |

单目序列video，存在周期性变化

+ 转周期性变化为刚性变化

+ 以刚性判断来聚类视角


<a name="distortion"></a>
## distortion

["Tangent Images for Mitigating Spherical Distortion"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Eder_Tangent_Images_for_Mitigating_Spherical_Distortion_CVPR_2020_paper.pdf) 2020 CVPR

> "tangent images"
> 
> **distortion** |

球型

二十面体


<a name="parallel"></a>
## parallel

["Parallel Structure from Motion from Local Increment to Global Averaging"](https://arxiv.org/pdf/1702.08601v3.pdf) 2017 Feb

> accurate、consistent
> 
> **parallel** |

不同于以往靠简化、牺牲精度，企图用并行来加速

+ camera clusters with overlapping
+ local increment SfM -> relative poses -> global motion averaging -> accurate and consistent global camera poses
+ track generation + local SfM + 3D point triangulation + bundle adjustment

1. a city-scale data-set (one million high-resolution images)


<a name="disambiguation"></a>
## disambiguation

["Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context"](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.pdf) 2017 CVPR

> 场景里具有重复结构的重建，需disambiguation
> 
> **disambiguation** |

当前靠背景纹理来解决，不能解决不清晰场景

+ encode global topology
+ viewpoints的流形manifold -> 逼近可用图像
+ 相似的结构造成重建困惑的，实际场景中都相距较远（测地线geodesic）
+ adjacent viewpoint -> adaptively identifying cameras -> manifold
+ geodesic consistency -> detect ambiguities


<a name="camera-model"></a>
## camera model

["Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption"](https://arxiv.org/pdf/2011.03106v1.pdf) 2020 Nov

> rolling shutter会导致图像畸变， 卷帘快门的影响如何矫正
> 
> **camera model** |

+ dnn -> depth、camera poses
+ 该方法无需假设相机 motion，输入图片即可
+ TUM rolling shutter dataset

["Uncertainty Based Camera Model Selection"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Polic_Uncertainty_Based_Camera_Model_Selection_CVPR_2020_paper.pdf) 2020 CVPR

> 相机模型决定SfM重建效果
> 
> **camera model** |

+ 基于不确定性估计自动选择相机模型

+ Accuracy-based Criterion


<a name="segmentation"></a>
## segmentation

["Three-dimensional Segmentation of Trees Through a Flexible Multi-Class Graph Cut Algorithm (MCGC)"](https://arxiv.org/pdf/1903.08481v1.pdf) 2019 Mar

> individual tree crown (ITC) 单棵树冠检测问题；树的种类较多，热带
> 
> **segmentation** |

Many algorithms exist for structurally simple forests including coniferous forests and plantations.

+ local three-dimensional geometry + density information + knowledge of crown allometries -> to segment individual tree crowns from LiDAR point clouds

+ 可增加信息，如spectral reflectance


<a name="fundamental-matrix"></a>
## fundamental matrix

["GPSfM: Global Projective SFM Using Algebraic Constraints on Multi-View Fundamental Matrices"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kasten_GPSfM_Global_Projective_SFM_Using_Algebraic_Constraints_on_Multi-View_Fundamental_CVPR_2019_paper.pdf) 2019 CVPR

> F 恢复P
> 
> **fundamental matrix** |

+ n-view fundamental matrix
+ 最小化全局代数误差

+ 对比已有方法，无需初始化，结果稳定


<a name="factorization"></a>
## factorization

["Trust No One: Low Rank Matrix Factorization Using Hierarchical RANSAC"](http://openaccess.thecvf.com/content_cvpr_2016/papers/Oskarsson_Trust_No_One_CVPR_2016_paper.pdf) 2016 CVPR

> 低秩矩阵分解
> 
> **factorization** |


<a name="optimization"></a>
## optimization

["A Unified Optimization Framework for Low-Rank Inducing Penalties"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ornhag_A_Unified_Optimization_Framework_for_Low-Rank_Inducing_Penalties_CVPR_2020_paper.pdf) 2020 CVPR

> 
> 
> **optimization** |

+ 两者正则化：unbiased non-convex formulations + weighted nuclear norm penalties


<a name="mesh"></a>
## mesh

["Meshlet Priors for 3D Mesh Reconstruction"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Badki_Meshlet_Priors_for_3D_Mesh_Reconstruction_CVPR_2020_paper.pdf) 2020 CVPR

> sota OccNet、AtlasNet，不能很好处理物体的先验和物体姿态之间的关系
> 
> **mesh** |

+ 不在训练集的物品，其局部特征可能在训练集中类似的物体上存在
+ meshlet: a dictionary of basic shape features


<a name="stereo"></a>
## stereo

["Multi-Frame GAN: Image Enhancement for Stereo Visual Odometry in Low Light"]() 2019 

> 
>
> **stereo** | **GAN** |

["DPSNET: END-TO-END DEEP PLANE SWEEP STEREO"](https://openreview.net/pdf?id=ryeYHi0ctQ) 2019 ICLR

> 对比 DeMoN、COLMAP、DeepMVS
> 
> **stereo** |

基于 cost colume 的 mulit-view stereo

+ warping through lth plane （sweep）？
+ Feature Volume
+ Cost Volume
+ Cost Fusion

["DeMoN: Depth and Motion Network for Learning Monocular Stereo"](https://arxiv.org/pdf/1612.02401) 2017 CVPR

> 将 SfM 看作学习问题
> 
> **stereo** | **SfM** | **DNN** | **pose** |

+ image pairs -> CNN -> depth、camera motion
+ CNN： multiple stacked encoder-decoder

+ 除了 depth、motion，还估计：1）surface normal；2）optical flow between the imgae；3）confidence of the matching

+ loss = spatial relative differences

1. 相比传统 two-frame SfM，更 accurate、robust
2. 相比 depth-from-single-image network，更 generalization


<a name="privacy"></a>
## privacy

["Privacy Preserving Structure-from-Motion"](https://www.microsoft.com/en-us/research/uploads/prod/2020/08/Geppert2020ECCV-1.pdf) 2020 ECCV

> 
> 
> **privacy** |

+ 将2D/3D特征点转换为随机的2D/3D线


<a name="tips"></a>
## tips

["Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media with Airlight and Scattering Coefficient Estimation"](https://arxiv.org/pdf/2011.09114v1.pdf) 2020 Nov

> dehazing 去雾，depth 未知，MVS 很难估计 photometric consistency
> 
> **dehazing** |

+ deep MVS 来去烟雾
+ dehazing cost volume

["Image Matching across Wide Baselines: From Paper to Practice"](https://arxiv.org/pdf/2003.01587v3.pdf) 2020 Mar

> 对于局部特征、鲁棒估计的基准算法，SfM可提升性能
> 
>

["Leveraging Photogrammetric Mesh Models for Aerial-Ground Feature Point Matching Toward Integrated 3D Reconstruction"](https://arxiv.org/pdf/2002.09085v2.pdf) 2020 Feb

> 
> 
> **三维重建** |

["Robust SfM with Little Image Overlap"](https://arxiv.org/pdf/1703.07957v2.pdf) 2017 Mar

> LineSfM；传统SfM至少需要trifocal的overlaps；减少overlap行不行：图之间只保证有重叠
> 
> **SfM** |

假设line coplanarity

+ 用bifocal估计相对scale变化
+ use trifocal info for line and/or point features
+ parameterless RANSAC-like approach -> robust


<a name="Multi-View_Stereo"></a>
## Multi-View Stereo

["TAPA-MVS: Textureless-Aware PAtchMatch Multi-View Stereo"](https://openaccess.thecvf.com/content_ICCV_2019/papers/Romanoni_TAPA-MVS_Textureless-Aware_PAtchMatch_Multi-View_Stereo_ICCV_2019_paper.pdf) 2019 ICCV

> 估计每个 view 的 depth map、normal map（法线图）；通过基于 patch match 的优化；photo-consistency 光度一致性，重建模型缺乏完整性、使其不可靠
> 
> **point cloud** | **Multi-View Stereo** |

+ 假设无纹理区域是分段平面

+ 数据集：ETH3D

MVS 关键是 depth map 估计

depth map：1）fuse into point cloud；2）volumetric 表达（voxel grid、Delaunay triangulation 德劳内三角）

+ Delaunay 三角化，没有点在三角形外接圆内，避免了“极瘦”的三角形

    连接圆心 -> Voronoi图（沃罗诺伊图）
    
    解凸包问题 convex hull
    
    z = x^2 + y^2，2d平面转三维，找3个点的下凸包，下凸包是2d平面的一个外接圆
    
+ patch match 性能较强

    随机某个点的 patch，再估计领域点

["Scalable Surface Reconstruction from Point Clouds with Extreme Scale and Density Diversity"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mostegel_Scalable_Surface_Reconstruction_CVPR_2017_paper.pdf) 2017 CVPR

> 现有的 mulit-scale surface reconstruction focus on：1）局部尺度变化；2）获取封闭网络，基于全局
> 
> **surface** | **Multi-View Stereo** |

+ a combiantion of: 1) octree data partitioning + 2) delaunay tetrahedralization + 3) graph cut


<a name="SLAM"></a>
## SLAM

## 增大视野

["OmniSLAM: Omnidirectional Localization and Dense Mapping for Wide-baseline Multi-camera Systems"](https://arxiv.org/pdf/2003.08056v1.pdf) 2020 ICRA

> 利用鱼眼相机实现全方向的SLAM。轻量DNN做各个方向的深度估计，整合深度估计到VO，利用深度估计结果做重投影优化
>
> **fisheye** | **多目** | **TSDF** | **深度估计** | **dense map** | **增大视野** |

["Sweepnet: Wide-baseline omnidirectional depth estimation"]() 2019 ICRA

>
>
> **fisheye** | **增大视野** | **深度估计** |

["Rovo: Robust omnidirectional visual odometry for wide-baseline wide-fov camera systems"]() 2019 ICRA

>
>
> **fisheye** | **增大视野** | **visual odometry** |

["Omnimvs: End-to-end learning for omnidirectional stereo matching"]() 2019

>
>
> **fisheye** | **增大视野** | **stereo** |


## LiDAR 

["Tightly coupled 3d lidar inertial odometry and mapping"]() 2019 ICRA

>
>
> **LiDAR** |


## 结构光

["Kinectfusion: Real-time dense surface mapping and tracking"]() 2011 ISMAR

> 
>
> **3D结构光** |


## 稠密建图

["Efficient large-scale stereo matching"]() 2010 ACCV

>
>
> **dense map** |

["Stereoscan: Dense 3d reconstruction in real-time"]() 2011 IEEE

>
>
> **dense map** |


## 深度估计

["Ga-net: Guided aggregation net for end-to-end stereo matching"]() 2019 CVPR

>
>
> **深度估计** |

["Pyramid stereo matching network"]() 2018 CVPR

>
>
> **深度估计** | **stereo** |

["Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation"]() 2018 ECCV

>
>
> **深度估计** |


## 特征提取

["Orb: An efficient alternative to sift or surf"]() 2011 ICCV

>
>
> **feature descriptor** |


## 特征点法

["Real-time monocular SLAM: Why filter?"]() 2010

["ORB-SLAM: a versatile and accurate monocular slam system"]() 2015 IEEE

>
>
> **单目** | **特征点法** |

["ORB-SLAM2: an Open-Source SLAM System forMonocular, Stereo and RGB-D Cameras"](https://arxiv.org/pdf/1610.06475.pdf) 2016 Oct

> a complete SLAM system for monocular, stereo and RGB-D cameras, including map reuse,loop closing and relocalization capabilities
>
> **单目** | **特征点法** | **stereo** | **RGB-D** |

parallel threads:

1) tracking; 2) local mapping; 3) loop closing; 4) full BA

+ place recongition -> relocalization，用于 tracking 失败、在已经 map 的场景中重新初始化、loop 检测

+ 维护一个co-visibiliy graph（两个关键帧观察到相同的点）和一个最小生成树（连接所有关键帧）

+ close points -> translation; far points -> orientation

["ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM"](https://arxiv.org/pdf/2007.11898.pdf) 2020 Jul

>
>
> **单目** | **特征点法** | **visual-inertial odometry** |

![Main system components of ORB-SLAM3](https://github.com/Brook1886/SfM-Notebook/blob/main/image/Main%20system%20components%20of%20ORB-SLAM3.png)


## 直接法 LSD

["Semi-Dense Visual Odometry for a Monocular Camera"](https://openaccess.thecvf.com/content_iccv_2013/papers/Engel_Semi-dense_Visual_Odometry_2013_ICCV_paper.pdf) 2013 ICCV

> 
>
> **semi-dense map** | **直接法** |

["LSD-SLAM: Large-Scale Direct Monocular SLAM"](https://jakobengel.github.io/pdf/engel14eccv.pdf) 2014 ECCV

> 
>
> **直接法** |


## 直接法 DSO

["Omnidirectional DSO: Direct Sparse Odometry with Fisheye Cameras"]() 2018

["Online Photometric Calibration of Auto Exposure Video for Realtime Visual Odometry and SLAM"]() 2018

### ["D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry"](https://arxiv.org/pdf/2003.01060.pdf) 2020

> 利用 self-supervised monocular depth estimation network 估计深度；将深度、pose、uncertainty整合到直接法VO；
> 
> **单目** | **visual odometry** | **DNN** |

Can the deep-predicted poses be employed to boost traditional VO?
How can we incorporate such uncertainty-predictions into optimization-based VO?

> 自监督：基于数据生成(恢复)、数据变换、多模态或辅助信息

+ 预测 brightness transformation parameters

+ DepthNet

    形式：encoder+decoder，encoder采用resnet18，decoder参考["Digging into self-supervised monocular
    depth estimation"](https://arxiv.org/pdf/1806.01260.pdf)

    输入：512\*256图片

    输出：3个channels，$D_t, D_t^s, \Sigma_t$

    训练过程：

    1. 初始化：encoder是resnet18(from PyTorch的torchvision)，采用ImageNet预训练；decoder随机
    2. 超参：batch为8，Adam优化器，epochs20\~40，学习率$10^{-4}$，最后5epochs降至$10^{-5}$

+ PoseNet

    形式：参考["Unsupervised learning of depth and ego-motion from video"](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf)

+ brightness parameters



["Rolling-Shutter Modelling for Visual-Inertial Odometry"]() 2019

["Direct Sparse Odometry With Rolling Shutter"]() 2018

["Deep Virtual Stereo Odometry: Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry"]() 2018

> DVSO
> 
> **深度估计** | **visual odometry** | **stereo** |

+ virtual stereo term

["LDSO: Direct Sparse Odometry with Loop Closure"]() 2018

### ["Direct Sparse Visual-Inertial Odometry using Dynamic Marginalization"](https://arxiv.org/pdf/1804.05625.pdf) 2018

>
>
> **visual-inertial odometry** |

### ["Stereo DSO: Large-Scale Direct Sparse Visual Odometry with Stereo Cameras"]() 2017

> 
> 
> **stereo** |

["Direct Sparse Odometry"](https://arxiv.org/pdf/1607.02565.pdf) 2016

>
>
> **直接法** |


["A Photometrically Calibrated Benchmark For Monocular Visual Odometry"]() 2016


## 直接法 RGB-D

["Dense visual SLAM for RGB-D cameras"]() 2013

> 
> 
> **RGB-D** |



## 视觉+IMU

["Keyframe-based visual-inertial odometry using nonlinear optimization"]() 2015

>
>
> **visual-inertial odometry** |

["Visual-inertial monocular SLAM with map reuse"]() 2017

>
>
> **visual-inertial odometry** |

[" VINS-Mono: A robust and versatile monocular visual-inertial state estimator"]() 2018

>
>
> **visual-inertial odometry** |


## 半直接法 SVO

["CNN-SVO: Improving the mapping in semi-direct visual odometry using singleimage depth prediction"]() 2019

>
>
> **半直接法** | **DNN** |


## 深度学习

["CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction."]() 2017

>
>
> **DNN** |

["Scale recovery for monocular visual odometry using depth estimated with deep convolutional neural fields"]() 2017

>
>
> **DNN** | **visual odometry** |

["Visual odometry revisited: What should be learnt?"]() 2019

>
>
> **DNN** | **visual odometry** |

["GCNv2: Efficient correspondence prediction for real-time slam"]() 2019

>
>
> **DNN** |



## 

["CodeSLAM-learning a compact, optimisable representation for dense visual SLAM"]() 2018

>
>
> 



## visual odometry

["Self-improving visual odometry"]() 2018

>
>
> **visual odometry** |


<a name="robot_grasp"></a>
## robot grasp

[Dex-Net 2.0]() 2017

建立数据库（数据驱动）

如何获取大量的三维模型？
- 合成数据库：3DNet、ShapeNet
- 真实物体扫描：YCB

机械臂模拟器：V-rep、Sapien、Mujoco、Gazebo


