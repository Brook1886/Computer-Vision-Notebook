# SfM-Notebook

关于SfM学习的记录，持续更新

## contents

- [papers](#papers)
    - [incremental ](#incremental)
    - [global](#global)
    - [multi-stage](#multi-stage)
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
    - [tips](#tips)

<a name="papers"></a>
## papers

<a name="incremental"></a>
### incremental
#### ["Structure-from-Motion Revisited"](https://demuc.de/papers/schoenberger2016sfm.pdf) 2016 CVPR

+ scene graph augmentation
    + 估计基本矩阵F，当inliers >= NF, image pair 几何匹配
    + 根据决定H的inliers数目NH划分H
        
        GRIC法
        
        如果 NH / NF < eHF, camera 在同一个场景移动
        
    + 对于标定好的相机
    
        E 本质矩阵，NE / NF > eEF, 说明标定正确
        
        分解E、三角测量、median triangulation angle \alpth_m, 可区分pure rotation、planar scenes
        
    + 辨别 WTF（watermarks、timestamps frames）不加入 scene graph
    + valid pairs
    
+ nest best view selection
    
    目标：minimize reconstruction error
    
    使用 uncertainty-driven 方法
    
    > PnP 相机姿态估计的精度，取决于观察点的数量、及其在images上的分布；
        没有（有误）相机标定下，估计内参
    
    

<a name="global"></a>
### global
#### ["Global Structure-from-Motion by Similarity Averaging"](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cui_Global_Structure-From-Motion_by_ICCV_2015_paper.pdf) 2015 ICCV

已知：images，由5点算法计算的E，EG graph；求：朝向R和相机中心c

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
    

<a name="multi-stage"></a>
### multi-stage
#### ["HSfM: Hybrid Structure-from-Motion"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_HSfM_Hybrid_Structure-from-Motion_CVPR_2017_paper.pdf) 2017 ICCV

> incremental 鲁棒、精度高，但是效率较低

> global 对outliers敏感

> accuracy + robustness + efficiency 都拿？

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
        
<a name="graph-based"></a>
### graph-based

#### ["NeuRoRA: Neural Robust Rotation Averaging"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690137.pdf) 2020 ECCV 

> robust cost functions 非线性、且基于噪声和outliers的分布假设

+ view-graph cleaning network（noise、outliers）+ fine-tune network
+ 用合成graphs训练

1. graph-based network可以用于其他graph-based geometric problems，像pose-graph optimization

#### ["Graph-Based Parallel Large Scale Structure from Motion"](https://arxiv.org/pdf/1912.10659v2.pdf) 2019 Dec

将大规模SfM问题，看作graph问题

+ images cluster
+ 最大生成树expand图片
+ local reconstruction
+ 最小生成树获取精确的相似变换
+ 最小高度树找到合适的anchor node，减少误差累计 

<a name="factor-graph"></a>
### factor graph
#### ["miniSAM: A Flexible Factor Graph Non-linear Least Squares Optimization Framework"](https://arxiv.org/pdf/1909.00903v1.pdf) 2019 Sep

factor graphs

非线性最小二乘优化问题

+ an open-source C++/Python framework
+ a wide list of sparse linear solvers

<a name="depth"></a>
### depth
#### ["SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments"](https://arxiv.org/pdf/2011.04408v1.pdf) 2020 Nov

季节、光照变换，缺乏数据集和基准

#### ["RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty"](https://arxiv.org/pdf/2011.10359v1.pdf) 2020 Nov

同时估计dense depth map和camera poses
indoor

传统分两步：1）相机估计；2）MVS

+ deep net -> "depth-planes" 线性组合 -> depth map
+ altered BA -> poses、dense reconstructions
+ high-quality sparse keypoint matches -> 优化：1）前一帧 "depth-planes" 线性组合；2）相机姿态

#### ["DeepV2D: Video to Depth with Differentiable Structure from Motion"](https://openreview.net/pdf?id=HJeO7RNKPr) 2020 ICLR

输出 motion 和 depth

#### ["Self-Supervised 3D Keypoint Learning for Ego-motion Estimation"](https://arxiv.org/pdf/1912.03426v3.pdf) 2019 Dec

detect and match viewpoint-invariant keypoint

> 基于学习的方法：keypoint 经单应到synthetic views

该方法对非共面且有光照变化的场景不泛化

+ video -> 自监督学习具有深度信息的关键点
+ 可微分SfM模型
+ 外观+几何匹配 -> 学习关键点、深度估计

#### ["Single Image Depth Estimation Trained via Depth from Defocus Cues"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gur_Single_Image_Depth_Estimation_Trained_via_Depth_From_Defocus_Cues_CVPR_2019_paper.pdf) 2019 CVPR

defocus cues 散焦视差

+ Point Spread Function：conv layer、散光圈（Circle-Of-Confusion）

KITTI and Make3D 数据集

#### ["Geometry meets semantics for semi-supervised monocular depth estimation"](https://arxiv.org/pdf/1810.04093v2.pdf) 2018 Oct

> single image 缺乏几何信息

> encoder-decoder依赖有效的特征表达

+ semantics -> 改善深度的估计
+ image warping loss
+ semi-supervised，semantic使用GT
+ cross-task loss

KITTI

<a name="feature"></a>
### feature
#### keypoint
#### ["SEKD: Self-Evolving Keypoint Detection and Description"](https://arxiv.org/pdf/2006.05077v1.pdf) 2020 Jun

+ local feature detector 与 descriptor 交互
+ 自监督，unlabeled natural images
+ training strategies

1. homography estimation, relative pose estimation, and structure-from-motion tasks

#### ["Neural Outlier Rejection for Self-Supervised Keypoint Learning"](https://openreview.net/pdf?id=Skx82ySYPH) 2020 ICLR

> 学习keypoint的方法有效，训练数据保证兴趣点准确较为困难

+ IO-Net (i.e. InlierOutlierNet)
+ 自监督：keypoint检测、描述、匹配
sample inlier和outlier set

+ KeyPointNet -> keypoint检测、描述
+ spatial discretizations
+ subpixel conv，上采用descriptor fmap分辨率，改善描述

#### feature dectect
#### ["Multi-View Optimization of Local Feature Geometry"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460647.pdf) 2020 ECCV

local feature detect

> 单视图提取关键点来提取特征不准，能否多视图

+ 拿不准的match先估计局部几何变换，利用多视图优化关键点位置，最小二乘

1. 能改善三角测量和相机定位
 
#### ["LiFF: Light Field Features in Scale and Depth"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dansereau_LiFF_Light_Field_Features_in_Scale_and_Depth_CVPR_2019_paper.pdf) 2019 CVPR

解决特征检测、描述

> 场景中light问题：部分遮挡、低对比度、表面反射、折射

+ 光场相机
+ 4D光场
+ 尺度不变、4D光场检测特征、对透视变换鲁棒

能提高SfM效果

#### feature description
#### ["LandscapeAR: Large Scale Outdoor Augmented Reality by Matching Photographs with Terrain Models Using Learned Descriptors"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf) 2020 ECCV

大规模AR

+ textured Digital Elevation Models (DEMs)

+ 用 SfM 重建出训练数据，训练 a cross-domain feature descriptor

+ 可在移动设备上用

#### ["LF-Net: Learning Local Features from Images"](http://papers.nips.cc/paper/7861-lf-net-learning-local-features-from-images.pdf) 2018 NeurIPS

+ 利用depth和motion在一个图片里定一个虚拟目标，网络输出用于其他图片
+ 两个分支，限制一个，让另一个可微
+ 在indoor（depth 来自3D传感器）和outdoor（depth来自SfM解的估计） datasets下训练

1. 无监督学习
2. 60fps QVGA图片（240×320）

#### ["GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.pdf) 2018 ECCV

整合多视图重建时的几何约束来学习局部特征

#### feature tracking
#### ["Integration of the 3D Environment for UAV Onboard Visual Object Tracking"](https://arxiv.org/pdf/2008.02834v3.pdf) 2020 Aug

> 困难：object occlusion, small-scale objects, background clutter, and abrupt camera motion

+ 在三维重建的场景中，检测和跟踪
+ a model-free visual object tracker, a sparse 3D reconstruction, and a state estimator
+ representing the position of the target in 3D space rather than in image space

low-altitude oblique view，image sequences

1. 比 plain visual cues 和 image-space-based state estimations 效果好

todo：SfM结合去动态物体算法

#### ["ENFT: Efficient Non-Consecutive Feature Tracking for Robust Structure-from-Motion"](https://arxiv.org/pdf/1510.08012v2.pdf) 2015 Oct

非连续特征Tracking

> SfM很依赖feature tracking

> 对于图片序列：对象移动、偶然遮挡、图片噪声如果处理不好，SfM效果不好

> 大规模重建越明显

ENFT用于match被打断的tracking（不同图片子序列，甚至不同video）
解决无明显特征、具有噪声、图片畸变等特征跟丢问题，快速再此tracking

+ 一种基于分割的SfM

#### feature matching
#### ["AdaLAM: Revisiting Handcrafted Outlier Detection"](https://arxiv.org/pdf/2006.04250v1.pdf) 2020 Jun

Local feature matching

> 匹配包含outliers

+ a hierarchical pipeline for effective outlier detection
+ 并行计算，fast

#### ["Robust Line Segments Matching via Graph Convolution Networks"](https://arxiv.org/pdf/2004.04993v2.pdf) 2020 Apr

直线匹配

纹理较少、重复结构场景，直线匹配更为重要（对SfM，SLAM）

+ GCN -> 匹配两图中的直线
+ 直线匹配转换为图的优化问题

<a name="outlier"></a>
### outlier
#### ["Efficient Outlier Removal in Large Scale Global Structure-from-Motion"](https://arxiv.org/pdf/1808.03041v4.pdf) 2018 Aug

> global outlier removal

+ a convex relaxed l_1 minimization，可用线性规划（LP）解
+ ideal l_0 minimization，可用an iteratively reweighted method解

<a name="bundle-adjustment"></a>
### bundle adjustment
#### ["RPBA -- Robust Parallel Bundle Adjustment Based on Covariance Information"](https://arxiv.org/pdf/1910.08138v1.pdf) 2019 Oct

并行BA

consensus-based optimization methods

+ adjustment 3d point -> covariance information

#### ["BA-Net: Dense Bundle Adjustment Network"](https://arxiv.org/pdf/1806.04807v3.pdf) 2018 Jun

dense SfM，基于feature度量的BA

+ 多视图几何约束 -> feature-metric error
+ depth parameterization 恢复深度
+ image -> 几个basis depth maps -> 线性组合 -> final depth （via feature-metric BA）

<a name="localization"></a>
### localization
#### ["Reference Pose Generation for Long-term Visual Localization via Learned Features and View Synthesis"](https://arxiv.org/pdf/2005.05179v3.pdf) 2020 May

视觉定位

> SfM依赖局部特征，外部环境变化易失败

> 手工标注特征对应可能不准确

+ 学习到的特征（3D模型和图片之间匹配到的特征）-> pose
+ 半自动

Aachen Day-Night dataset 有47%提升

#### ["Cascaded Parallel Filtering for Memory-Efficient Image-Based Localization"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Cascaded_Parallel_Filtering_for_Memory-Efficient_Image-Based_Localization_ICCV_2019_paper.pdf) 2019 ICCV

> Image-based localization (IBL) ：估计 camera poses，规模大SfM如何节省memory

+ cascaded parallel filtering：利用feature, visibility and geometry information来filter wrong matches

<a name="calibration"></a>
### calibration
#### ["Infrastructure-based Multi-Camera Calibration using Radial Projections"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610324.pdf) 2020 ECCV

> 多相机系统，已知相机内参，估计外参

using 3D map

假设畸变主要为径向

+ 先初步估计每个相机的外参，再求内参和精确的外参

1. 比先估计内参和姿态、再求外参的方法鲁棒

<a name="motion"></a>
### motion
#### ["Relative Pose from Deep Learned Depth and a Single Affine Correspondence"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf) 2020 ECCV

+ 结合 non-metric monocular depth + affine correspondences -> 从single correspondence 估计 relative pose
+ 1-point RANSAC approaches
+ 1AC+D solver

使用 global SfM 在 1DSfM 数据集上验证

#### ["Resultant Based Incremental Recovery of Camera Pose from Pairwise Matches"](https://arxiv.org/pdf/1901.09364v1.pdf) 2019 Jan

+ six-point online algorithm 恢复外参，incremental 到第n张图片
+ Dixon resultant？
+ Bernstein's theorem？证明了复解个数的严格上下界

#### ["Flow-Motion and Depth Network for Monocular Stereo and Beyond"](https://arxiv.org/pdf/1909.05452v1.pdf) 2019 Sep

+ 2 images、intrinsic -> pose、depth map
+ network估计 光流和相机姿态
+ 三角测量层encode光流和相机姿态
+ target images的depth -> network -> 估计source image的depth
+ 提供给网络训练的数据集

#### ["Trifocal Relative Pose from Lines at Points and its Efficient Solution"](https://arxiv.org/pdf/1903.09755v3.pdf) 2019 Mar

> relative pose estimation

mixed point、line correspondences、three views

+ 最小化问题：1）3points+1line；2）2points+2lines
+ a suitable homotopy continuation technique

解决2views失败的重建

<a name="non-rigid"></a>
### non-rigid
#### ["Deep NRSfM++: Towards 3D Reconstruction in the Wild"](https://arxiv.org/pdf/2001.10090v1.pdf) 2020 Jan

> non-rigid

2D landmarks stemming？

> Classical NRSfM 方法不能 handle 大规模图集且只能处理少数 shape
> 当前仍然有的问题：不能 handle missing/occluded points；仅仅弱透视相机模型

#### ["DefSLAM: Tracking and Mapping of Deforming Scenes from Monocular Sequences"](https://arxiv.org/pdf/1908.08918v2.pdf) 2019 Aug

解决deforming scenes

+ Shape-from-Template (SfT) + Non-Rigid Structure-from-Motion (NRSfM)
+ deformation tracking thread -> poses、deformation

#### ["Deep Interpretable Non-Rigid Structure from Motion"](https://arxiv.org/pdf/1902.10840v1.pdf) 2019 Feb

> NRSfM问题困难：图片数量；shape变化的handle程度

+ DNN -> camera poses、3D点（2D坐标系生成？）
+ DNN 可解释为多层稀疏字典学习问题
+ 基于权重提出一种评估方法，需要多少GT能确保所需置信度

#### ["Structure from Recurrent Motion: From Rigidity to Recurrency"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Structure_From_Recurrent_CVPR_2018_paper.pdf) 2018 CVPR

> 解决Non-Rigid Structure-from-Motion (NRSfM)

单目序列video，存在周期性变化

+ 转周期性变化为刚性变化

+ 以刚性判断来聚类视角

<a name="distortion"></a>
### distortion
#### ["Tangent Images for Mitigating Spherical Distortion"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Eder_Tangent_Images_for_Mitigating_Spherical_Distortion_CVPR_2020_paper.pdf) 2020 CVPR

"tangent images"

球型

二十面体

<a name="parallel"></a>
### parallel
#### ["Parallel Structure from Motion from Local Increment to Global Averaging"](https://arxiv.org/pdf/1702.08601v3.pdf) 2017 Feb

> accurate、consistent

不同于以往靠简化、牺牲精度，企图用并行来加速

+ camera clusters with overlapping
+ local increment SfM -> relative poses -> global motion averaging -> accurate and consistent global camera poses
+ track generation + local SfM + 3D point triangulation + bundle adjustment

1. a city-scale data-set (one million high-resolution images)

<a name="disambiguation"></a>
### disambiguation
#### ["Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context"](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.pdf) 2017 CVPR

> 场景里具有重复结构的重建，需disambiguation

当前靠背景纹理来解决，不能解决不清晰场景

+ encode global topology
+ viewpoints的流形manifold -> 逼近可用图像
+ 相似的结构造成重建困惑的，实际场景中都相距较远（测地线geodesic）
+ adjacent viewpoint -> adaptively identifying cameras -> manifold
+ geodesic consistency -> detect ambiguities

<a name="camera-model"></a>
### camera model
#### ["Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption"](https://arxiv.org/pdf/2011.03106v1.pdf) 2020 Nov

> rolling shutter会导致图像畸变， 卷帘快门的影响如何矫正

+ dnn -> depth、camera poses
+ 该方法无需假设相机 motion，输入图片即可
+ TUM rolling shutter dataset

#### ["Uncertainty Based Camera Model Selection"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Polic_Uncertainty_Based_Camera_Model_Selection_CVPR_2020_paper.pdf) 2020 CVPR

> 相机模型决定SfM重建效果

+ 基于不确定性估计自动选择相机模型

+ Accuracy-based Criterion

<a name="segmentation"></a>
### segmentation
#### ["Three-dimensional Segmentation of Trees Through a Flexible Multi-Class Graph Cut Algorithm (MCGC)"](https://arxiv.org/pdf/1903.08481v1.pdf) 2019 Mar

> individual tree crown (ITC) 单棵树冠检测问题

Many algorithms exist for structurally simple forests including coniferous forests and plantations.

> 树的种类较多，热带

+ local three-dimensional geometry + density information + knowledge of crown allometries -> to segment individual tree crowns from LiDAR point clouds

+ 可增加信息，如spectral reflectance

<a name="fundamental-matrix"></a>
### fundamental matrix
#### ["GPSfM: Global Projective SFM Using Algebraic Constraints on Multi-View Fundamental Matrices"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kasten_GPSfM_Global_Projective_SFM_Using_Algebraic_Constraints_on_Multi-View_Fundamental_CVPR_2019_paper.pdf) 2019 CVPR

> F  恢复P

+ n-view fundamental matrix
+ 最小化全局代数误差

+ 对比已有方法，无需初始化，结果稳定

<a name="factorization"></a>
### factorization
#### ["Trust No One: Low Rank Matrix Factorization Using Hierarchical RANSAC"](http://openaccess.thecvf.com/content_cvpr_2016/papers/Oskarsson_Trust_No_One_CVPR_2016_paper.pdf) 2016 CVPR

低秩矩阵分解

<a name="optimization"></a>
### optimization
#### ["A Unified Optimization Framework for Low-Rank Inducing Penalties"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ornhag_A_Unified_Optimization_Framework_for_Low-Rank_Inducing_Penalties_CVPR_2020_paper.pdf) 2020 CVPR

+ 两者正则化：unbiased non-convex formulations + weighted nuclear norm penalties

<a name="mesh"></a>
### mesh
#### ["Meshlet Priors for 3D Mesh Reconstruction"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Badki_Meshlet_Priors_for_3D_Mesh_Reconstruction_CVPR_2020_paper.pdf) 2020 CVPR

> sota OccNet、AtlasNet，不能很好处理物体的先验和物体姿态之间的关系

+ 不在训练集的物品，其局部特征可能在训练集中类似的物体上存在
+ meshlet: a dictionary of basic shape features

<a name="stereo"></a>
### stereo
#### ["DPSNET: END-TO-END DEEP PLANE SWEEP STEREO"](https://openreview.net/pdf?id=ryeYHi0ctQ) 2019 ICLR

对比 DeMoN、COLMAP、DeepMVS

基于 cost colume 的 mulit-view stereo

+ warping through lth plane （sweep）？
+ Feature Volume
+ Cost Volume
+ Cost Fusion

<a name="tips"></a>
### tips
#### ["Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media with Airlight and Scattering Coefficient Estimation"](https://arxiv.org/pdf/2011.09114v1.pdf) 2020 Nov

> dehazing 去雾，depth 未知，MVS 很难估计 photometric consistency

+ deep MVS 来去烟雾
+ dehazing cost volume

#### ["Image Matching across Wide Baselines: From Paper to Practice"](https://arxiv.org/pdf/2003.01587v3.pdf) 2020 Mar

对于局部特征、鲁棒估计的基准算法，SfM可提升性能

#### ["Leveraging Photogrammetric Mesh Models for Aerial-Ground Feature Point Matching Toward Integrated 3D Reconstruction"](https://arxiv.org/pdf/2002.09085v2.pdf) 2020 Feb

#### ["Robust SfM with Little Image Overlap"](https://arxiv.org/pdf/1703.07957v2.pdf) 2017 Mar

LineSfM

> 传统SfM至少需要trifocal的overlaps

> 减少overlap行不行：图之间只保证有重叠

假设line coplanarity

+ 用bifocal估计相对scale变化
+ use trifocal info for line and/or point features
+ parameterless RANSAC-like approach -> robust
