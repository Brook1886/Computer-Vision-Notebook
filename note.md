2019 Dec
"Graph-Based Parallel Large Scale Structure from Motion"
https://arxiv.org/pdf/1912.10659v2.pdf


将大规模SfM问题，看作graph问题

# images cluster
# 最大生成树expand图片
# local reconstruction
# 最小生成树获取精确的相似变换
# 最小高度树找到合适的anchor node，减少误差累计 
2019 Sep
"miniSAM: A Flexible Factor Graph Non-linear Least Squares Optimization Framework"
https://arxiv.org/pdf/1909.00903v1.pdf

factor graphs

非线性最小二乘优化问题

# an open-source C++/Python framework
# a wide list of sparse linear solvers2020 ICLR
"DeepV2D: Video to Depth with Differentiable Structure from Motion"
https://openreview.net/pdf?id=HJeO7RNKPr

DeepV2D  motion 和 depth2018 NeurIPS
"LF-Net: Learning Local Features from Images"
http://papers.nips.cc/paper/7861-lf-net-learning-local-features-from-images.pdf

# 利用depth和motion在一个图片里定一个虚拟目标，网络输出用于其他图片
# 两个分支，限制一个，让另一个可微
# 在indoor（depth 来自3D传感器）和outdoor（depth来自SfM解的估计） datasets下训练

 无监督学习
 60fps QVGA图片（240×320）2015 Oct
"ENFT: Efficient Non-Consecutive Feature Tracking for Robust Structure-from-Motion"
https://arxiv.org/pdf/1510.08012v2.pdf

非连续特征Tracking

？SfM很依赖feature tracking
？对于图片序列：对象移动、偶然遮挡、图片噪声如果处理不好，SfM效果不好
？大规模重建越明显

ENFT用于match被打断的tracking（不同图片子序列，甚至不同video）
解决无明显特征、具有噪声、图片畸变等特征跟丢问题，快速再此tracking

# 一种基于分割的SfM2018 Jun
"BA-Net: Dense Bundle Adjustment Network"
https://arxiv.org/pdf/1806.04807v3.pdf

dense SfM，基于feature度量的BA

# 多视图几何约束  feature-metric error
# depth parameterization 恢复深度
# image  几个basis depth maps  线性组合  final depth （via feature-metric BA）2018 ECCV
"GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"
http://openaccess.thecvf.com/content_ECCV_2018/papers/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.pdf

整合多视图重建时的几何约束来学习局部特征2020 May
"Reference Pose Generation for Long-term Visual Localization via Learned Features and View Synthesis"
https://arxiv.org/pdf/2005.05179v3.pdf

视觉定位

？SfM依赖局部特征，外部环境变化易失败
？手工标注特征对应可能不准确

# 学习到的特征（3D模型和图片之间匹配到的特征） pose
# 半自动

 Aachen Day-Night dataset 有47%提升
2019 Dec
"Self-Supervised 3D Keypoint Learning for Ego-motion Estimation"
https://arxiv.org/pdf/1912.03426v3.pdf

detect and match viewpoint-invariant keypoint

？基于学习的方法：keypoint 经单应到synthetic views
该方法对非共面且有光照变化的场景不泛化

# video  自监督学习具有深度信息的关键点
# 可微分SfM模型
# 外观+几何匹配  学习关键点、深度估计
2020 Jun
"AdaLAM: Revisiting Handcrafted Outlier Detection"
https://arxiv.org/pdf/2006.04250v1.pdf

 Local feature matching

？匹配包含outliers

# a hierarchical pipeline for effective outlier detection
# 并行计算，fast
2020 ECCV
"Infrastructure-based Multi-Camera Calibration using Radial Projections"
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610324.pdf

？多相机系统，已知相机内参，估计外参
using 3D map

假设畸变主要为径向

# 先初步估计每个相机的外参，再求内参和精确的外参

 比先估计内参和姿态、再求外参的方法鲁棒
2019 Sep
"Flow-Motion and Depth Network for Monocular Stereo and Beyond"
https://arxiv.org/pdf/1909.05452v1.pdf

# 2 images、intrinsic  pose、depth map
# network估计 光流和相机姿态
# 三角测量层encode光流和相机姿态
# target images的depth  network  估计source image的depth
# 提供给网络训练的数据集
2017 Mar
"Robust SfM with Little Image Overlap"
https://arxiv.org/pdf/1703.07957v2.pdf

LineSfM

？传统SfM至少需要trifocal的overlaps
？减少overlap行不行：图之间只保证有重叠

假设line coplanarity
# 用bifocal估计相对scale变化
# use trifocal info for line and/or point features
# parameterless RANSAC-like approach  robust

2020 ECCV
"Multi-View Optimization of Local Feature Geometry"
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460647.pdf

local feature detect

？单视图提取关键点来提取特征不准，能否多视图

# 拿不准的match先估计局部几何变换，利用多视图优化关键点位置，最小二乘

 能改善三角测量和相机定位2018 Oct
"Geometry meets semantics for semi-supervised monocular depth estimation"
https://arxiv.org/pdf/1810.04093v2.pdf

？single image 缺乏几何信息
？encoder-decoder依赖有效的特征表达

# semantics  改善深度的估计
# image warping loss
# semi-supervised，semantic使用GT
# cross-task loss

KITTI
2020 ICLR
"Neural Outlier Rejection for Self-Supervised Keypoint Learning"
https://openreview.net/pdf?id=Skx82ySYPH

？学习keypoint的方法有效，训练数据保证兴趣点准确较为困难

# IO-Net (i.e. InlierOutlierNet)
# 自监督：keypoint检测、描述、匹配
sample inlier和outlier set

# KeyPointNet  keypoint检测、描述
# spatial discretizations
# subpixel conv，上采用descriptor fmap分辨率，改善描述

2019 CVPR
"Single Image Depth Estimation Trained via Depth from Defocus Cues"
http://openaccess.thecvf.com/content_CVPR_2019/papers/Gur_Single_Image_Depth_Estimation_Trained_via_Depth_From_Defocus_Cues_CVPR_2019_paper.pdf

defocus cues 散焦视差

# Point Spread Function：conv layer、散光圈（Circle-Of-Confusion）

 KITTI and Make3D 数据集2020 Mar
"Image Matching across Wide Baselines: From Paper to Practice"
https://arxiv.org/pdf/2003.01587v3.pdf

对于局部特征、鲁棒估计的基准算法，SfM可提升性能2020 Apr
"Robust Line Segments Matching via Graph Convolution Networks"
https://arxiv.org/pdf/2004.04993v2.pdf

直线匹配

纹理较少、重复结构场景，直线匹配更为重要（对SfM，SLAM）

# GCN  匹配两图中的直线
# 直线匹配转换为图的优化问题2019 Feb
"Deep Interpretable Non-Rigid Structure from Motion"
https://arxiv.org/pdf/1902.10840v1.pdf

？NRSfM问题困难：图片数量；shape变化的handle程度

# DNN  camera poses、3D点（2D坐标系生成？）
# DNN 可解释为多层稀疏字典学习问题
# 基于权重提出一种评估方法，需要多少GT能确保所需置信度

2020 CVPR
"Tangent Images for Mitigating Spherical Distortion"
http://openaccess.thecvf.com/content_CVPR_2020/papers/Eder_Tangent_Images_for_Mitigating_Spherical_Distortion_CVPR_2020_paper.pdf

"tangent images"

球型

二十面体

2020 Nov
"RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty"
https://arxiv.org/pdf/2011.10359v1.pdf

同时估计dense depth map和camera poses
indoor

传统分两步：1）相机估计；2）MVS

# deep net  "depth-planes" 线性组合  depth map
# altered BA  poses、dense reconstructions
# high-quality sparse keypoint matches  优化：1）前一帧 "depth-planes" 线性组合；2）相机姿态
2017 Feb
"Parallel Structure from Motion from Local Increment to Global Averaging"
https://arxiv.org/pdf/1702.08601v3.pdf

？accurate、consistent

不同于以往靠简化、牺牲精度，企图用并行来加速

# camera clusters with overlapping
# local increment SfM  relative poses  global motion averaging  accurate and consistent global camera poses
# track generation + local SfM + 3D point triangulation + bundle adjustment

 a city-scale data-set (one million high-resolution images)
2019 Aug
"DefSLAM: Tracking and Mapping of Deforming Scenes from Monocular Sequences"
https://arxiv.org/pdf/1908.08918v2.pdf

解决deforming scenes

# Shape-from-Template (SfT) + Non-Rigid Structure-from-Motion (NRSfM)
# deformation tracking thread  poses、deformation

2020 Aug
"Integration of the 3D Environment for UAV Onboard Visual Object Tracking"
https://arxiv.org/pdf/2008.02834v3.pdf

？困难：object occlusion, small-scale objects, background clutter, and abrupt camera motion

# 在三维重建的场景中，检测和跟踪
# a model-free visual object tracker, a sparse 3D reconstruction, and a state estimator
# representing the position of the target in 3D space rather than in image space

 low-altitude oblique view，image sequences

 比 plain visual cues 和 image-space-based state estimations 效果好

todo：SfM结合去动态物体算法
2019 CVPR
"LiFF: Light Field Features in Scale and Depth"
http://openaccess.thecvf.com/content_CVPR_2019/papers/Dansereau_LiFF_Light_Field_Features_in_Scale_and_Depth_CVPR_2019_paper.pdf

解决特征检测、描述

？场景中light问题：部分遮挡、低对比度、表面反射、折射

# 光场相机
# 4D光场
# 尺度不变、4D光场检测特征、对透视变换鲁棒

能提高SfM效果2019 Oct
"RPBA -- Robust Parallel Bundle Adjustment Based on Covariance Information"
https://arxiv.org/pdf/1910.08138v1.pdf

并行BA

 consensus-based optimization methods

# adjustment 3d point  covariance information
2019 Mar
"Trifocal Relative Pose from Lines at Points and its Efficient Solution"
https://arxiv.org/pdf/1903.09755v3.pdf

？relative pose estimation

mixed point、line correspondences、three views

# 最小化问题：1）3points+1line；2）2points+2lines
# a suitable homotopy continuation technique

解决2views失败的重建2017 CVPR
"Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context"
http://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.pdf

？场景里具有重复结构的重建，需disambiguation
当前靠背景纹理来解决，不能解决不清晰场景

# encode global topology
# viewpoints的流形manifold  逼近可用图像
# 相似的结构造成重建困惑的，实际场景中都相距较远（测地线geodesic）
# adjacent viewpoint  adaptively identifying cameras  manifold
# geodesic consistency  detect ambiguities
2020 Jun
"SEKD: Self-Evolving Keypoint Detection and Description"
https://arxiv.org/pdf/2006.05077v1.pdf

# local feature detector 与 descriptor 交互
# 自监督，unlabeled natural images
# training strategies

 homography estimation, relative pose estimation, and structure-from-motion tasks2019 ICCV
"Cascaded Parallel Filtering for Memory-Efficient Image-Based Localization"
http://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Cascaded_Parallel_Filtering_for_Memory-Efficient_Image-Based_Localization_ICCV_2019_paper.pdf

？Image-based localization (IBL) ：估计 camera poses，规模大SfM如何节省memory

# cascaded parallel filtering：利用feature, visibility and geometry information来filter wrong matches
2020 ECCV
"NeuRoRA: Neural Robust Rotation Averaging"
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690137.pdf

？robust cost functions 非线性、且基于噪声和outliers的分布假设

# view-graph cleaning network（noise、outliers）+ fine-tune network
# 用合成graphs训练

 graph-based network可以用于其他graph-based geometric problems，像pose-graph optimization 2018 Aug
"Efficient Outlier Removal in Large Scale Global Structure-from-Motion"
https://arxiv.org/pdf/1808.03041v4.pdf

 ？global outlier removal

# a convex relaxed  minimization，可用线性规划（LP）解
# ideal  minimization，可用an iteratively reweighted method解
2020 CVPR
"Uncertainty Based Camera Model Selection"
http://openaccess.thecvf.com/content_CVPR_2020/papers/Polic_Uncertainty_Based_Camera_Model_Selection_CVPR_2020_paper.pdf

？相机模型决定SfM重建效果

# 基于不确定性估计自动选择相机模型

# Accuracy-based Criterion
2020 Feb
"Leveraging Photogrammetric Mesh Models for Aerial-Ground Feature Point Matching Toward Integrated 3D Reconstruction"
https://arxiv.org/pdf/2002.09085v2.pdf


2020 Jan
"Deep NRSfM++: Towards 3D Reconstruction in the Wild"
https://arxiv.org/pdf/2001.10090v1.pdf

？non-rigid

 2D landmarks stemming？

？Classical NRSfM方法不能handle大规模图集且只能处理少数shape
？当前仍然有的问题：不能handle missing/occluded points；仅仅弱透视相机模型

2019 Mar
"Three-dimensional Segmentation of Trees Through a Flexible Multi-Class Graph Cut Algorithm (MCGC)"
https://arxiv.org/pdf/1903.08481v1.pdf

？individual tree crown (ITC) 单棵树冠检测问题
 Many algorithms exist for structurally simple forests including coniferous forests and plantations.

？树的种类较多，热带

# local three-dimensional geometry + 
   density information + 
   knowledge of crown allometries + 
    to segment individual tree crowns from LiDAR point clouds

# 可增加信息，如spectral reflectance
2019 CVPR
"GPSfM: Global Projective SFM Using Algebraic Constraints on Multi-View Fundamental Matrices"
http://openaccess.thecvf.com/content_CVPR_2019/papers/Kasten_GPSfM_Global_Projective_SFM_Using_Algebraic_Constraints_on_Multi-View_Fundamental_CVPR_2019_paper.pdf

？F  恢复P

# n-view fundamental matrix
# 最小化全局代数误差

# 对比已有方法，无需初始化，结果稳定
2020 Nov
"SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments"
https://arxiv.org/pdf/2011.04408v1.pdf

季节、光照变换，缺乏数据集和基准

2020 ECCV
"Relative Pose from Deep Learned Depth and a Single Affine Correspondence"
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf

# 结合 non-metric monocular depth +
   affine correspondences  
   从single correspondence估计relative pose
# 1-point RANSAC approaches
# 1AC+D solver

使用global SfM在 1DSfM 数据集上验证
2020 ECCV
"LandscapeAR: Large Scale Outdoor Augmented Reality by Matching Photographs with Terrain Models Using Learned Descriptors"
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf

大规模AR

# textured Digital Elevation Models (DEMs)

# 用SfM重建出训练数据，训练a cross-domain feature descriptor

# 可在移动设备上用
2020 Nov
"Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media with Airlight and Scattering Coefficient Estimation"
https://arxiv.org/pdf/2011.09114v1.pdf

？dehazing 去雾，depth未知，MVS很难估计photometric consistency

# deep MVS 来去烟雾
# dehazing cost volume2020 Nov
"Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption"
https://arxiv.org/pdf/2011.03106v1.pdf

？rolling shutter会导致图像畸变， 卷帘快门的影响如何矫正

# dnn  depth、camera poses
# 该方法无需假设相机motion，输入图片即可
# TUM rolling shutter dataset2019 Jan
"Resultant Based Incremental Recovery of Camera Pose from Pairwise Matches"
https://arxiv.org/pdf/1901.09364v1.pdf

# six-point online algorithm 恢复外参，incremental 到第n张图片
# Dixon resultant？
# Bernstein's theorem？证明了复解个数的严格上下界
2018 CVPR
"Structure from Recurrent Motion: From Rigidity to Recurrency"
http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Structure_From_Recurrent_CVPR_2018_paper.pdf

？解决Non-Rigid Structure-from-Motion (NRSfM)
单目序列video，存在周期性变化

# 转周期性变化为刚性变化

# 以刚性判断来聚类视角
2016 CVPR
"Trust No One: Low Rank Matrix Factorization Using Hierarchical RANSAC"
http://openaccess.thecvf.com/content_cvpr_2016/papers/Oskarsson_Trust_No_One_CVPR_2016_paper.pdf

低秩矩阵分解

2009 NeurlPS
"Matrix Completion from Noisy Entries"
http://papers.nips.cc/paper/3777-matrix-completion-from-noisy-entries.pdf

# OptSpace

2020 CVPR
"A Unified Optimization Framework for Low-Rank Inducing Penalties"
http://openaccess.thecvf.com/content_CVPR_2020/papers/Ornhag_A_Unified_Optimization_Framework_for_Low-Rank_Inducing_Penalties_CVPR_2020_paper.pdf

# 两者正则化：unbiased non-convex formulations + weighted nuclear norm penalties