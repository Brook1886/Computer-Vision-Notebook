# SfM-Notebook
关于SfM学习的记录，持续更新

## contents

- [papers](#papers)
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
    - [tips](#tips)

<a name="papers"></a>
## papers
<a name="graph-based"></a>
### graph-based

#### ["NeuRoRA: Neural Robust Rotation Averaging"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690137.pdf) 2020 ECCV 

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

#### ["RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty"](https://arxiv.org/pdf/2011.10359v1.pdf) 2020 Nov

#### ["DeepV2D: Video to Depth with Differentiable Structure from Motion"](https://openreview.net/pdf?id=HJeO7RNKPr) 2020 ICLR

输出 motion 和 depth

#### ["Self-Supervised 3D Keypoint Learning for Ego-motion Estimation"](https://arxiv.org/pdf/1912.03426v3.pdf) 2019 Dec

#### ["Single Image Depth Estimation Trained via Depth from Defocus Cues"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gur_Single_Image_Depth_Estimation_Trained_via_Depth_From_Defocus_Cues_CVPR_2019_paper.pdf) 2019 CVPR

#### ["Geometry meets semantics for semi-supervised monocular depth estimation"](https://arxiv.org/pdf/1810.04093v2.pdf) 2018 Oct

<a name="feature"></a>
### feature
#### keypoint
#### ["SEKD: Self-Evolving Keypoint Detection and Description"](https://arxiv.org/pdf/2006.05077v1.pdf) 2020 Jun

#### ["Neural Outlier Rejection for Self-Supervised Keypoint Learning"](https://openreview.net/pdf?id=Skx82ySYPH) 2020 ICLR

#### feature dectect
#### ["Multi-View Optimization of Local Feature Geometry"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460647.pdf) 2020 ECCV

#### ["LiFF: Light Field Features in Scale and Depth"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dansereau_LiFF_Light_Field_Features_in_Scale_and_Depth_CVPR_2019_paper.pdf) 2019 CVPR

#### feature description
#### ["LandscapeAR: Large Scale Outdoor Augmented Reality by Matching Photographs with Terrain Models Using Learned Descriptors"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf) 2020 ECCV

#### ["LF-Net: Learning Local Features from Images"](http://papers.nips.cc/paper/7861-lf-net-learning-local-features-from-images.pdf) 2018 NeurIPS

+ 利用depth和motion在一个图片里定一个虚拟目标，网络输出用于其他图片
+ 两个分支，限制一个，让另一个可微
+ 在indoor（depth 来自3D传感器）和outdoor（depth来自SfM解的估计） datasets下训练

1. 无监督学习
2. 60fps QVGA图片（240×320）

#### ["GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.pdf) 2018 ECCV

#### feature tracking
#### ["Integration of the 3D Environment for UAV Onboard Visual Object Tracking"](https://arxiv.org/pdf/2008.02834v3.pdf) 2020 Aug

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

#### ["Robust Line Segments Matching via Graph Convolution Networks"](https://arxiv.org/pdf/2004.04993v2.pdf) 2020 Apr

<a name="outlier"></a>
### outlier
#### ["Efficient Outlier Removal in Large Scale Global Structure-from-Motion"](https://arxiv.org/pdf/1808.03041v4.pdf) 2018 Aug

<a name="bundle-adjustment"></a>
### bundle adjustment
#### ["RPBA -- Robust Parallel Bundle Adjustment Based on Covariance Information"](https://arxiv.org/pdf/1910.08138v1.pdf) 2019 Oct

#### ["BA-Net: Dense Bundle Adjustment Network"](https://arxiv.org/pdf/1806.04807v3.pdf) 2018 Jun

dense SfM，基于feature度量的BA

# 多视图几何约束 -> feature-metric error
# depth parameterization 恢复深度
# image -> 几个basis depth maps -> 线性组合 -> final depth （via feature-metric BA）

<a name="localization"></a>
### localization
#### ["Reference Pose Generation for Long-term Visual Localization via Learned Features and View Synthesis"](https://arxiv.org/pdf/2005.05179v3.pdf) 2020 May

#### ["Cascaded Parallel Filtering for Memory-Efficient Image-Based Localization"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Cascaded_Parallel_Filtering_for_Memory-Efficient_Image-Based_Localization_ICCV_2019_paper.pdf) 2019 ICCV

<a name="calibration"></a>
### calibration
#### ["Infrastructure-based Multi-Camera Calibration using Radial Projections"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610324.pdf) 2020 ECCV

<a name="motion"></a>
### motion
#### ["Relative Pose from Deep Learned Depth and a Single Affine Correspondence"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf) 2020 ECCV

#### ["Resultant Based Incremental Recovery of Camera Pose from Pairwise Matches"](https://arxiv.org/pdf/1901.09364v1.pdf) 2019 Jan

#### ["Flow-Motion and Depth Network for Monocular Stereo and Beyond"](https://arxiv.org/pdf/1909.05452v1.pdf) 2019 Sep

#### ["Trifocal Relative Pose from Lines at Points and its Efficient Solution"](https://arxiv.org/pdf/1903.09755v3.pdf) 2019 Mar

<a name="non-rigid"></a>
### non-rigid
#### ["Deep NRSfM++: Towards 3D Reconstruction in the Wild"](https://arxiv.org/pdf/2001.10090v1.pdf) 2020 Jan

#### ["DefSLAM: Tracking and Mapping of Deforming Scenes from Monocular Sequences"](https://arxiv.org/pdf/1908.08918v2.pdf) 2019 Aug

#### ["Deep Interpretable Non-Rigid Structure from Motion"](https://arxiv.org/pdf/1902.10840v1.pdf) 2019 Feb

#### ["Structure from Recurrent Motion: From Rigidity to Recurrency"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Structure_From_Recurrent_CVPR_2018_paper.pdf) 2018 CVPR

<a name="distortion"></a>
### distortion
#### ["Tangent Images for Mitigating Spherical Distortion"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Eder_Tangent_Images_for_Mitigating_Spherical_Distortion_CVPR_2020_paper.pdf) 2020 CVPR

<a name="parallel"></a>
### parallel
#### ["Parallel Structure from Motion from Local Increment to Global Averaging"](https://arxiv.org/pdf/1702.08601v3.pdf) 2017 Feb

<a name="disambiguation"></a>
### disambiguation
#### ["Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context"](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.pdf) 2017 CVPR

<a name="camera-model"></a>
### camera model
#### ["Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption"](https://arxiv.org/pdf/2011.03106v1.pdf) 2020 Nov

#### ["Uncertainty Based Camera Model Selection"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Polic_Uncertainty_Based_Camera_Model_Selection_CVPR_2020_paper.pdf) 2020 CVPR

<a name="segmentation"></a>
### segmentation
#### ["Three-dimensional Segmentation of Trees Through a Flexible Multi-Class Graph Cut Algorithm (MCGC)"](https://arxiv.org/pdf/1903.08481v1.pdf) 2019 Mar

<a name="fundamental-matrix"></a>
### fundamental matrix
#### ["GPSfM: Global Projective SFM Using Algebraic Constraints on Multi-View Fundamental Matrices"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kasten_GPSfM_Global_Projective_SFM_Using_Algebraic_Constraints_on_Multi-View_Fundamental_CVPR_2019_paper.pdf) 2019 CVPR

<a name="factorization"></a>
### factorization
#### ["Trust No One: Low Rank Matrix Factorization Using Hierarchical RANSAC"](http://openaccess.thecvf.com/content_cvpr_2016/papers/Oskarsson_Trust_No_One_CVPR_2016_paper.pdf) 2016 CVPR

<a name="optimization"></a>
### optimization
#### ["A Unified Optimization Framework for Low-Rank Inducing Penalties"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ornhag_A_Unified_Optimization_Framework_for_Low-Rank_Inducing_Penalties_CVPR_2020_paper.pdf) 2020 CVPR

<a name="tips"></a>
### tips
#### ["Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media with Airlight and Scattering Coefficient Estimation"](https://arxiv.org/pdf/2011.09114v1.pdf) 2020 Nov

#### ["Image Matching across Wide Baselines: From Paper to Practice"](https://arxiv.org/pdf/2003.01587v3.pdf) 2020 Mar

#### ["Leveraging Photogrammetric Mesh Models for Aerial-Ground Feature Point Matching Toward Integrated 3D Reconstruction"](https://arxiv.org/pdf/2002.09085v2.pdf) 2020 Feb

#### ["Robust SfM with Little Image Overlap"](https://arxiv.org/pdf/1703.07957v2.pdf) 2017 Mar

