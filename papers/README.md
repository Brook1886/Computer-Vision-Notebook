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

["Towards Linear-time Incremental Structure from Motion"](http://ccwu.me/vsfm/vsfm.pdf), 2013 3DV

<a name="global_SfM"></a>
## global SfM

["Global Structure-from-Motion by Similarity Averaging"](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Cui_Global_Structure-From-Motion_by_ICCV_2015_paper.pdf) 2015 ICCV
    
["Robust Camera Location Estimation by Convex Programming"](https://web.math.princeton.edu/~amits/publications/CVPR2015-SfM-Published.pdf) 2015 IEEE

["Robust Global Translations with 1DSfM"](http://www.cs.cornell.edu/projects/1dsfm/docs/1DSfM_ECCV14.pdf) 2014 ECCV
   
["Efficient and Robust Large-Scale Rotation Averaging"](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Chatterjee_Efficient_and_Robust_2013_ICCV_paper.pdf) 2013 ICCV

<a name="hierarchical_SfM"></a>
## hierarchical SfM

["Hierarchical structure-and-motion recovery from uncalibrated images"](https://arxiv.org/pdf/1506.00395) 2015 Jun

<a name="multi-stage_SfM"></a>
## multi-stage SfM

["HSfM: Hybrid Structure-from-Motion"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cui_HSfM_Hybrid_Structure-from-Motion_CVPR_2017_paper.pdf) 2017 ICCV

## 基于学习 SfM

["Supervising the new with the old: learning SFM from SFM"]() 2018

<a name="graph-based"></a>
## graph-based

["NeuRoRA: Neural Robust Rotation Averaging"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690137.pdf) 2020 ECCV 

["Graph-Based Parallel Large Scale Structure from Motion"](https://arxiv.org/pdf/1912.10659v2.pdf) 2019 Dec

["GraphMatch: Efficient Large-Scale Graph Construction for Structure from Motion"](https://arxiv.org/pdf/1710.01602) 2017 Oct

["Graph-Based Consistent Matching for Structure-from-Motion"](http://www.eccv2016.org/files/posters/P-2A-19.pdf) 2016 ECCV

<a name="factor-graph"></a>
## factor graph

["miniSAM: A Flexible Factor Graph Non-linear Least Squares Optimization Framework"](https://arxiv.org/pdf/1909.00903v1.pdf) 2019 Sep

<a name="depth"></a>
## depth

["SeasonDepth: Cross-Season Monocular Depth Prediction Dataset and Benchmark under Multiple Environments"](https://arxiv.org/pdf/2011.04408v1.pdf) 2020 Nov

["RidgeSfM: Structure from Motion via Robust Pairwise Matching Under Depth Uncertainty"](https://arxiv.org/pdf/2011.10359v1.pdf) 2020 Nov

["DeepV2D: Video to Depth with Differentiable Structure from Motion"](https://openreview.net/pdf?id=HJeO7RNKPr) 2020 ICLR

["Consistent Video Depth Estimation"](https://arxiv.org/pdf/2004.15021.pdf) 2020 Aug

["Self-Supervised 3D Keypoint Learning for Ego-motion Estimation"](https://arxiv.org/pdf/1912.03426v3.pdf) 2019 Dec

["Single Image Depth Estimation Trained via Depth from Defocus Cues"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gur_Single_Image_Depth_Estimation_Trained_via_Depth_From_Defocus_Cues_CVPR_2019_paper.pdf) 2019 CVPR

["MegaDepth: Learning Single-View Depth Prediction from Internet Photos"](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.pdf) 2018 CVPR

["Geometry meets semantics for semi-supervised monocular depth estimation"](https://arxiv.org/pdf/1810.04093v2.pdf) 2018 Oct

["Digging into self-supervised monocular depth estimation"](https://arxiv.org/pdf/1806.01260.pdf) 2019

["Unsupervised monocular depth estimation with left-right consistency"]() 2016

["Deeper depth prediction with fully convolutional residual networks"]() 2016

["Semi-supervised deep learning for monocular depth map prediction"]() 2017

["Unsupervised learning of depth and ego-motion from video"](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf) 2017

["Depth map prediction from a single image using a multi-scale deep network"]() 2014

["Depth and surface normal estimation from monocular images using regression on deep features and hierarchical CRFs"]() 2015

["Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture"]() 2015

["Depth from videos in the wild: Unsupervised monocular depth learning from unknown cameras"]() 2019

["Unsupervised learning of depth and ego-motion from monocular video using 3d geometric constraints"]() 2018

["Learning depth from monocular videos using direct methods"]() 2018

["GeoNet: Unsupervised learning of dense depth, optical flow and camera pose"]() 2018

["Unsupervised learning of monocular depth estimation and visual odometry with deep feature reconstruction"]() 2018

["Spatial transformer networks"]() 2017

["Neural RGB→D Sensing: Depth and Uncertainty from a Video Camera"](https://arxiv.org/abs/1901.02571) 2019

## uncertainty estimation

["What uncertainties do we need in bayesian deep learning for computer vision?"]() 2017

["Multi-task learning using uncertainty to weigh losses for scene geometry and semantics"]() 2018

<a name="feature"></a>
## feature

["SEKD: Self-Evolving Keypoint Detection and Description"](https://arxiv.org/pdf/2006.05077v1.pdf) 2020 Jun

["Neural Outlier Rejection for Self-Supervised Keypoint Learning"](https://openreview.net/pdf?id=Skx82ySYPH) 2020 ICLR

["Multi-View Optimization of Local Feature Geometry"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460647.pdf) 2020 ECCV

["LiFF: Light Field Features in Scale and Depth"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dansereau_LiFF_Light_Field_Features_in_Scale_and_Depth_CVPR_2019_paper.pdf) 2019 CVPR

["LandscapeAR: Large Scale Outdoor Augmented Reality by Matching Photographs with Terrain Models Using Learned Descriptors"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf) 2020 ECCV

["LF-Net: Learning Local Features from Images"](http://papers.nips.cc/paper/7861-lf-net-learning-local-features-from-images.pdf) 2018 NeurIPS

["GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.pdf) 2018 ECCV

["Integration of the 3D Environment for UAV Onboard Visual Object Tracking"](https://arxiv.org/pdf/2008.02834v3.pdf) 2020 Aug

["Fast connected components computation in large graphs by vertex pruning"](http://for.unipi.it/alessandro_lulli/files/2015/07/J002_FastConnectedComponentsComputationInLargeGraphsByVertexPruning.pdf) 2016 Jul

["ENFT: Efficient Non-Consecutive Feature Tracking for Robust Structure-from-Motion"](https://arxiv.org/pdf/1510.08012v2.pdf) 2015 Oct

["AdaLAM: Revisiting Handcrafted Outlier Detection"](https://arxiv.org/pdf/2006.04250v1.pdf) 2020 Jun

["Robust Line Segments Matching via Graph Convolution Networks"](https://arxiv.org/pdf/2004.04993v2.pdf) 2020 Apr

["Efficient Outlier Removal in Large Scale Global Structure-from-Motion"](https://arxiv.org/pdf/1808.03041v4.pdf) 2018 Aug

<a name="bundle-adjustment"></a>
## bundle adjustment

["DeepSFM: Structure From Motion Via Deep Bundle Adjustment"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460222.pdf) 2020 ECCV

["RPBA -- Robust Parallel Bundle Adjustment Based on Covariance Information"](https://arxiv.org/pdf/1910.08138v1.pdf) 2019 Oct

["BA-Net: Dense Bundle Adjustment Network"](https://arxiv.org/pdf/1806.04807v3.pdf) 2018 Jun

["BAD SLAM: Bundle adjusted direct RGB-D SLAM"]() 2019

<a name="localization"></a>
## localization

["Reference Pose Generation for Long-term Visual Localization via Learned Features and View Synthesis"](https://arxiv.org/pdf/2005.05179v3.pdf) 2020 May

["Cascaded Parallel Filtering for Memory-Efficient Image-Based Localization"](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cheng_Cascaded_Parallel_Filtering_for_Memory-Efficient_Image-Based_Localization_ICCV_2019_paper.pdf) 2019 ICCV

<a name="calibration"></a>
## calibration

["Calibration-free Structure-from-Motion with Calibrated Radial Trifocal Tensors"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123500375.pdf) 2020 ECCV

["Infrastructure-based Multi-Camera Calibration using Radial Projections"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123610324.pdf) 2020 ECCV

<a name="motion"></a>
## motion

["Relative Pose from Deep Learned Depth and a Single Affine Correspondence"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570613.pdf) 2020 ECCV

["Resultant Based Incremental Recovery of Camera Pose from Pairwise Matches"](https://arxiv.org/pdf/1901.09364v1.pdf) 2019 Jan

["Flow-Motion and Depth Network for Monocular Stereo and Beyond"](https://arxiv.org/pdf/1909.05452v1.pdf) 2019 Sep

["Trifocal Relative Pose from Lines at Points and its Efficient Solution"](https://arxiv.org/pdf/1903.09755v3.pdf) 2019 Mar

["DeepVO: Towards end-to-end visual odometry with deep recurrent convolutional neural networks"]() 2017

["UnDeepVO: Monocular visual odometry through unsupervised deep learning"]() 2017

<a name="non-rigid"></a>
## non-rigid

["Deep NRSfM++: Towards 3D Reconstruction in the Wild"](https://arxiv.org/pdf/2001.10090v1.pdf) 2020 Jan

["C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion"](https://arxiv.org/pdf/1909.02533.pdf) 2019 Oct

["DefSLAM: Tracking and Mapping of Deforming Scenes from Monocular Sequences"](https://arxiv.org/pdf/1908.08918v2.pdf) 2019 Aug

["Deep Interpretable Non-Rigid Structure from Motion"](https://arxiv.org/pdf/1902.10840v1.pdf) 2019 Feb

["Structure from Recurrent Motion: From Rigidity to Recurrency"](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Structure_From_Recurrent_CVPR_2018_paper.pdf) 2018 CVPR

<a name="distortion"></a>
## distortion

["Tangent Images for Mitigating Spherical Distortion"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Eder_Tangent_Images_for_Mitigating_Spherical_Distortion_CVPR_2020_paper.pdf) 2020 CVPR

<a name="parallel"></a>
## parallel

["Parallel Structure from Motion from Local Increment to Global Averaging"](https://arxiv.org/pdf/1702.08601v3.pdf) 2017 Feb

<a name="disambiguation"></a>
## disambiguation

["Distinguishing the Indistinguishable: Exploring Structural Ambiguities via Geodesic Context"](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.pdf) 2017 CVPR

<a name="camera-model"></a>
## camera model

["Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption"](https://arxiv.org/pdf/2011.03106v1.pdf) 2020 Nov

["Uncertainty Based Camera Model Selection"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Polic_Uncertainty_Based_Camera_Model_Selection_CVPR_2020_paper.pdf) 2020 CVPR

<a name="segmentation"></a>
## segmentation

["Three-dimensional Segmentation of Trees Through a Flexible Multi-Class Graph Cut Algorithm (MCGC)"](https://arxiv.org/pdf/1903.08481v1.pdf) 2019 Mar

<a name="fundamental-matrix"></a>
## fundamental matrix

["GPSfM: Global Projective SFM Using Algebraic Constraints on Multi-View Fundamental Matrices"](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kasten_GPSfM_Global_Projective_SFM_Using_Algebraic_Constraints_on_Multi-View_Fundamental_CVPR_2019_paper.pdf) 2019 CVPR

<a name="factorization"></a>
## factorization

["Trust No One: Low Rank Matrix Factorization Using Hierarchical RANSAC"](http://openaccess.thecvf.com/content_cvpr_2016/papers/Oskarsson_Trust_No_One_CVPR_2016_paper.pdf) 2016 CVPR

<a name="optimization"></a>
## optimization

["A Unified Optimization Framework for Low-Rank Inducing Penalties"](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ornhag_A_Unified_Optimization_Framework_for_Low-Rank_Inducing_Penalties_CVPR_2020_paper.pdf) 2020 CVPR

<a name="mesh"></a>
## mesh

["Meshlet Priors for 3D Mesh Reconstruction"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Badki_Meshlet_Priors_for_3D_Mesh_Reconstruction_CVPR_2020_paper.pdf) 2020 CVPR

<a name="stereo"></a>
## stereo

["Multi-Frame GAN: Image Enhancement for Stereo Visual Odometry in Low Light"]() 2019 

["DPSNET: END-TO-END DEEP PLANE SWEEP STEREO"](https://openreview.net/pdf?id=ryeYHi0ctQ) 2019 ICLR

["DeMoN: Depth and Motion Network for Learning Monocular Stereo"](https://arxiv.org/pdf/1612.02401) 2017 CVPR

<a name="privacy"></a>
## privacy

["Privacy Preserving Structure-from-Motion"](https://www.microsoft.com/en-us/research/uploads/prod/2020/08/Geppert2020ECCV-1.pdf) 2020 ECCV

<a name="tips"></a>
## tips

["Dehazing Cost Volume for Deep Multi-view Stereo in Scattering Media with Airlight and Scattering Coefficient Estimation"](https://arxiv.org/pdf/2011.09114v1.pdf) 2020 Nov

["Image Matching across Wide Baselines: From Paper to Practice"](https://arxiv.org/pdf/2003.01587v3.pdf) 2020 Mar

["Leveraging Photogrammetric Mesh Models for Aerial-Ground Feature Point Matching Toward Integrated 3D Reconstruction"](https://arxiv.org/pdf/2002.09085v2.pdf) 2020 Feb

["Robust SfM with Little Image Overlap"](https://arxiv.org/pdf/1703.07957v2.pdf) 2017 Mar

<a name="Multi-View_Stereo"></a>
## Multi-View Stereo

["TAPA-MVS: Textureless-Aware PAtchMatch Multi-View Stereo"](https://openaccess.thecvf.com/content_ICCV_2019/papers/Romanoni_TAPA-MVS_Textureless-Aware_PAtchMatch_Multi-View_Stereo_ICCV_2019_paper.pdf) 2019 ICCV

["Scalable Surface Reconstruction from Point Clouds with Extreme Scale and Density Diversity"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mostegel_Scalable_Surface_Reconstruction_CVPR_2017_paper.pdf) 2017 CVPR

<a name="SLAM"></a>
## SLAM

## 增大视野

["OmniSLAM: Omnidirectional Localization and Dense Mapping for Wide-baseline Multi-camera Systems"](https://arxiv.org/pdf/2003.08056v1.pdf) 2020 ICRA

["Sweepnet: Wide-baseline omnidirectional depth estimation"]() 2019 ICRA

["Rovo: Robust omnidirectional visual odometry for wide-baseline wide-fov camera systems"]() 2019 ICRA

["Omnimvs: End-to-end learning for omnidirectional stereo matching"]() 2019

## LiDAR 

["Tightly coupled 3d lidar inertial odometry and mapping"]() 2019 ICRA

## 结构光

["Kinectfusion: Real-time dense surface mapping and tracking"]() 2011 ISMAR

## 稠密建图

["Efficient large-scale stereo matching"]() 2010 ACCV

["Stereoscan: Dense 3d reconstruction in real-time"]() 2011 IEEE

## 深度估计

["Ga-net: Guided aggregation net for end-to-end stereo matching"]() 2019 CVPR

["Pyramid stereo matching network"]() 2018 CVPR

["Occlusions, motion and depth boundaries with a generic network for disparity, optical flow or scene flow estimation"]() 2018 ECCV

## 特征提取

["Orb: An efficient alternative to sift or surf"]() 2011 ICCV

## 特征点法

["Real-time monocular SLAM: Why filter?"]() 2010

["ORB-SLAM: a versatile and accurate monocular slam system"]() 2015 IEEE

["ORB-SLAM2: an Open-Source SLAM System forMonocular, Stereo and RGB-D Cameras"](https://arxiv.org/pdf/1610.06475.pdf) 2016 Oct

["ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM"](https://arxiv.org/pdf/2007.11898.pdf) 2020 Jul

## 直接法 LSD

["Semi-Dense Visual Odometry for a Monocular Camera"](https://openaccess.thecvf.com/content_iccv_2013/papers/Engel_Semi-dense_Visual_Odometry_2013_ICCV_paper.pdf) 2013 ICCV

["LSD-SLAM: Large-Scale Direct Monocular SLAM"](https://jakobengel.github.io/pdf/engel14eccv.pdf) 2014 ECCV

## 直接法 DSO

["Omnidirectional DSO: Direct Sparse Odometry with Fisheye Cameras"]() 2018

["Online Photometric Calibration of Auto Exposure Video for Realtime Visual Odometry and SLAM"]() 2018

["D3VO: Deep Depth, Deep Pose and Deep Uncertainty for Monocular Visual Odometry"](https://arxiv.org/pdf/2003.01060.pdf) 2020

["Rolling-Shutter Modelling for Visual-Inertial Odometry"]() 2019

["Direct Sparse Odometry With Rolling Shutter"]() 2018

["Deep Virtual Stereo Odometry: Leveraging Deep Depth Prediction for Monocular Direct Sparse Odometry"]() 2018

["LDSO: Direct Sparse Odometry with Loop Closure"]() 2018

["Direct Sparse Visual-Inertial Odometry using Dynamic Marginalization"](https://arxiv.org/pdf/1804.05625.pdf) 2018

["Stereo DSO: Large-Scale Direct Sparse Visual Odometry with Stereo Cameras"]() 2017

["Direct Sparse Odometry"](https://arxiv.org/pdf/1607.02565.pdf) 2016

["A Photometrically Calibrated Benchmark For Monocular Visual Odometry"]() 2016

## 直接法 RGB-D

["Dense visual SLAM for RGB-D cameras"]() 2013

## 视觉+IMU

["Keyframe-based visual-inertial odometry using nonlinear optimization"]() 2015

["Visual-inertial monocular SLAM with map reuse"]() 2017

[" VINS-Mono: A robust and versatile monocular visual-inertial state estimator"]() 2018

## 半直接法 SVO

["CNN-SVO: Improving the mapping in semi-direct visual odometry using singleimage depth prediction"]() 2019

## 深度学习

["CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction."]() 2017

["Scale recovery for monocular visual odometry using depth estimated with deep convolutional neural fields"]() 2017

["Visual odometry revisited: What should be learnt?"]() 2019

["GCNv2: Efficient correspondence prediction for real-time slam"]() 2019

["CodeSLAM-learning a compact, optimisable representation for dense visual SLAM"]() 2018

## visual odometry

["Self-improving visual odometry"]() 2018

<a name="robot_grasp"></a>
## robot grasp

["Dex-Net 2.0"]() 2017

