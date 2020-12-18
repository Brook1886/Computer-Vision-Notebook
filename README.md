# Notebook

关于SLAM、SfM、MVS学习的记录，持续更新

# contents
- [SLAM](./SLAM/README.md/#SLAM)
    - [note](./SLAM/README.md/#note)
    - [papers](./SLAM/README.md/#papers)
        - [ORB-SLAM](./SLAM/README.md/#ORB-SLAM)
- [SfM](./SfM/README.md/#SfM)
    - [papers](./SfM/README.md/#papers)
        - [incremental ](./SfM/README.md/#incremental)
        - [global](./SfM/README.md/#global)
        - [hierarchical](./SfM/README.md/#hierarchical)
        - [multi-stage](./SfM/README.md/#multi-stage)
        - [graph-based](./SfM/README.md/#graph-based)
        - [factor graph](./SfM/README.md/#factor-graph)
        - [depth](./SfM/README.md/#depth)
        - [feature](./SfM/README.md/#feature)
        - [outlier](./SfM/README.md/#outlier)
        - [bundle adjustment](./SfM/README.md/#bundle-adjustment)
        - [localization](./SfM/README.md/#localization)
        - [calibration](./SfM/README.md/#calibration)
        - [motion](./SfM/README.md/#motion)
        - [non-rigid](./SfM/README.md/#non-rigid)
        - [distortion](./SfM/README.md/#distortion)
        - [parallel](./SfM/README.md/#parallel)
        - [disambiguation](./SfM/README.md/#disambiguation)
        - [camera model](./SfM/README.md/#camera-model)
        - [segmentation](./SfM/README.md/#segmentation)
        - [fundamental matrix](./SfM/README.md/#fundamental-matrix)
        - [factorization](./SfM/README.md/#factorization)
        - [optimization](./SfM/README.md/#optimization)
        - [mesh](./SfM/README.md/#mesh)  
        - [stereo](./SfM/README.md/#stereo)  
        - [privacy](./SfM/README.md/#privacy)
        - [tips](./SfM/README.md/#tips)
- [MVS](./MVS/README.md/#MVS)
    - [papers](./MVS/README.md/#papers)
        - [point cloud](./MVS/README.md/#point-cloud)
        - [surface](./MVS/README.md/#surface)

<a name="MVS"></a>
## MVS

<a name="point-cloud"></a>
### point cloud

#### ["TAPA-MVS: Textureless-Aware PAtchMatch Multi-View Stereo"](https://openaccess.thecvf.com/content_ICCV_2019/papers/Romanoni_TAPA-MVS_Textureless-Aware_PAtchMatch_Multi-View_Stereo_ICCV_2019_paper.pdf) 2019 ICCV

估计每个 view 的 depth map、normal map（法线图）

通过基于 patch match 的优化

> photo-consistency 光度一致性，重建模型缺乏完整性、使其不可靠

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

<a name="surface"></a>
### surface

#### ["Scalable Surface Reconstruction from Point Clouds with Extreme Scale and Density Diversity"](https://openaccess.thecvf.com/content_cvpr_2017/papers/Mostegel_Scalable_Surface_Reconstruction_CVPR_2017_paper.pdf) 2017 CVPR

+ a combiantion of: 1) octree data partitioning + 2) delaunay tetrahedralization + 3) graph cut

> 现有的 mulit-scale surface reconstruction focus on：1）局部尺度变化；2）获取封闭网络，基于全局

