# SLAM introduction

+ 类型

    激光传感器、视觉、微惯性导航组合IMU

    + 基于扩展卡尔曼滤波器的SLAM（EKF SLAM）

        计算复杂、滤波精度不高

    + 基于Rao-Blackwellized粒子滤波器的SLAM（RBPF SLAM）

        FastSLAM：融合EKF和粒子滤波器的优点，计算复杂度降低、更鲁棒

        Gmapping：基于栅格地图的RBPF SLAM

    + 基于图优化的SLAM（Graph SLAM）

        全局视角

        iSAM（incremental Smoothing and Mapping）：前一步雅可比矩阵，增量更新当前雅可比矩阵

        Unscented iSAM, Hog-Man, Toro

        G2O框架

    + 基于视觉的SLAM（vSLAM）

        1）前端，视觉里程计VO，visual odometry；2）后端，闭环优化，loop closure

        + 基于特征点

            特征点提取：SIFT，SURF，FAST，ORB

            MonoSLAM

            PTAM

            ORB-SLAM2

        + 基于直接法

            直接对图像的像素光度进行操作

            DVO，RGB-D传感器、t分布

            LSD-SLAM，直接图像配准方法、基于滤波的半稠密深度地图估计方法

            DSO，稀疏点

        + 特征点+直接法

            SVO，半直接，显式构建野值测量模型的概率构图方法，成功应用于无人机

            LIBVISO2 + LSD-SLAM

    + 基于视觉和IMU信息的SLAM（viSLAM)

        前端，VIO，visual inertial odometry；

        作用：辅助两帧图像完成特征跟踪、参数优化时提供参数约束；

        方式：1）松耦合，分别用视觉、IMU信息估计相机运动，再融合运动，计算量小，但没考虑信息之间的联系，精度受限；2）紧耦合，联合视觉、IMU信息，建立运动方程、观测方程，计算量大，但是精度高

        GAFD，Gravity Aligned Feature Descriptors，重力方向和特征点方向之差来辅助特征匹配

        VI ORB-SLAM，增加IMU预积分

    + 动态场景下的SLAM

        RDSLAM，浙大，能应对场景的结构、颜色变化，基于时序先验的自适应RANSAC方法，基于RGBD数据使用K均值聚类给场景中的物体分类（静态或动态）

        闭环检测、基于多种特征、语义...

+ 难点：

    经典SLAM:

    维数爆炸、计算复杂度（如何满足实时性）、数据关联（观测量与已知特征信息匹配）、噪声、动态环境、“绑架”问题、粒子退化问题

    vSLAM:

    实时性、累积误差

    + 特征检测

        鲁棒 + 重现 + 显著 + 高效 + 准确

    + 帧间配准

        对应 -> 相对变换 -> 同一坐标系

        模型帧，数据帧，位姿估计

        算法：随机采样一致性算法RANSAC，迭代最近点算法ICP

        视觉里程计VO，基于光度一致性的视觉里程计DVO，基于Kinect的视觉里程计FOVIS，帧到模型的视觉里程计FVO

    + 闭环检测

        实质上是一种检测观测数据相似性的算法

        图优化、词袋模型（Bag-of-Words, BoW，视觉特征聚类后建map）、转为分类问题

        难点：对准确率要求非常高，几乎100%，否则会有歧义；数据规模大，影响实时性

    + 地图构建

        类型：

       + 度量地图（Metric Map）

            强调能准确表示地图中物体的位置关系

            分为稀疏（路标）、稠密（占据网格，grid），现主要用稠密

            占据网格地图可直接用于导航算法

            二维：矩阵；三维：八叉树

            格点有占据、空闲、未知三种状态

            耗费存储，有些细节无用，对误差敏感

        + 拓扑地图（Topological Map)

            map是一个Graph，地图元素-节点，元素之间关系-边

        + 语义地图（Semantic Map）

            地图元素具有含义，便于识别

        + 混合地图（Hybrid Map)

            核心思想：小范围度量、大范围拓扑

        表示方法：

        + 点云地图（Point Cloud Map)

        + 高程地图（Elevation Map)

        + 立体占用地图（Volumetric Occupancy Map)

            Octomap

+ 方向：
    
    直接法和特征法结合、IMU、动态场景、语义、长期SLAM、多机器人
