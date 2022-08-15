/* 徐工研究院地面边线提取.
 * 首先对前方3个雷达的点云与rtk_odom进行时间同步
 */

#include <ros/ros.h>
#include <stdint.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tf/tf.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// PCL specific includes
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "ctime"

#include <fstream>
#include <cmath>


#define PointType pcl::PointXYZ
#define  PI  3.1415926535
using namespace std;
using namespace Eigen;

class BoundaryDetector {
private:
    ros::NodeHandle nh;

    ros::Subscriber sub_lidar, sub_livox_left, sub_livox_right, sub_odom;
    ros::Publisher pub_final;
    ros::Publisher pub_total;
    ros::Publisher pub_debug;

    bool savePCD;
    bool readPCD;

    int frame_cnt;
    int Cnt;  //累加的帧数
    double X_min;
    double ds_size;
    double ransac_th;
    double cluster_r;

    pcl::PointCloud <PointType> cloud_filtered; // 直通滤波用
    pcl::PointCloud<PointType>::Ptr pointcloud_adder; // 点云累加
    pcl::PointCloud <PointType> laserCloud_ds;

    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_; //雷达订阅
    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_left_; //左补盲雷达订阅
    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_right_; //右补盲雷达订阅
    message_filters::Subscriber <nav_msgs::Odometry> sub_odom_; //rtk的odom订阅
    typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;
    typedef message_filters::Synchronizer <MySyncPolicy> Sync;
    boost::shared_ptr <Sync> sync_;//时间同步器

    geometry_msgs::PoseStamped last_pose;
    bool first_frame;

    pcl::ModelCoefficients::Ptr coefficients; // ransac用
    pcl::PointIndices::Ptr inliers;
    pcl::PointCloud <PointType> laserCloudGround;

    pcl::search::KdTree<PointType>::Ptr tree;  // 聚类用
    pcl::PointCloud <pcl::PointXYZ> largestCluster; // 最大聚类
    pcl::PointXYZ minPt, maxPt;
    PointType last_left_point_oneSlice, last_right_point_oneSlice;  // 上一个横条中的左右边沿点。便于比较当前帧是否可以接受

    pcl::PointCloud <pcl::PointXYZ> boundary_point_left_local; // 算法计算出的路面边缘点 (3帧的)
    pcl::PointCloud <pcl::PointXYZ> boundary_point_right_local; // 算法计算出的路面边缘点(3帧的)
    pcl::PointCloud <pcl::PointXYZ> boundary_left_inMap; // 在odom或map下的坐标(3帧的)
    pcl::PointCloud <pcl::PointXYZ> boundary_right_inMap; // 在odom或map下的坐标(3帧的)

    pcl::PointCloud <pcl::PointXYZ> left_Pts_added; // 各帧的左边沿点叠加在一起
    pcl::PointCloud <pcl::PointXYZ> right_Pts_added; // 各帧的右边沿点叠加在一起

    std_msgs::Header cloud_header;
    tf2_ros::TransformBroadcaster br3; // 发布当前时刻位姿在各个坐标系中的变换

    pcl::PointCloud<pcl::PointXYZ>::Ptr history_cloud; //存放历史边界点的点云
    pcl::PCDReader reader;    //定义点云读取对象
    pcl::PCDWriter writer;
    ros::Timer timer;

public:
    BoundaryDetector() :
            nh(),
            frame_cnt(0),
            first_frame(true)
    {
        coefficients.reset(new pcl::ModelCoefficients);
        inliers.reset(new pcl::PointIndices);
        tree.reset(new pcl::search::KdTree <PointType>);
        pointcloud_adder.reset(new pcl::PointCloud <PointType>);  // 初始化
        history_cloud.reset(new pcl::PointCloud <PointType>);

        nh.param("Cnt",Cnt,4);
        nh.param("X_min", X_min, 20.0);  // 点云x轴直通滤波的起点. 从baselink前方13米处开始，避开盲区。太小虚假则虚假边界点多
        nh.param("ds_size", ds_size, 0.3);  // default: 0.3
        nh.param("ransac_th", ransac_th, 0.25);
        nh.param("cluster_r", cluster_r, 0.4);

        nh.param("save_pcd", savePCD, false);
        nh.param("read_pcd", readPCD, false);
        if (readPCD == true)
        {
            // 加载左历史边沿点数据
            if (reader.read("boundary.pcd", *history_cloud) < 0)   // 默认在 ~/.ros目录下，即$ROS_HOMES
            {
                PCL_ERROR("\a->边沿点文件不存在，创建新文件！\n");
                ofstream fout("boundary.pcd");
                fout.close();
                reader.read("boundary.pcd", *history_cloud); // 创建完，再次打开该文件
                std::cout << " File boundary.pcd opened " << std::endl;
            } else {
                std::cout << " File boundary.pcd opened " << std::endl;
            }
        }
        if (savePCD == true)
        {
            // 创建定时器, 每隔20s保存更新的边界点
            timer = nh.createTimer(ros::Duration(20.0), &BoundaryDetector::timerCallBack, this);
            timer.start();
        }

        sub_lidar_.subscribe(nh, "/iv_pointsBaselink", 5);
        sub_lidar_left_.subscribe(nh, "/livox/lidar_left", 5);
        sub_lidar_right_.subscribe(nh, "/livox/lidar_right", 5);
        sub_odom_.subscribe(nh, "/gps/odom", 20); // topic名称改了
        pub_final = nh.advertise<sensor_msgs::PointCloud2>("/road_boundary", 10);
        pub_debug = nh.advertise<sensor_msgs::PointCloud2>("/debug", 10);  // 调试用
        pub_total = nh.advertise<sensor_msgs::PointCloud2>("/total", 10);  // 发布合并后的点云

        //回调
        sync_.reset(new Sync(MySyncPolicy(10), sub_lidar_, sub_lidar_left_, sub_lidar_right_, sub_odom_));
        sync_->registerCallback(boost::bind(&BoundaryDetector::callback, this, _1, _2, _3, _4));
    }

    ~BoundaryDetector() {}

    void callback(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg,
                  const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgLeft,
                  const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgRight,
                  const nav_msgs::Odometry::ConstPtr &odomMsg);

    void planeFitting(pcl::PointCloud <PointType> pc_filtered);

    void objectCluster(pcl::PointCloud<PointType>::ConstPtr pc_ground);

    void PickBoundary(pcl::PointCloud<PointType>::ConstPtr pc_cluster);

    void BoundarySparse(Eigen::Matrix4f transform_1);

    void timerCallBack(const ros::TimerEvent& e);
};


void BoundaryDetector::callback(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg,
                                const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgLeft,
                                const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgRight,
                                const nav_msgs::Odometry::ConstPtr &odomMsg)
{
//    cloud_header = laserCloudMsg->header;
    cloud_header = odomMsg->header;
//  将3个雷达点云的点云合为1个
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudMiddle(new pcl::PointCloud <PointType>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudLivoxLeft(new pcl::PointCloud <PointType>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudLivoxRight(new pcl::PointCloud <PointType>);
    pcl::fromROSMsg(*laserCloudMsg, *laserCloudMiddle);
    // 去掉点云NaN点
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudMiddle, laserCloudIn, indices);  // 去除NaN点， indices为保留点的索引

    pcl::fromROSMsg(*laserCloudMsgLeft, *laserCloudLivoxLeft);
    indices.clear();
    pcl::removeNaNFromPointCloud(*laserCloudLivoxLeft, *laserCloudLivoxLeft, indices);  // 去除NaN点
    laserCloudIn = laserCloudIn + *laserCloudLivoxLeft;  // 相加的点云要求字段相同

    pcl::fromROSMsg(*laserCloudMsgRight, *laserCloudLivoxRight);
    indices.clear();
    pcl::removeNaNFromPointCloud(*laserCloudLivoxRight, *laserCloudLivoxRight, indices);  // 去除NaN点
    laserCloudIn = laserCloudIn + *laserCloudLivoxRight; // 3个雷达点云叠加
    laserCloudMiddle->clear();
    laserCloudLivoxLeft->clear();
    laserCloudLivoxRight->clear();

    // 先直通滤波，减小后面的计算量
    pcl::PassThrough <PointType> pass;
    pass.setInputCloud(laserCloudIn.makeShared());
    pass.setFilterFieldName("x");
    // X_min小可以充分发挥补盲雷达的作用。
    pass.setFilterLimits(X_min, X_min+23.0);    // 向前取23m
    pass.filter(cloud_filtered);


// 把点云按照odom进行平移和旋转，再累加
    geometry_msgs::PoseStamped position_3d;
    position_3d.pose.position.x = odomMsg->pose.pose.position.x;
    position_3d.pose.position.y = odomMsg->pose.pose.position.y;
    position_3d.pose.position.z = odomMsg->pose.pose.position.z;
    position_3d.pose.orientation = odomMsg->pose.pose.orientation;
    if (first_frame == true) {
        last_pose = position_3d;  // 上电第0帧，只执行1次
        first_frame = false;
        return;
    } else {
        //定义变换矩阵，将点云平移
        Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
        float dx = position_3d.pose.position.x - last_pose.pose.position.x;  //x
        float dy = position_3d.pose.position.y - last_pose.pose.position.y;  //y
        float dz = position_3d.pose.position.z - last_pose.pose.position.z;  //z
//        Define a translation on the axis.
        transform_2.translation() << dx, dy, dz;

        //输入的四元数, 转化成欧拉角。提取原位姿旋转四元数的pitch和roll; yaw保持不变
        Eigen::Quaterniond q(odomMsg->pose.pose.orientation.w, odomMsg->pose.pose.orientation.x,
                             odomMsg->pose.pose.orientation.y, odomMsg->pose.pose.orientation.z); // 当前帧的位姿
//        Eigen::Quaterniond delta_q = q1.conjugate() * q; // 相对于第1帧的位姿旋转. 但是怎么保证上电时刻车辆位姿是理想状态呢？
        tf2::Quaternion dq(q.x(), q.y(), q.z(), q.w());  // 相对于map全局坐标系的位姿旋转
        tf2::Matrix3x3 m(dq);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);  // 先求yaw
        transform_2.rotate (Eigen::AngleAxisf (roll, Eigen::Vector3f::UnitX()));
        transform_2.rotate (Eigen::AngleAxisf (pitch, Eigen::Vector3f::UnitY()));
//        transform_2.rotate (Eigen::AngleAxisf (yaw, Eigen::Vector3f::UnitZ()));
//        std::cout << "roll pitch is " << roll*57.0 << ' ' << pitch*57.0 << std::endl;
        pcl::transformPointCloud(cloud_filtered, cloud_filtered, transform_2);
        last_pose = position_3d; // 保存当前位姿
    }

    *pointcloud_adder = *pointcloud_adder + cloud_filtered;  // 对连续几帧进行叠加
    frame_cnt++;

    if (frame_cnt == Cnt) {
//        // For debug：发布几帧叠加的点云
//        sensor_msgs::PointCloud2 pc_debug;
//        pcl::toROSMsg(*pointcloud_adder, pc_debug);
//        pc_debug.header.stamp = cloud_header.stamp;
//        pc_debug.header.frame_id = "base_link";
//        pub_debug.publish(pc_debug);   // topic name

        // RANSAC地面分割，得到地面点云和非地面点云
        planeFitting(*pointcloud_adder);

        // 下采样
        pcl::VoxelGrid <pcl::PointXYZ> filter;
        filter.setInputCloud(laserCloudGround.makeShared());
        // 设置体素栅格的大小
        filter.setLeafSize(ds_size, ds_size, ds_size);
        filter.filter(laserCloud_ds);

        // 发布所有的下采样后点云
//        sensor_msgs::PointCloud2 pc_total;
//        pcl::toROSMsg(laserCloud_ds, pc_total);
//        pc_total.header.stamp = cloud_header.stamp;
//        pc_total.header.frame_id = "base_link";
//        pub_total.publish(pc_total);   // topic name

        // 求出最大聚类
        objectCluster(laserCloud_ds.makeShared());

        // 根据横条形状，依次求每3帧在base_link下的边界点
        PickBoundary(largestCluster.makeShared());

        // 将两侧边缘点转换到map坐标系下
        // 定义变换矩阵
        Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
        transform_1(0, 3) = last_pose.pose.position.x; //x  如果用当前帧的话，偏差太大
        transform_1(1, 3) = last_pose.pose.position.y; //y
        transform_1(2, 3) = last_pose.pose.position.z; //z
        //输入的四元数 转化成旋转矩阵
        tf::Quaternion quaternion_1(last_pose.pose.orientation.x, last_pose.pose.orientation.y,
                                    last_pose.pose.orientation.z, last_pose.pose.orientation.w);//x,y,z,w
        tf::Matrix3x3 Matrix_1;
        Matrix_1.setRotation(quaternion_1);
        tf::Vector3 v1_1, v1_2, v1_3;
        v1_1 = Matrix_1[0];
        v1_2 = Matrix_1[1];
        v1_3 = Matrix_1[2];
        transform_1(0, 0) = v1_1[0];
        transform_1(0, 1) = v1_1[1];
        transform_1(0, 2) = v1_1[2];
        transform_1(1, 0) = v1_2[0];
        transform_1(1, 1) = v1_2[1];
        transform_1(1, 2) = v1_2[2];
        transform_1(2, 0) = v1_3[0];
        transform_1(2, 1) = v1_3[1];
        transform_1(2, 2) = v1_3[2];
//        std::cout<< "transform_1:\n" << transform_1 <<std::endl << std::endl;
        //输入的base_link下的边界点点云，把它们转化到map坐标系下
        pcl::transformPointCloud(boundary_point_left_local, boundary_left_inMap, transform_1);
        boundary_left_inMap.header.frame_id = "map";
        boundary_left_inMap.width = boundary_left_inMap.points.size();
        boundary_left_inMap.height = 1;
        boundary_left_inMap.is_dense = true;
        // 将各帧合成为一个点云
        left_Pts_added += boundary_left_inMap;
        left_Pts_added.header.frame_id = "map";
        left_Pts_added.width = left_Pts_added.points.size();
        left_Pts_added.height = 1;
        left_Pts_added.is_dense = true;

        pcl::transformPointCloud(boundary_point_right_local, boundary_right_inMap, transform_1);
        boundary_right_inMap.header.frame_id = "map";
        boundary_right_inMap.width = boundary_right_inMap.points.size();
        boundary_right_inMap.height = 1;
        boundary_right_inMap.is_dense = true;
        // 合成
        right_Pts_added += boundary_right_inMap;
        right_Pts_added.header.frame_id = "map";
        right_Pts_added.width = right_Pts_added.points.size();
        right_Pts_added.height = 1;
        right_Pts_added.is_dense = true;

        boundary_point_left_local.clear();  // 将每3帧融合后的边界点云清空（这是base_link下的）
        boundary_point_right_local.clear();
        boundary_left_inMap.clear();
        boundary_right_inMap.clear();

        // 根据车辆行进位姿进行边界点的提取和更新
        BoundarySparse(transform_1);

        frame_cnt = 0;  // 重新开始计数
        pointcloud_adder->clear();
    }
}

// ransac地面分割
void BoundaryDetector::planeFitting(pcl::PointCloud <PointType> pc_filtered) {
    // 创建分割对象
    pcl::SACSegmentation <PointType> seg;
    // 可选设置
    seg.setOptimizeCoefficients(true);
    seg.setMaxIterations(200);  // 循环次数
    // 必须设置
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ransac_th); // 从内点到拟合平面的距离 0.2
    seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
    seg.setEpsAngle(12.0 / 180.0 * PI);

    seg.setInputCloud(pc_filtered.makeShared());  //输入为点云指针
    seg.segment(*inliers, *coefficients);
    pcl::ExtractIndices <PointType> extract;
    if (inliers->indices.size() < 300)
    {
        ROS_INFO("Too small inliers in RANSAC model, skip!"); // 排除近处的密集点
        return;
    }
    else
    {
        extract.setInputCloud(pc_filtered.makeShared());
        extract.setIndices(inliers);
        extract.filter(laserCloudGround);  // 地面点云
//        std::cout << "Number of inlier points:" << laserCloudGround.points.size() << std::endl;
//        double ratio = double(laserCloudGround.points.size())/double(pc_filtered.points.size());
//        std::cout << "ratio is :" << ratio << std::endl;
//        extract.setNegative (true);
//        extract.filter (laserCloudNotGround); // 非地面点云
    }
}

// 找到地面点中的最大聚类
void BoundaryDetector::objectCluster(pcl::PointCloud<PointType>::ConstPtr pc_ground) {
    // 对非地面点云进行聚类
    tree->setInputCloud(pc_ground);
    std::vector <pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction <pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_r); //近邻搜索的搜索半径。需要多次调节 0.4
    ec.setMinClusterSize(1000);
//    ec.setMaxClusterSize(200000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(pc_ground);
    ec.extract(cluster_indices);
//    std::cout << "It has " << cluster_indices.size() << " planes by clustering" << endl;

    // 寻找最大聚类。 extract后的聚类是按照降序排列的，所以第一个聚类就是最大聚类
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
         it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr oneCluster(new pcl::PointCloud <pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
            oneCluster->points.push_back(pc_ground->points[*pit]); // 把点分别归入某一类

        oneCluster->width = oneCluster->points.size();
        oneCluster->height = 1;
        oneCluster->is_dense = true;

        // 要求每一个最大聚类的中心位置不能太偏左或偏右；
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*oneCluster, centroid);
        if (centroid(1) > 20.0 || centroid(1) < -10)  // 质心左偏可以多一些，因为车靠右行驶，左边相对不准确
        {
            std::cout << "Centroid of the biggest cluster is not in middle!" << endl;  // centroid(1)是y轴坐标
            return;
        }
        if ((double(oneCluster->points.size()) / double(pc_ground->points.size())) < 0.6)  // 占比不能太小
        {
            std::cout << "The ratio (the largest cluster / total number) is small !" << endl;
            return;
        }

        pcl::copyPointCloud(*oneCluster, largestCluster);   // 最终选择的路面
        largestCluster.header.frame_id = "base_link";
        largestCluster.width = largestCluster.points.size();
        largestCluster.height = 1;
        largestCluster.is_dense = true;
//        std::cout << "number of points in largest cluster:" << largestCluster.points.size() << std::endl;
    }

//     For debug：发布最大聚类的点云
    sensor_msgs::PointCloud2 pc_debug;
    pcl::toROSMsg(largestCluster, pc_debug);
    pc_debug.header.stamp = cloud_header.stamp;
    pc_debug.header.frame_id = "base_link";
    pub_debug.publish(pc_debug);   // topic name
}

// 在base_link坐标系中计算边界点坐标
void BoundaryDetector::PickBoundary(pcl::PointCloud<pcl::PointXYZ>::ConstPtr pc_cluster) {
    PointType minPt, maxPt;
    pcl::getMinMax3D(*pc_cluster, minPt, maxPt);  //获取点云最大最小值

    pcl::PointCloud<PointType>::Ptr cloud_passthrough(new pcl::PointCloud <PointType>);
    PointType minPt_slice, maxPt_slice;
    PointType left_point, right_point;
    Eigen::Vector4f center;

    bool first_slice = true;
    double dis = minPt.x + 3.0*ds_size; // 去除头尾形状不的那些部分
    while (dis < maxPt.x -3.0*ds_size) {
        // 创建滤波器对象
        pcl::PassThrough <PointType> pass;
        pass.setInputCloud(pc_cluster);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(dis, dis + ds_size);  // 依次增大
        pass.filter(*cloud_passthrough);
        pcl::getMinMax3D(*cloud_passthrough, minPt_slice, maxPt_slice);  //获取一个横条点云最大最小值
        pcl::compute3DCentroid(*cloud_passthrough, center);

        left_point.x = center(0);
        left_point.y = maxPt_slice.y;
        left_point.z = center(2);
//        left_point.z = maxPt_slice.z;
//        double theta_left = atan(left_point.y/left_point.x)/PI*180.0; // 一般小于35度
//        std::cout << "theta_left is" << atan(left_point.y/left_point.x)/PI*180.0 << std::endl;
        right_point.x = center(0);
        right_point.y = minPt_slice.y;
        right_point.z = center(2);
//        double theta_right = atan(-1.0*right_point.y/right_point.x)/PI*180.0; // 一般小于35度
//        std::cout << "theta_right is" << atan(-1.0*right_point.y/right_point.x)/PI*180.0 << std::endl;
        if (first_slice == true) {
            // 左右边缘点初始化
            last_right_point_oneSlice = right_point;
            first_slice = false;
            continue;
        }

        // x轴变化0.3m, 质心在y轴变化应小于1.0m; 否则忽略该帧。如实反映路面宽阔的地方
        // 而且要求该点水平夹角不能太大，否则判定为雷达视场边界产生的虚假边沿点
//        if (theta_left>0.0 && theta_left<38.0 && (abs(left_point.y - last_left_point_oneSlice.y) < 2.0))
        if (abs(left_point.y - last_left_point_oneSlice.y) < 12.0)
        {
            if(left_point.y > 0.0) // 左边沿点不大可能在车的右边
                boundary_point_left_local.points.push_back(left_point);
        }
        else {std::cout << "Two left points of neighbour slice is too large. " << std::endl; }

        // x轴变化0.3m, y轴变化应小于1.0m. 右侧看的清楚
        if (abs(right_point.y - last_right_point_oneSlice.y) < 12.0)
        {
            if(right_point.y < 0.0) // 右边沿点不大可能在车的左边
                boundary_point_right_local.points.push_back(right_point);
        }
        else {std::cout << "Two right points of neighbour slice is too large. "<< std::endl; }
        last_right_point_oneSlice = right_point;

        dis += ds_size/1.3;  //???
    }
    boundary_point_left_local.width = boundary_point_left_local.points.size();
    boundary_point_left_local.height = 1;
    boundary_point_left_local.is_dense = true;
    boundary_point_left_local.header.frame_id = "base_link";

    boundary_point_right_local.width = boundary_point_right_local.points.size();
    boundary_point_right_local.height = 1;
    boundary_point_right_local.is_dense = true;
    boundary_point_right_local.header.frame_id = "base_link";

    cloud_passthrough->clear();
}

// 在base_link坐标系下，进行边界点的稀疏和更新
void BoundaryDetector::BoundarySparse(Eigen::Matrix4f transform_1)
{
    // 求出transform_1的逆变换。将左边界点从map转换到base_link下，便于根据车辆行进位姿进行边界点的提取
    pcl::transformPointCloud(left_Pts_added, left_Pts_added, transform_1.inverse());
    left_Pts_added.header.frame_id = "base_link";

//    // 发布累加的左边沿点
    sensor_msgs::PointCloud2 pc_total;
    pcl::toROSMsg(left_Pts_added, pc_total);
    pc_total.header.stamp = cloud_header.stamp;
    pc_total.header.frame_id = "base_link";
    pub_total.publish(pc_total);   // topic name

    // 直通滤波，选取左边沿点云的质心。作为最终的边沿点
    pcl::PointCloud<PointType>::Ptr cloud_select(new pcl::PointCloud <PointType>);
    Eigen::Vector4f centerPt;
    PointType minPt, maxPt;
    pcl::PassThrough <PointType> pass;
    pass.setInputCloud(left_Pts_added.makeShared());
    pass.setFilterFieldName("x");
    pass.setFilterLimits(X_min+1.0, X_min+1.0+1.5*ds_size);   // slice长度是1.5*ds_size。要和下面历史点云的直通滤波取值相同
    pass.filter(*cloud_select);
    pcl::compute3DCentroid(*cloud_select, centerPt);
    pcl::getMinMax3D(*cloud_select, minPt, maxPt);  //获取点云最大最小值
    Eigen::Matrix3f covariance;
    PointType Pt_temp;      // 这是该slice的最新左边界点
    Pt_temp.x = centerPt(0);
//    Pt_temp.y = centerPt(1);
    Pt_temp.y = maxPt.y; // 用最左侧点代替该地方的质点y坐标
    Pt_temp.z = centerPt(2);

    // 求出历史边沿点的逆变换，从map转换到base_link下，便于融合
    pcl::transformPointCloud(*history_cloud, *history_cloud, transform_1.inverse());
    history_cloud->header.frame_id = "base_link";
    pcl::PointCloud<PointType> cloud_temp, cloud_temp_left, cloud_temp_right;
    pass.setInputCloud(history_cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(X_min+1.0, X_min+1.0+1.5*ds_size);;   // 为了避免对历史点云左右分割时，发生错误，尽量不要取太大
    pass.filter(cloud_temp);
    pass.setFilterLimitsNegative(true);
    pass.filter(*history_cloud); // 去掉该slice中的旧点，以便后面加入新融合的点
    pass.setFilterLimitsNegative(false);
    // 对历史边沿点的这个slice（即cloud_temp），再分割为左边沿点和右边沿点。分别进行融合
    pass.setInputCloud(cloud_temp.makeShared());
    pass.setFilterFieldName("y");
    pass.setFilterLimits(0.0, 100.0);
    pass.filter(cloud_temp_left);  // 得到历史左边沿点
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-100.0, 0.0);
    pass.filter(cloud_temp_right);   // 得到历史右边沿点

    // 只有当累加边沿点的方差较小时，才将当前的左侧新边沿点，加入历史边沿点的slice; 同时至少有2个高障碍物点，或者为空
    // 车宽约3.5m。 这个范围内不可能有边沿点。但是这样判断去除不干净。
    if(maxPt.y-minPt.y < 10.0)
    {
//        std::cout << "Left width: " << maxPt.y-minPt.y << std::endl;
        cloud_temp_left.points.push_back(Pt_temp);
        cloud_temp_left.header.frame_id = "base_link";
        cloud_temp_left.width = cloud_temp_left.points.size();
        cloud_temp_left.height = 1;
        cloud_temp_left.is_dense = true;
        Eigen::Vector4f centerPt_temp;
        pcl::compute3DCentroid(cloud_temp_left, centerPt_temp); // 融合后的左侧点
        PointType left_Pt;
        left_Pt.x = centerPt_temp(0);
        left_Pt.y = centerPt_temp(1);
        left_Pt.z = centerPt_temp(2);
        history_cloud->points.push_back(left_Pt); // 将左侧横条质心，加入历史点云
    }

    // 直通滤波，将left_Pts_added中处理过的点云滤除，否则该点云越来越大，浪费存储空间
    pass.setInputCloud(left_Pts_added.makeShared());
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-10.0, 200.0); // 只保留前向这一段点云
    pass.filter(left_Pts_added);
    //再次转换到map坐标系下
    pcl::transformPointCloud(left_Pts_added, left_Pts_added, transform_1);
    left_Pts_added.header.frame_id = "map";


    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 开始处理右侧边界点
    // 求出transform_1的逆变换。将右边界点从map转换到base_link下，便于根据车辆行进位姿进行边界点的提取
    pcl::transformPointCloud(right_Pts_added, right_Pts_added, transform_1.inverse());
    right_Pts_added.header.frame_id = "base_link";

    // 直通滤波，选取右边沿点云的质心, 作为最终的边沿点
//    pcl::PassThrough <PointType> pass;
    pass.setInputCloud(right_Pts_added.makeShared());
    pass.setFilterFieldName("x");
    pass.setFilterLimits(X_min+1.0, X_min+1.0+1.5*ds_size);;   // 选取这一段稀疏化，既不浪费，又不添加更多计算量
    pass.filter(*cloud_select);
    pcl::compute3DCentroid(*cloud_select, centerPt);
    pcl::getMinMax3D(*cloud_select, minPt, maxPt);  //获取点云最大最小值
//    PointType Pt_temp;
    Pt_temp.x = centerPt(0);
    Pt_temp.y = centerPt(1);
//    Pt_temp.y = minPt.y;
    Pt_temp.z = centerPt(2);

    // 右侧也做类似的检验
    if(maxPt.y-minPt.y < 10.0)
    {
//        std::cout << "Right width: " << maxPt.y-minPt.y << std::endl;
        cloud_temp_right.points.push_back(Pt_temp);  // 将新产生的那个点融合到历史点云的这一段
        cloud_temp_right.header.frame_id = "base_link";
        cloud_temp_right.width = cloud_temp_right.points.size();
        cloud_temp_right.height = 1;
        cloud_temp_right.is_dense = true;
        Eigen::Vector4f centerPt_temp;
        pcl::compute3DCentroid(cloud_temp_right, centerPt_temp); // 融合后的右侧边沿点
        PointType right_Pt;
        right_Pt.x = centerPt_temp(0);
        right_Pt.y = centerPt_temp(1);
        right_Pt.z = centerPt_temp(2);
        history_cloud->points.push_back(right_Pt); // 将右侧横条的质心加入历史点云
    }

    // 直通滤波，将right_Pts_added中处理过的点云滤除，否则该点云越来越大，浪费存储空间
    pass.setInputCloud(right_Pts_added.makeShared());
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-10.0, 200.0); // 只保留前向这一段点云
    pass.filter(right_Pts_added);
    //再次转换到map坐标系下
    pcl::transformPointCloud(right_Pts_added, right_Pts_added, transform_1);
    right_Pts_added.header.frame_id = "map";


    // 将历史边沿点从base_link再转换到map下，并publish
    pcl::transformPointCloud(*history_cloud, *history_cloud, transform_1);
    history_cloud->header.frame_id = "map";
    history_cloud->width = history_cloud->points.size();
    history_cloud->height = 1;
    history_cloud->is_dense = true;

    // 发布更新后的点云
    sensor_msgs::PointCloud2 pc_publish;
    pcl::toROSMsg(*history_cloud, pc_publish);
    pc_publish.header.stamp = cloud_header.stamp;
    pc_publish.header.frame_id = "map";
    pub_final.publish(pc_publish);   // topic name: /road_boundary

    cloud_select->clear();
}

// 保存边界点到文件中
void BoundaryDetector::timerCallBack(const ros::TimerEvent &e)
{
    if (history_cloud->points.size() != 0)
    {
        std::cout << "write to boundary.pcd." << std::endl;
        pcl::PCDWriter writer;
        writer.write("boundary.pcd", *history_cloud, false);    //true，则保存为Binary格式，速度更快
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "boundary_detect");
    ros::NodeHandle nh;

    BoundaryDetector BD;

    ros::spin();

    return 0;
}
