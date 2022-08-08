/* 徐工研究院装载场边线提取.
 * 前向4个雷达点云组合，后向2个雷达组合。
 * 分别使用障碍物的边线，z轴直通滤波，作为地面边线。
 * 在map坐标系中进行融合。
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
#include <pcl/filters/radius_outlier_removal.h>
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
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "ctime"

#include <fstream>
#include <cmath>
#include <vector>


#define PointType pcl::PointXYZ
#define  PI  3.1415926535
using namespace std;
using namespace Eigen;

class BoundaryDetector {
private:
    ros::NodeHandle nh;

    bool savePCD;
    bool readPCD;

    int frame_cnt;
    int Cnt;  //累加的帧数
//    double X_min;
    double ds_size;
    double ransac_th;
    double cluster_r;

    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_iv; //iv 300线雷达订阅
    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_1271;  // 后向雷达 （高）
    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_1501;  // 前向中间雷达
    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_1541;  // 左补盲
    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_0181;  // 右补盲
    message_filters::Subscriber <sensor_msgs::PointCloud2> sub_lidar_0201;  // 后向雷达 （低）
    message_filters::Subscriber <nav_msgs::Odometry> sub_odom_; //rtk odom
    typedef message_filters::sync_policies::ApproximateTime <sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
                                                            sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
                                                            sensor_msgs::PointCloud2, sensor_msgs::PointCloud2,
                                                            nav_msgs::Odometry> MySyncPolicy;
    typedef message_filters::Synchronizer <MySyncPolicy> Sync;
    boost::shared_ptr <Sync> sync_;//时间同步器

    ros::Publisher pub_final;
    ros::Publisher pub_total;
    ros::Publisher pub_debug;
    ros::Publisher pub_debug1;
    ros::Publisher pub_fan_shaped;

    geometry_msgs::PoseStamped last_pose;
    bool first_frame;
    Eigen::Quaterniond q0;  // 第1帧点云的姿态
    double roll_0, pitch_0, yaw_0;
    Eigen::Matrix4f transform_1;

    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloud_front_add;
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloud_back_add;
    pcl::PointCloud <pcl::PointXYZ> boundary_point_fan;
    pcl::PointCloud <pcl::PointXYZ> boundary_point_fan_all;
    pcl::PointCloud <pcl::PointXYZ> boundary_fan_inMap; // 在odom或map下的坐标

    pcl::ModelCoefficients::Ptr coefficients; // ransac用
    pcl::PointIndices::Ptr inliers;
    pcl::PointCloud<PointType> laserCloudGround;
    pcl::PointCloud<PointType> laserCloudNotGround;
    pcl::PointCloud<PointType> laserCloudNotGround_front;
    pcl::PointCloud<PointType> laserCloudNotGround_fan;
    pcl::PointCloud<PointType> laserCloudNotGround_back;
    pcl::PointCloud<pcl::PointXYZ> laserCloud_tmp;
    pcl::PointCloud<PointType> laserCloudObstacles;
    pcl::PointCloud<pcl::PointXYZI> boundary_points_local;
    pcl::PointCloud<pcl::PointXYZI> boundary_points_inMap;

    pcl::search::KdTree<PointType>::Ptr tree;  // 聚类用

    std_msgs::Header cloud_header;

    pcl::PointCloud<pcl::PointXYZ>::Ptr history_cloud; //存放历史边界点的点云
    pcl::PCDReader reader;    //定义点云读取对象
    pcl::PCDWriter writer;
    ros::Timer timer;

//    struct cluster_obstacle
//    {
//        int points_number;
//        Eigen::Vector4f centroid;
//        PointType minPt;
//        PointType maxPt;
//    };
//    vector<cluster_obstacle> obstacles;

public:
    BoundaryDetector() :
            nh(),
            frame_cnt(0),
            first_frame(true)
    {
        coefficients.reset(new pcl::ModelCoefficients);
        inliers.reset(new pcl::PointIndices);
        tree.reset(new pcl::search::KdTree <PointType>);
        history_cloud.reset(new pcl::PointCloud<PointType>);
        laserCloud_front_add.reset(new pcl::PointCloud<PointType>);
        laserCloud_back_add.reset(new pcl::PointCloud<PointType>);

        nh.param("Cnt",Cnt,4);
//        nh.param("X_min", X_min, 20.0);  // 点云x轴直通滤波的起点. 从baselink前方13米处开始，避开盲区。太小虚假则虚假边界点多
        nh.param("ds_size", ds_size, 0.10);  // default: 0.15
        nh.param("ransac_th", ransac_th, 0.3);
        nh.param("cluster_r", cluster_r, 0.2);

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

        sub_lidar_iv.subscribe(nh, "/iv_points", 5);
        sub_lidar_1271.subscribe(nh, "/livox/lidar_3WEDH7600121271", 5);
        sub_lidar_1501.subscribe(nh, "/livox/lidar_3WEDH7600121501", 5);
        sub_lidar_1541.subscribe(nh, "/livox/lidar_3WEDH7600121541", 5);
        sub_lidar_0181.subscribe(nh, "/livox/lidar_3WEDJ1500100181", 5);
        sub_lidar_0201.subscribe(nh, "/livox/lidar_3WEDJ1500100201", 5);
        sub_odom_.subscribe(nh, "/gps/odom", 30);

        pub_final = nh.advertise<sensor_msgs::PointCloud2>("/road_boundary", 10);
        pub_debug = nh.advertise<sensor_msgs::PointCloud2>("/debug", 10);  // 调试用
        pub_debug1 = nh.advertise<sensor_msgs::PointCloud2>("/debug1", 10);  // 调试用
        pub_total = nh.advertise<sensor_msgs::PointCloud2>("/total", 10);  // 发布合并后的点云

        pub_fan_shaped = nh.advertise<sensor_msgs::PointCloud2>("/road_fan_shaped", 10);

        //回调
        sync_.reset(new Sync(MySyncPolicy(10), sub_lidar_iv, sub_lidar_1501, sub_lidar_1541, sub_lidar_0181,
                             sub_lidar_1271, sub_lidar_0201, sub_odom_));
        sync_->registerCallback(boost::bind(&BoundaryDetector::callback, this, _1, _2, _3, _4, _5, _6, _7));
    }

    ~BoundaryDetector() {}

    void callback(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgiv,
                  const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg1501,
                  const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg1541,
                  const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg0181,
                  const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg1271,
                  const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg0201,
                  const nav_msgs::Odometry::ConstPtr &odomMsg);

    void planeFitting(pcl::PointCloud<PointType>::ConstPtr pointCloudInput);

    void objectCluster(pcl::PointCloud<PointType>::ConstPtr pc_not_ground);

    void PickBoundary_front();

    void timerCallBack(const ros::TimerEvent& e);
};


void BoundaryDetector::callback(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsgiv,
                                const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg1501,
                                const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg1541,
                                const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg0181,
                                const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg1271,
                                const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg0201,
                                const nav_msgs::Odometry::ConstPtr &odomMsg)
{
    // 对iv 300线雷达点云进行旋转
    Eigen::Quaterniond q1(0.0, 0.707, 0.0, 0.707); // base_link到innovusion的旋转关系;
    q1.normalize();
    Eigen::Vector3d t1(7.975, 0.0, 1.045); // base_link到innovusion的平移关系
    Eigen::Affine3d T1w(q1);
    T1w.pretranslate(t1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudIn(new pcl::PointCloud<PointType>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloudTransform(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*laserCloudMsgiv, *laserCloudIn);
    pcl::transformPointCloud(*laserCloudIn, *laserCloudTransform, T1w); // 对iv 300线雷达点云进行旋转
    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloud_front(new pcl::PointCloud<PointType>);
    *laserCloud_front = *laserCloud_front + *laserCloudTransform;
    /*
     * Eigen::Quaterniond q_1(odomMsg->pose.pose.orientation.w, odomMsg->pose.pose.orientation.x,
                            odomMsg->pose.pose.orientation.y, odomMsg->pose.pose.orientation.z); // 当前帧的位姿
       double roll1, pitch1, yaw1;
       tf2::Quaternion dq1(q_1.x(), q_1.y(), q_1.z(), q_1.w());
       tf2::Matrix3x3 m1(dq1);
       m1.getRPY(roll1, tch1, yaw1);
       if(abs(pitch1 * 57.3) > 5 || abs(pitch1 * 57.3) < 4)
       {
           std::cout<<pitch1 * 57.3 <<std::endl;
           return;
       }*/
    pcl::fromROSMsg(*laserCloudMsg1501, *laserCloudIn);
    *laserCloud_front = *laserCloud_front + *laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg1541, *laserCloudIn);
    *laserCloud_front = *laserCloud_front + *laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg0181, *laserCloudIn);
    *laserCloud_front = *laserCloud_front + *laserCloudIn;  // 前向雷达点云
    pcl::PassThrough<PointType> pass;
    pass.setInputCloud(laserCloud_front);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-1.0, 120.0);
    pass.filter(*laserCloud_front);  // 许多密集点构成的边界点
    // 下采样
//    pcl::VoxelGrid<pcl::PointXYZ> filter;
//    filter.setInputCloud(laserCloud_front);
//    filter.setLeafSize(ds_size, ds_size, ds_size);     // 设置体素栅格的大小
//    filter.filter(*laserCloud_front);



    pcl::PointCloud<pcl::PointXYZ>::Ptr laserCloud_back(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*laserCloudMsg1271, *laserCloudIn);
    *laserCloud_back = *laserCloud_back + *laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg0201, *laserCloudIn);
    *laserCloud_back = *laserCloud_back + *laserCloudIn;  // 后向雷达点云
    pass.setInputCloud(laserCloud_back);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-120.0, 10.0);
    pass.filter(*laserCloud_back);  // 许多密集点构成的边界点
//    // 下采样
//    filter.setInputCloud(laserCloud_back);
//    filter.setLeafSize(ds_size, ds_size, ds_size);     // 设置体素栅格的大小
//    filter.filter(*laserCloud_back);



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

        if(frame_cnt == 0)    // 记录每4帧中第0帧的姿态为q1
        {
            Eigen::Quaterniond q(odomMsg->pose.pose.orientation.w, odomMsg->pose.pose.orientation.x,
                                 odomMsg->pose.pose.orientation.y, odomMsg->pose.pose.orientation.z); // 当前帧的位姿
            double roll, pitch, yaw;
            tf2::Quaternion dq1(q.x(), q.y(), q.z(), q.w());
            tf2::Matrix3x3 m1(dq1);
            m1.getRPY(roll_0, pitch_0, yaw_0);  // 求roll和pitch
        }

        //输入的四元数, 转化成欧拉角。提取原位姿旋转四元数的pitch和roll; yaw保持不变
        Eigen::Quaterniond q(odomMsg->pose.pose.orientation.w, odomMsg->pose.pose.orientation.x,
                             odomMsg->pose.pose.orientation.y, odomMsg->pose.pose.orientation.z); // 当前帧的位姿
        double roll, pitch, yaw;
        tf2::Quaternion dq1(q.x(), q.y(), q.z(), q.w());
        tf2::Matrix3x3 m1(dq1);
        m1.getRPY(roll, pitch, yaw);  // 求roll和pitch
        double droll = roll - roll_0;
        double dpitch = pitch - pitch_0;
        double dyaw = yaw - yaw_0;
        //        std::cout << "roll pitch yaw is " << droll*57.0 << ' ' << dpitch*57.0 << ' ' << dyaw*57.0 << std::endl;
//        transform_2.rotate (Eigen::AngleAxisf (droll, Eigen::Vector3f::UnitX()));
//        transform_2.rotate (Eigen::AngleAxisf (dpitch, Eigen::Vector3f::UnitY()));
        transform_2.rotate (Eigen::AngleAxisf (dyaw, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud(*laserCloud_front, *laserCloud_front, transform_2);
        pcl::transformPointCloud(*laserCloud_back, *laserCloud_back, transform_2);
        last_pose = position_3d; // 保存当前位姿
    }

    *laserCloud_front_add = *laserCloud_front_add + *laserCloud_front;  // 对连续几帧进行叠加
    *laserCloud_back_add = *laserCloud_back_add + *laserCloud_back;  // 对连续几帧进行叠加
    laserCloud_front->clear();
    laserCloud_back->clear();
    frame_cnt++;
    if (frame_cnt == 8) {
        cloud_header = odomMsg->header;
        // 定义从local到map的变换矩阵
        transform_1 = Eigen::Matrix4f::Identity();
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

//        // For debug：发布几帧叠加的点云
//        sensor_msgs::PointCloud2 pc_debug;
//        pcl::toROSMsg(*pointcloud_adder, pc_debug);
//        pc_debug.header.stamp = cloud_header.stamp;
//        pc_debug.header.frame_id = "base_link";
//        pub_debug.publish(pc_debug);   // topic name

        // RANSAC地面分割，得到地面点云和非地面点云laserCloudNotGround
        planeFitting(laserCloud_front_add);
        // 半径滤波
        // 创建半径滤波（模板）类对象
        pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
        // 设置输入点云
        ror.setInputCloud(laserCloud_tmp.makeShared());
        // 设置搜索半径
        ror.setRadiusSearch(ds_size*2.0);
        // 设置半径范围内的最少点数阈值
        ror.setMinNeighborsInRadius(5);
        // 执行滤波，并带出结果数据
        ror.filter(laserCloud_tmp);  //保存滤波结果到cloud_filtered

        pcl::copyPointCloud(laserCloud_tmp, laserCloudNotGround_front);

        pcl::copyPointCloud(laserCloudNotGround_front, laserCloudNotGround_fan);

        //去除离群点
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(laserCloudNotGround_fan.makeShared());
        sor.setMeanK(20);
        sor.setStddevMulThresh(1.0);
        sor.filter(laserCloudNotGround_fan);

       pcl::PassThrough <PointType> pass;
        pass.setInputCloud(laserCloudNotGround_fan.makeShared());
        pass.setFilterFieldName("z");
        // X_min小可以充分发挥补盲雷达的作用。
        pass.setFilterLimits(-1.5, 0);
        pass.filter(laserCloudNotGround_fan);
/*
        //pcl::PassThrough <PointType> pass;
        pass.setInputCloud(laserCloudNotGround_fan.makeShared());
        pass.setFilterFieldName("y");
        // X_min小可以充分发挥补盲雷达的作用。
        pass.setFilterLimits(-35, 45);
        pass.filter(laserCloudNotGround_fan);
        */
        PickBoundary_front();

        pcl::transformPointCloud(boundary_point_fan, boundary_fan_inMap, transform_1);
        //boundary_fan_inMap.header.frame_id = "map";
        boundary_fan_inMap.width = boundary_fan_inMap.points.size();
        boundary_fan_inMap.height = 1;
        boundary_fan_inMap.is_dense = true;
        // 将各帧合成为一个点云
        boundary_point_fan_all += boundary_fan_inMap;
        //boundary_point_fan_all.header.frame_id = "map";
        boundary_point_fan_all.width = boundary_point_fan_all.points.size();
        boundary_point_fan_all.height = 1;
        boundary_point_fan_all.is_dense = true;

        //去除离群点
        /*pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor1;
        sor1.setInputCloud(boundary_point_fan_all.makeShared());
        sor1.setMeanK(10);
        sor1.setStddevMulThresh(0.5);
        sor1.filter(boundary_point_fan_all);*/

        /*pcl::transformPointCloud(boundary_point_fan_all, boundary_point_fan_all, transform_1.inverse());
        boundary_point_fan_all.header.frame_id = "base_link";*/

        /*pcl::transformPointCloud(boundary_point_fan_all, boundary_point_fan_all, transform_1.inverse());
        boundary_point_fan_all.header.frame_id = "base_link";*/

        //    //     For debug：发布边界点云
        sensor_msgs::PointCloud2 pc_debug;
        pcl::toROSMsg(laserCloudNotGround_front, pc_debug);
        pc_debug.header.stamp = cloud_header.stamp;
        pc_debug.header.frame_id = "base_link";
        pub_debug.publish(pc_debug);   // topic name

        sensor_msgs::PointCloud2 pc_debug_1;
        pcl::toROSMsg(laserCloudNotGround_fan, pc_debug_1);
        pc_debug_1.header.stamp = cloud_header.stamp;
        pc_debug_1.header.frame_id = "base_link";
        pub_debug1.publish(pc_debug_1);

        // 发布更新后的点云
        sensor_msgs::PointCloud2 pc_publish_front;
        pcl::toROSMsg(boundary_point_fan_all, pc_publish_front);
        pc_publish_front.header.stamp = cloud_header.stamp;
        pc_publish_front.header.frame_id = "map";
        pub_fan_shaped.publish(pc_publish_front);   // topic name: /road_fan_shaped

        planeFitting(laserCloud_back_add);
        pcl::copyPointCloud(laserCloud_tmp, laserCloudNotGround_back);

//        objectCluster(laserCloudNotGround_front.makeShared());   // 对前方非地面点进行聚类。因为前后方的平面不一样，平均高度不同
//        objectCluster(laserCloudNotGround_back.makeShared());  // 对后方非地面点进行聚类




        //输入的base_link下的边界点点云，把它们转化到map坐标系下
        pcl::transformPointCloud(boundary_points_local, boundary_points_inMap, transform_1);
        boundary_points_inMap.header.frame_id = "base_link";
        boundary_points_inMap.width = boundary_points_inMap.points.size();
        boundary_points_inMap.height = 1;
        boundary_points_inMap.is_dense = true;


        // 针对当前帧的每个聚类，进行历史边界点云的更新。都是在map坐标系下进行
//        BoundarySparse();

        frame_cnt = 0;  // 重新开始计数
        laserCloud_front_add->clear();
        laserCloud_back_add->clear();
        laserCloudObstacles.clear();
        boundary_points_local.clear();  // 清空该帧的边沿点

        boundary_point_fan.clear();
        boundary_fan_inMap.clear();
    }
}

// ransac地面分割
void BoundaryDetector::planeFitting(pcl::PointCloud<PointType>::ConstPtr pointCloudInput) {
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
    seg.setEpsAngle(15.0 / 180.0 * PI);

    seg.setInputCloud(pointCloudInput);  //输入为点云指针
    seg.segment(*inliers, *coefficients);
    pcl::ExtractIndices <PointType> extract;
    if (inliers->indices.size() < 3000)
    {
        ROS_INFO("Too small inliers in RANSAC model. skip!"); // 排除近处的密集点
        return;
    }
    else
    {
        extract.setInputCloud(pointCloudInput);
        extract.setIndices(inliers);
        extract.filter(laserCloudGround);  // 地面点云
//        double ratio = double(laserCloudGround.points.size())/double(pointCloudInput.points.size());
//        std::cout << "ratio is :" << ratio << std::endl;
        extract.setNegative (true);
        extract.filter(laserCloudNotGround); // 非地面点云
//        std::cout << "Number of nonGround points:" << laserCloudNotGround.points.size() << std::endl;
    }

    laserCloud_tmp.clear();
    // 过滤掉平面以下的点云
    for(int i=0; i<laserCloudNotGround.points.size(); i++)
        if ((laserCloudNotGround.points[i].x * coefficients->values[0] +
        laserCloudNotGround.points[i].y * coefficients->values[1] +
        laserCloudNotGround.points[i].z * coefficients->values[2] +
        coefficients->values[3] > 0.0) &&
                (laserCloudNotGround.points[i].x * coefficients->values[0] +
                 laserCloudNotGround.points[i].y * coefficients->values[1] +
                 laserCloudNotGround.points[i].z * coefficients->values[2] +
                 coefficients->values[3] < 0.4))
            laserCloud_tmp.push_back(laserCloudNotGround.points[i]);
    //std::cout << " number of points in boundary: " << laserCloud_tmp.points.size() << std::endl;

}

// 找到非地面点中的最大聚类。并记录质心和最大最小点
void BoundaryDetector::objectCluster(pcl::PointCloud<PointType>::ConstPtr pc_not_ground)
{
    // 计算平面点的质心，便于后面进行比较
    Eigen::Vector4f centroid_ground;
    pcl::compute3DCentroid(laserCloudGround, centroid_ground);

    // 对非地面点云进行聚类
    tree->setInputCloud(pc_not_ground);
    std::vector <pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction <pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_r); //近邻搜索的搜索半径。需要多次调节 0.4
    ec.setMinClusterSize(25);
//    ec.setMaxClusterSize(200000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(pc_not_ground);
    ec.extract(cluster_indices);
//    std::cout << "It has " << cluster_indices.size() << " planes by clustering" << endl;

//    int j=0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin();
         it != cluster_indices.end(); ++it)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster(new pcl::PointCloud <pcl::PointXYZ>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
            Cluster->points.push_back(pc_not_ground->points[*pit]); // 把点分别归入某一类
        Cluster->width = Cluster->points.size();
        Cluster->height = 1;
        Cluster->is_dense = true;

        // 记录每个聚类的位置
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*Cluster, centroid);
        PointType minPt, maxPt;
        pcl::getMinMax3D(*Cluster, minPt, maxPt);  //获取该聚类点云最大最小值
        // 聚类中心的半径在70m以外且较小的物体，不考虑
        if ((centroid(0)*centroid(0) + centroid(1)*centroid(1) > 5000.0) && (Cluster->points.size() < 50) )
            continue;

//        struct cluster_obstacle oneCluster;
//        oneCluster.points_number = Cluster->points.size();
//        oneCluster.centroid = centroid;
//        oneCluster.minPt = minPt;
//        oneCluster.maxPt = maxPt;
//        obstacles.push_back(oneCluster);  // 将该聚类的信息保存到obstacles结构体数组中
        laserCloudObstacles = laserCloudObstacles + *Cluster;     // 把聚类的点合并在一起

        // 对每个聚类计算边沿点
        pcl::PointCloud<pcl::PointXYZ>::Ptr Cluster_boundary(new pcl::PointCloud <pcl::PointXYZ>);
//        if(centroid(2) > centroid_ground(2))   // 聚类高于地面
//        {
            pcl::PassThrough<PointType> pass;
            pass.setInputCloud(Cluster);
            pass.setFilterFieldName("z");
            pass.setFilterLimits(centroid_ground(2) + ransac_th, centroid_ground(2) + ransac_th + 0.2);
            pass.filter(*Cluster_boundary);  // 许多密集点构成的边界点
            // 下采样
            pcl::VoxelGrid <pcl::PointXYZ> filter;
            filter.setInputCloud(Cluster_boundary);
            filter.setLeafSize(ds_size, ds_size, ds_size);
            filter.filter(*Cluster_boundary);  // 得到该聚类的边沿点
            for(int nIndex = 0; nIndex < Cluster_boundary->points.size(); nIndex++)
            {
                pcl::PointXYZI Pt;
                Pt.x = Cluster_boundary->points[nIndex].x;
                Pt.y = Cluster_boundary->points[nIndex].y;
                Pt.z = Cluster_boundary->points[nIndex].z;
                Pt.intensity = 0.1;                         // 附加上强度
                boundary_points_local.points.push_back(Pt); // 得到所有聚类的边沿点
            }

            // 把历史边界点也转化到base_link下，然后采用角度滤波的方法，对每个聚类进行融合
//            pcl::transformPointCloud(boundary_points_local, boundary_points_inMap, transform_1);
//            boundary_points_inMap.header.frame_id = "map";
//            boundary_points_inMap.width = boundary_points_inMap.points.size();
//            boundary_points_inMap.height = 1;
//            boundary_points_inMap.is_dense = true;
            // 与历史边沿点融合


//        }
//        else if (centroid(2) > centroid_ground(2))   // 聚类低于地面
//        {
//            pcl::PassThrough <PointType> pass;
//            pass.setInputCloud(Cluster);
//            pass.setFilterFieldName("z");
//            pass.setFilterLimits(centroid_ground(2) -ransac_th -0.2, centroid_ground(2) -ransac_th);
//            pass.filter(*Cluster_boundary);
//            // 下采样
//            pcl::VoxelGrid <pcl::PointXYZ> filter;
//            filter.setInputCloud(Cluster_boundary);
//            filter.setLeafSize(ds_size, ds_size, ds_size);
//            filter.filter(*Cluster_boundary);  // 稀疏后的边界点
//            for(int nIndex = 0; nIndex < Cluster_boundary->points.size(); nIndex++)
//            {
//                pcl::PointXYZI Pt;
//                Pt.x = Cluster_boundary->points[nIndex].x;
//                Pt.y = Cluster_boundary->points[nIndex].y;
//                Pt.z = Cluster_boundary->points[nIndex].z;
//                Pt.intensity = 0.1;                         // 附加上强度
//                boundary_points_local.points.push_back(Pt);
//            }
//
//        }
//        j++;
//        std::cout << "j= " << j << std::endl;

    }

    // 发布所有的聚类点云
    sensor_msgs::PointCloud2 pc_total;
    pcl::toROSMsg(laserCloudObstacles, pc_total);
    pc_total.header.stamp = cloud_header.stamp;
    pc_total.header.frame_id = "base_link";
    pub_total.publish(pc_total);   // topic name



}

//在base_link坐标系中计算边界点坐标  扇形
void BoundaryDetector::PickBoundary_front() {
    //vector<vector<double> > r_vec(90,vector<double>(10000,0));
    vector<vector<double> > r_radius(900,vector<double>(1,0));
    vector<vector<double> > r_num(900,vector<double>(1,0));
    double angle1 , r1 , max_num;
    int flag;
    bool first = true;
    for (int num = 0; num < laserCloudNotGround_fan.size() - 1; ++num) {
        //  r_vec[num].resize(outlinerremove_ds.size(),0);
        angle1 = atan2(laserCloudNotGround_fan.points[num].x,laserCloudNotGround_fan.points[num].y) * 57.3;  //转成角度
        r1  = sqrt(laserCloudNotGround_fan.points[num].x * laserCloudNotGround_fan.points[num].x
                   + laserCloudNotGround_fan.points[num].y * laserCloudNotGround_fan.points[num].y);
        flag = int(angle1 * 5);
        if(first)
        {
            r_radius[flag][0] = r1;
            r_num[flag][0] = num;
            first = false;
        }
        max_num = std::max(r1,r_radius[flag][0]);
        if (max_num > r_radius[flag][0] )
        {
            if(angle1 > 60 && angle1 < 115 )
            {
               /* if (r1 > 60 && r1 < 70){
                    r_radius[flag][0] = max_num;
                    r_num[flag][0] = num;}*/
                  //  std::cout<<"1  "<<angle1<<"  "<<r1<<std::endl;
                continue;
            } else
            {
                r_radius[flag][0] = max_num;
                r_num[flag][0] = num;
               // std::cout<<"2  "<<angle1<<"  "<<r1<<std::endl;
            }
        }
    }
    for (int i = 0; i < 900; ++i) {
        boundary_point_fan.points.push_back(laserCloudNotGround_fan.points[ r_num[i][0] ]);
    }

    boundary_point_fan.width = boundary_point_fan.points.size();
    boundary_point_fan.height = 1;
    boundary_point_fan.is_dense = true;
    boundary_point_fan.header.frame_id = "base_link";

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
