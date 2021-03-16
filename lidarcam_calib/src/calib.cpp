#include "ros/ros.h"
#include <pcl/io/pcd_io.h>   //pcd读写类相关头文件
#include <pcl/point_types.h> //pcl中支持的点类型头文件
//#include <pcl/sample_consensus/ransac.h>
//#include <pcl/sample_consensus/sac_model_plane.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/model_types.h>  //分割目标模型头文件
#include <pcl/sample_consensus/method_types.h> //分割计方法头文件
#include <pcl/segmentation/sac_segmentation.h> //基于随机采样一致性(ransac)分割的类的头文件

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/Householder>
#include <Eigen/Jacobi>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

//#include <pcl/console/parse.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/search/impl/search.hpp>

#include <iostream>
#include <cmath>
#include <vector>

//求向量对应的反对称阵
Eigen::Matrix3d inv_sym(const Eigen::Vector3d& line)
{
  Eigen::Matrix3d temp;
  temp << 0, -line(2), line(1),
          line(2), 0, -line(0),
          -line(1), line(0), 0;
  return temp;
}

//将相机坐标系下的点转换至像素坐标系
Eigen::Vector2d cam2pixel(const Eigen::Vector3d& camera_point)
{
  Eigen::Matrix3d K;
  K << 1061.37439737547, 0, 980.706836288949,
       0, 1061.02435228316, 601.685030610243,
       0, 0, 1;
  Eigen::Vector3d cam_p = K*camera_point;
  Eigen::Vector2d uv(cam_p(0)/cam_p(2), cam_p(1)/cam_p(2));
  return uv;
}


int main(int argc, char** argv)
{
    // 从pcd文件读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); //原始点云指针
    if( pcl::io::loadPCDFile<pcl::PointXYZ>("/home/jerry/catkin_ws/data/1.pcd", *cloud) == -1 ) //还是用绝对路径比较好
    {
      PCL_ERROR("Couldn't read file /home/jerry/catkin_ws/data/1.pcd \n");
      return -1;
    }
    //std::cout << cloud->points.size() << std::endl;

    // 对非标定板部分的点云进行滤除
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filter(new pcl::PointCloud<pcl::PointXYZ>); //滤除后的点云指针
    for(int i=0; i<cloud->points.size(); ++i) //points是std::vector<pcl::PointXYZ>类型
    {
      if( (cloud->points[i].x>1.75)&&(cloud->points[i].x<5.2) &&
          (cloud->points[i].y>-0.1)&&(cloud->points[i].y<1.6) &&
          (cloud->points[i].z>-0.6)&&(cloud->points[i].z<0.6) )
      {
        //cloud_filter->points.push_back(cloud->points[i]); 这样写是错误的！
        cloud_filter->push_back(cloud->points[i]);
      }
    }
    pcl::io::savePCDFile("/home/jerry/catkin_ws/data/filter.pcd", *cloud_filter);

    // 用RANSAC拟合平面
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients); //创建一个模型参数对象用于记录结果
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices); //inliers表示误差能容忍的点即内点
    pcl::SACSegmentation<pcl::PointXYZ> seg; //创建一个分割器
    seg.setOptimizeCoefficients(true); //该设置决定留下的是分割掉的点还是分割剩下的点
    seg.setModelType(pcl::SACMODEL_PLANE); //设置目标几何形状
    seg.setMethodType(pcl::SAC_RANSAC); //分割方法：ransac
    seg.setDistanceThreshold(0.01); //设置容忍的误差范围即阈值
    seg.setInputCloud(cloud_filter); //输入点云
    seg.segment(*inliers, *coefficients); //分割点云

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ransac(new pcl::PointCloud<pcl::PointXYZ>); //拟合后的点云指针
    for(int i=0; i<inliers->indices.size(); ++i)
    {
      int inlier_index = inliers->indices[i];
      cloud_ransac->push_back(cloud_filter->points[inlier_index]);
    }
    pcl::io::savePCDFile("/home/jerry/catkin_ws/data/ransac.pcd", *cloud_ransac);

    // 1-1.激光：求标定板边界向量
    Eigen::Vector3d p11,p12;
    p11 << 2.495, 0.160, 0.223;
    p12 << 2.535, 0.187, 0.311;
    Eigen::Vector3d p21,p22;
    p21 << 2.239, 0.075, -0.352;
    p22 << 2.075, 0.473, -0.415;

    Eigen::Vector3d ll1 = p12 - p11;
    ll1 = ll1/( sqrt(ll1(0)*ll1(0)+ll1(1)*ll1(1)+ll1(2)*ll1(2)) );
    Eigen::Vector3d ll2 = p22 - p21;
    ll2 = ll2/( sqrt(ll2(0)*ll2(0)+ll2(1)*ll2(1)+ll2(2)*ll2(2)) );

    // 1-2.激光：求标定板法向量
    Eigen::Vector3d nl = inv_sym(ll1)*ll2;
    nl = nl/( sqrt(nl(0)*nl(0)+nl(1)*nl(1)+nl(2)*nl(2)) );

    // 2-1.图像：求标定板边界向量
    //由github标定工具获得标定板与相机外参
    Eigen::Isometry3d Tcb = Eigen::Isometry3d::Identity();
    cv::Mat cv_rotmat;
    Eigen::Matrix3d rot_mat;
    cv::Mat cv_rvec = ( cv::Mat_<double>(3,1) << -0.329133, -2.80677, -0.68856 );
    cv::Rodrigues(cv_rvec, cv_rotmat);
    cv2eigen(cv_rotmat, rot_mat);
    Tcb.rotate(rot_mat);
    Tcb.pretranslate(Eigen::Vector3d(-0.400131, -0.442675, 1.9672));
//    Tcb = Tcb.inverse();
    std::cout << "标定板与相机外参：\n" << Tcb.matrix() << std::endl;

    //根据外参求得标定板上的点在相机坐标系下的坐标
    Eigen::Vector3d q11,q12;
    q11 << -0.225, 0.3, 0;
    q12 << -0.225, 0, 0;
    q11 = Tcb*q11;
    q12 = Tcb*q12;
    Eigen::Vector3d q21,q22;
    q21 << 0, 0.825, 0;
    q22 << 0.3, 0.825, 0;
    q21 = Tcb*q21;
    q22 = Tcb*q22;

    Eigen::Vector3d lc1 = q12 - q11;
    lc1 = lc1/( sqrt(lc1(0)*lc1(0)+lc1(1)*lc1(1)+lc1(2)*lc1(2)) );
    Eigen::Vector3d lc2 = q22 - q21;
    lc2 = lc2/( sqrt(lc2(0)*lc2(0)+lc2(1)*lc2(1)+lc2(2)*lc2(2)) );

    // 2-2.图像：求标定板法向量
    Eigen::Vector3d nc = inv_sym(lc1)*lc2;
    nc = nc/( sqrt(nc(0)*nc(0)+nc(1)*nc(1)+nc(2)*nc(2)) );

    // 3.构造方程求解Rcl
    Eigen::Matrix3d Rcl,X1,X2;
    X1.block<3,1>(0,0) = ll1;
    X1.block<3,1>(0,1) = ll2;
    X1.block<3,1>(0,2) = nl;
    X2.block<3,1>(0,0) = lc1;
    X2.block<3,1>(0,1) = lc2;
    X2.block<3,1>(0,2) = nc;
    Rcl = X2*X1.inverse();
    std::cout << std::endl << "旋转外参Rcl：\n" << Rcl << std::endl << std::endl;

    // 4.构造方程求tcl
    Eigen::Vector3d tcl;
    Eigen::Matrix3d A1,A2;
    A1 = Eigen::Matrix3d::Identity() - lc1*lc1.transpose();
    A2 = Eigen::Matrix3d::Identity() - lc2*lc2.transpose();
    Eigen::Vector3d b1,b2;
    b1 = A1*(q11-Rcl*p11);
    b2 = A2*(q22-Rcl*p22);
//    std::cout << A1.inverse()*b1;
    Eigen::Matrix<double, 6, 3> H;
    Eigen::Matrix<double, 6, 1> b;
    H.block<3,3>(0,0) = A1;
    H.block<3,3>(3,0) = A2;
    b.block<3,1>(0,0) = b1;
    b.block<3,1>(3,0) = b2;
    tcl = H.colPivHouseholderQr().solve(b);
    std::cout << std::endl << "平移外参tcl：\n" << tcl.transpose() << std::endl << std::endl;

    // 5.验证外参的准确性
    //读取图像在z1_undistorted.jpg
    cv::Mat img;
    img = cv::imread("/home/jerry/catkin_ws/data/z1_undistorted.jpg");
//    Eigen::Isometry3d Tcl;
//    Tcl.rotate(Rcl);
//    Tcl.pretranslate(tcl);
//    std::cout << "外参Tcl：\n" << Tcl.matrix() << std::endl << std::endl;
    //将雷达坐标系下的平面点云转换至相机坐标系下
    std::vector<Eigen::Vector3d> lidar_points;
    for(int i=0; i<inliers->indices.size(); ++i)
    {
      int inlier_index = inliers->indices[i];
      Eigen::Vector3d lidar_point(cloud_filter->points[inlier_index].x, cloud_filter->points[inlier_index].y, cloud_filter->points[inlier_index].z);
      lidar_points.push_back(lidar_point);
    }
    //将相机坐标系下的点转换至像素平面 并显示
    cv::Mat img_show = img.clone();
    for(int i=0; i<lidar_points.size(); ++i)
    {
      Eigen::Vector2d uv = cam2pixel(Rcl*lidar_points[i]+tcl);
      if(uv(0)>4 && uv(0)<img_show.cols-4 && uv(1)>4 && uv(1)<img_show.rows-4)
      {
          cv::circle(img_show, cv::Point2d(uv(0),uv(1)), 3, cv::Scalar(0,255,0), 2);
//          std::cout << "yesok!" << std::endl;
      }
    }
    cv::imshow("calib_check", img_show);
    cv::waitKey(0);

    return 0;
}
