#ifndef LIDAR_CAMERA_CALIB_HPP
#define LIDAR_CAMERA_CALIB_HPP

#include "CustomMsg.h"
#include "common.h"
#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sstream>
#include <std_msgs/Header.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <unordered_map>

#define calib
#define online
class Calibration {
public:
  ros::NodeHandle nh_;
  ros::Publisher rgb_cloud_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("rgb_cloud", 1);
  ros::Publisher init_rgb_cloud_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("init_rgb_cloud", 1);
  ros::Publisher planner_cloud_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("planner_cloud", 1);
  ros::Publisher line_cloud_pub_ =
      nh_.advertise<sensor_msgs::PointCloud2>("line_cloud", 1);
  ros::Publisher image_pub_ =
      nh_.advertise<sensor_msgs::Image>("camera_image", 1);
  enum ProjectionType { DEPTH, INTENSITY, BOTH };
  enum Direction { UP, DOWN, LEFT, RIGHT };
  std::string lidar_topic_name_ = "";
  std::string image_topic_name_ = "";

  int rgb_edge_minLen_ = 200;
  int rgb_canny_threshold_ = 20;
  int min_depth_ = 2.5;
  int max_depth_ = 50;
  int plane_max_size_ = 5;
  float detect_line_threshold_ = 0.02;
  int line_number_ = 0;
  int color_intensity_threshold_ = 5;
  Eigen::Vector3d adjust_euler_angle_;
  //void spaceToPlane(std::vector<cv::Point3f> &P3list, std::vector<cv::Point2f> &P2list);
  Calibration(const std::string &image_file, const std::string &pcd_file,const std::string &debug_dir,
              const std::string &calib_config_file);
  void loadImgAndPointcloud(const std::string bag_path,
                            pcl::PointCloud<pcl::PointXYZI>::Ptr &origin_cloud,
                            cv::Mat &rgb_img);
  bool loadCameraConfig(const std::string &camera_file);
  bool loadCalibConfig(const std::string &config_file);
  bool loadConfig(const std::string &configFile);
  bool checkFov(const cv::Point2d &p);
  void colorCloud(const Vector6d &extrinsic_params, const int density,
                  const cv::Mat &rgb_img,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr &color_cloud);
  void edgeDetector(const int &canny_threshold, const int &edge_threshold,const std::string &debug_dir,
                    const cv::Mat &src_img, cv::Mat &edge_img,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud);
  void lidarToSphere(const std::string &pcd_file,const std::string &debug_dir); //获取雷达深度图提取深度非连续边缘

  void projection(const Vector6d &extrinsic_params,
                  const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
                  const ProjectionType projection_type, const bool is_fill_img,
                  cv::Mat &projection_img);
  void calcLine(const std::vector<Plane> &plane_list, const double voxel_size,
                const Eigen::Vector3d origin,
                std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list);
  cv::Mat fillImg(const cv::Mat &input_img, const Direction first_direct,
                  const Direction second_direct);
  void buildPnp(const Vector6d &extrinsic_params, const int dis_threshold,
                const bool show_residual,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d,
                const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d,
                std::vector<PnPData> &pnp_list);
  void
  buildVPnp(const Vector6d &extrinsic_params, const int dis_threshold,
            const bool show_residual,
            const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d,
            std::vector<VPnPData> &pnp_list);

  cv::Mat
  getConnectImg(const int dis_threshold,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr &rgb_edge_cloud,
                const pcl::PointCloud<pcl::PointXYZ>::Ptr &depth_edge_cloud);
  cv::Mat getProjectionImg(const Vector6d &extrinsic_params);
  void initVoxel(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                 const float voxel_size,
                 std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map);
  void LiDAREdgeExtraction(
      const std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map,
      const float ransac_dis_thre, const int plane_size_threshold,
      pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d);
  void calcDirection(const std::vector<Eigen::Vector2d> &points,
                     Eigen::Vector2d &direction);
  void calcResidual(const Vector6d &extrinsic_params,
                    const std::vector<VPnPData> vpnp_list,
                    std::vector<float> &residual_list);
  void calcCovarance(const Vector6d &extrinsic_params,
                     const VPnPData &vpnp_point, const float pixel_inc,
                     const float range_inc, const float degree_inc,
                     Eigen::Matrix2f &covarance);
  
  // 相机内参
  float fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_, k3_, s_;
  int width_, height_;
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  cv::Mat init_extrinsic_;

  typedef std::pair<int, int> Pair;    //use for get lidardepth image 
  Pair kFlatImageSize = {2160, 3840};    //深度图分辨率
  const float kRadPerPix = (M_PI * 2) / kFlatImageSize.second;  

  int is_use_custom_msg_;
  float voxel_size_ = 1.0;
  float down_sample_size_ = 0.02;
  float ransac_dis_threshold_ = 0.02;
  float plane_size_threshold_ = 60;
  float theta_min_;
  float theta_max_;
  float direction_theta_min_;
  float direction_theta_max_;
  float min_line_dis_threshold_ = 0.03;
  float max_line_dis_threshold_ = 0.06;

  cv::Mat rgb_image_;
  cv::Mat image_;
  cv::Mat grey_image_;
  // 裁剪后的灰度图像
  cv::Mat cut_grey_image_;

  // 初始旋转矩阵
  Eigen::Matrix3d init_rotation_matrix_;
  // 初始平移向量
  Eigen::Vector3d init_translation_vector_;

  // 存储从pcd/bag处获取的原始点云
  pcl::PointCloud<pcl::PointXYZI>::Ptr raw_lidar_cloud_;

  // 存储平面交接点云
  pcl::PointCloud<pcl::PointXYZI>::Ptr plane_line_cloud_;
  std::vector<int> plane_line_number_;
  // 存储RGB图像边缘点的2D点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr rgb_egde_cloud_;
  // 存储LiDAR Depth/Intensity图像边缘点的2D点云
  pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_edge_cloud_;
  struct DepthEdgeInfo {
    Eigen::Vector2f tangent;
    float front_depth;
    float back_depth;
  };
  pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_depth_edge_cloud_;
  std::vector<DepthEdgeInfo> lidar_depth_edge_info_;
};

Calibration::Calibration(const std::string &image_file,
                         const std::string &pcd_file,
                         const std::string &debug_dir,
                         const std::string &calib_config_file) {

  loadCalibConfig(calib_config_file);

  image_ = cv::imread(image_file, cv::IMREAD_UNCHANGED);
  if (!image_.data) {
    std::string msg = "Can not load image from " + image_file;
    ROS_ERROR_STREAM(msg.c_str());
    exit(-1);
  } else {
    std::string msg = "Sucessfully load image!";
    ROS_INFO_STREAM(msg.c_str());
  }
  width_ = image_.cols;
  height_ = image_.rows;
  // check rgb or gray
  if (image_.type() == CV_8UC1) {
    grey_image_ = image_;
  } else if (image_.type() == CV_8UC3) {
    cv::cvtColor(image_, grey_image_, cv::COLOR_BGR2GRAY);
  } else {
    std::string msg = "Unsupported image type, please use CV_8UC3 or CV_8UC1";
    ROS_ERROR_STREAM(msg.c_str());
    exit(-1);
  }
  cv::Mat edge_image;

  edgeDetector(rgb_canny_threshold_, rgb_edge_minLen_, debug_dir, grey_image_, edge_image,
               rgb_egde_cloud_);
  std::string msg = "Sucessfully extract edge from image, edge size:" +
                    std::to_string(rgb_egde_cloud_->size());
  ROS_INFO_STREAM(msg.c_str());

  lidar_depth_edge_cloud_ =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  lidar_depth_edge_info_.clear();

  raw_lidar_cloud_ =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  ROS_INFO_STREAM("Loading point cloud from pcd file.");
  if (!pcl::io::loadPCDFile(pcd_file, *raw_lidar_cloud_)) {
    // down_sampling_voxel(*raw_lidar_cloud_, 0.02);
    std::string msg = "Sucessfully load pcd, pointcloud size: " +
                      std::to_string(raw_lidar_cloud_->size());
    ROS_INFO_STREAM(msg.c_str());
  } else {
    std::string msg = "Unable to load " + pcd_file;
    ROS_ERROR_STREAM(msg.c_str());
    exit(-1);
  }

  Eigen::Vector3d lwh(50, 50, 30);
  Eigen::Vector3d origin(0, -25, -10);
  std::vector<VoxelGrid> voxel_list;
  std::unordered_map<VOXEL_LOC, Voxel *> voxel_map;
  initVoxel(raw_lidar_cloud_, voxel_size_, voxel_map);
  LiDAREdgeExtraction(voxel_map, ransac_dis_threshold_, plane_size_threshold_,
                      plane_line_cloud_);    //先注释
};

bool Calibration::loadCameraConfig(const std::string &camera_file) {
  cv::FileStorage cameraSettings(camera_file, cv::FileStorage::READ);
  if (!cameraSettings.isOpened()) {
    std::cerr << "Failed to open camera settings file at " << camera_file
              << std::endl;
    exit(-1);
  }
  width_ = cameraSettings["Camera.width"];
  height_ = cameraSettings["Camera.height"];
  cameraSettings["CameraMat"] >> camera_matrix_;
  cameraSettings["DistCoeffs"] >> dist_coeffs_;
  fx_ = camera_matrix_.at<double>(0, 0);
  cx_ = camera_matrix_.at<double>(0, 2);
  fy_ = camera_matrix_.at<double>(1, 1);
  cy_ = camera_matrix_.at<double>(1, 2);
  k1_ = dist_coeffs_.at<double>(0, 0);
  k2_ = dist_coeffs_.at<double>(0, 1);
  p1_ = dist_coeffs_.at<double>(0, 2);
  p2_ = dist_coeffs_.at<double>(0, 3);
  k3_ = dist_coeffs_.at<double>(0, 4);
  std::cout << "Camera Matrix: " << std::endl << camera_matrix_ << std::endl;
  std::cout << "Distortion Coeffs: " << std::endl << dist_coeffs_ << std::endl;
  return true;
};

bool Calibration::loadCalibConfig(const std::string &config_file) {
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened()) {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    exit(-1);
  } else {
    ROS_INFO("Sucessfully load calib config file");
  }
  fSettings["ExtrinsicMat"] >> init_extrinsic_;
  init_rotation_matrix_ << init_extrinsic_.at<double>(0, 0),
      init_extrinsic_.at<double>(0, 1), init_extrinsic_.at<double>(0, 2),
      init_extrinsic_.at<double>(1, 0), init_extrinsic_.at<double>(1, 1),
      init_extrinsic_.at<double>(1, 2), init_extrinsic_.at<double>(2, 0),
      init_extrinsic_.at<double>(2, 1), init_extrinsic_.at<double>(2, 2);
  init_translation_vector_ << init_extrinsic_.at<double>(0, 3),
      init_extrinsic_.at<double>(1, 3), init_extrinsic_.at<double>(2, 3);
  rgb_canny_threshold_ = fSettings["Canny.gray_threshold"];
  rgb_edge_minLen_ = fSettings["Canny.len_threshold"];
  voxel_size_ = fSettings["Voxel.size"];
  down_sample_size_ = fSettings["Voxel.down_sample_size"];
  plane_size_threshold_ = fSettings["Plane.min_points_size"];
  plane_max_size_ = fSettings["Plane.max_size"];
  ransac_dis_threshold_ = fSettings["Ransac.dis_threshold"];
  min_line_dis_threshold_ = fSettings["Edge.min_dis_threshold"];
  max_line_dis_threshold_ = fSettings["Edge.max_dis_threshold"];
  theta_min_ = fSettings["Plane.normal_theta_min"];
  theta_max_ = fSettings["Plane.normal_theta_max"];
  theta_min_ = cos(DEG2RAD(theta_min_));
  theta_max_ = cos(DEG2RAD(theta_max_));
  direction_theta_min_ = cos(DEG2RAD(30.0));
  direction_theta_max_ = cos(DEG2RAD(150.0));
  color_intensity_threshold_ = fSettings["Color.intensity_threshold"];
  return true;
};

// Color the point cloud by rgb image using given extrinsic
void Calibration::colorCloud(
    const Vector6d &extrinsic_params, const int density,
    const cv::Mat &input_image,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr &color_cloud) {
  cv::Mat rgb_img;
  if (input_image.type() == CV_8UC3) {
    rgb_img = input_image;
  } else if (input_image.type() == CV_8UC1) {
    cv::cvtColor(input_image, rgb_img, cv::COLOR_GRAY2BGR);
  }
  std::vector<cv::Point3f> pts_3d;
  for (size_t i = 0; i < lidar_cloud->size(); i += density) {
    pcl::PointXYZI point = lidar_cloud->points[i];
    float depth = sqrt(pow(point.x, 2) + pow(point.y, 2) + pow(point.z, 2));
    if (depth > 2 && depth < 50 &&
        point.intensity >= color_intensity_threshold_) {
      pts_3d.emplace_back(cv::Point3f(point.x, point.y, point.z));
    }
  }
  Eigen::AngleAxisd rotation_vector3;
  rotation_vector3 =
      Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  cv::Mat distortion_coeff =
      (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
  cv::Mat r_vec =
      (cv::Mat_<double>(3, 1)
           << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
  cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                   extrinsic_params[4], extrinsic_params[5]);
  std::vector<cv::Point2f> pts_2d;
  cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                    pts_2d);
  //spaceToPlane(pts_3d,pts_2d);
  int image_rows = rgb_img.rows;
  int image_cols = rgb_img.cols;
  color_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  for (size_t i = 0; i < pts_2d.size(); i++) {
    if (pts_2d[i].x >= 0 && pts_2d[i].x < image_cols && pts_2d[i].y >= 0 &&
        pts_2d[i].y < image_rows) {
      cv::Scalar color =
          rgb_img.at<cv::Vec3b>((int)pts_2d[i].y, (int)pts_2d[i].x);
      if (color[0] == 0 && color[1] == 0 && color[2] == 0) {
        continue;
      }
      if (pts_3d[i].x > 100) {
        continue;
      }
      pcl::PointXYZRGB p;
      p.x = pts_3d[i].x;
      p.y = pts_3d[i].y;
      p.z = pts_3d[i].z;
      // p.a = 255;
      p.b = color[0];
      p.g = color[1];
      p.r = color[2];
      color_cloud->points.push_back(p);
    }
  }
  color_cloud->width = color_cloud->points.size();
  color_cloud->height = 1;
}


//直角坐标下雷达点云转换到极坐标系下，再从极坐标系下转化到平面图像（深度图）  
void Calibration::lidarToSphere(const std::string &pcd_file,
                         const std::string &debug_dir) {
    cout<<"lidarToSphere start"<<endl;
    float theta_min = M_PI, theta_max = -M_PI;
    float phi_min = std::numeric_limits<float>::max();
    float phi_max = -std::numeric_limits<float>::max();
    pcl::io::loadPCDFile(pcd_file, *raw_lidar_cloud_);

    pcl::PointCloud<pcl::PointXYZI>::Ptr polar_cloud(
          new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*raw_lidar_cloud_, *polar_cloud);
    for (auto &point : polar_cloud->points) {
        float radius = point.getVector3fMap().norm(); //模长   
        float phi = atan2(point.y, point.x);    //方位角
        float theta = acos(point.z / radius);   //极角
        point.x = theta;
        point.y = phi;
        point.z = radius;
        if (theta > theta_max) { theta_max = theta;}
        else if (theta < theta_min) { theta_min = theta;}
    //      if (point.y < phi_min) {
    //     phi_min = point.y;
    // }
    // if (point.y > phi_max) {
    //     phi_max = point.y;
    // }
    }

// void Calibration::sphereToPlane(CloudI::Ptr& polar_cloud) {   //极坐标点云转换成平面图像

    typedef pcl::PointXYZI  PointI;
    pcl::KdTreeFLANN<PointI> kdtree;
    pcl::PointCloud<pcl::PointXYZI>::Ptr polar_flat_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*polar_cloud, *polar_flat_cloud);

    cout << "----- LiDAR: SphereToPlane -----" << endl;   //遍历全部点云，将极坐标转换为平面坐标
    int image_rows=2160;     //3240;
    int image_cols=3840;          //5760;
    // 定义数据容器
    // cv::Mat flat_img = cv::Mat::zeros(image_.rows, image_.cols, CV_8U); 这段函数中后面的image_.rows改成image_rows
    cv::Mat flat_img = cv::Mat::zeros(image_rows, image_cols, CV_8U);

    cout << "image_.rows: height" << image_rows << endl;
    cout << "image_.col: width: " << image_cols << endl;

    // 定义标签和地图
    typedef vector<int> Tags;
    vector<vector<Tags>> tagsMap;
    vector<vector<Tags>> tags_map(image_rows, vector<Tags>(image_cols));

//     // 遍历极坐标点云
//     for (size_t i = 0; i < polar_cloud->points.size(); ++i) {
//         const auto &point = polar_cloud->points[i];
//         int u = static_cast<int>((theta_max - point.x) / kRadPerPix);
//         int v = static_cast<int>((point.y + M_PI) / kRadPerPix );

//         if (u >= 0 && u < image_rows && v >= 0 && v < image_cols) {
//             flat_img.at<uchar>(u, v) =2*static_cast<uchar>(point.z); //static_cast<uchar>(point.intensity) ;  
//             tags_map[u][v].push_back(i); // 将点的索引添加到 tag 中
//         }
// }
    lidar_depth_edge_info_.clear();
    lidar_depth_edge_info_.reserve(polar_cloud->points.size());
    if (lidar_depth_edge_cloud_) {
      lidar_depth_edge_cloud_->clear();
    }

    // 初始化图像缓存
    cv::Mat intensity_img = cv::Mat::zeros(image_rows, image_cols, CV_32F);
    cv::Mat count_img = cv::Mat::zeros(image_rows, image_cols, CV_32F);
    cv::Mat min_depth_img(
        image_rows, image_cols, CV_32F,
        cv::Scalar(std::numeric_limits<float>::infinity()));
    cv::Mat max_depth_img(
        image_rows, image_cols, CV_32F,
        cv::Scalar(-std::numeric_limits<float>::infinity()));
    cv::Mat depth_valid_mask(image_rows, image_cols, CV_8U, cv::Scalar(0));

    // 遍历点云
    for (size_t i = 0; i < polar_cloud->points.size(); ++i) {
      const auto &point = polar_cloud->points[i];
      int u = static_cast<int>((theta_max - point.x) / kRadPerPix);
      int v = static_cast<int>((point.y + M_PI) / kRadPerPix);

      if (u >= 0 && u < image_rows && v >= 0 && v < image_cols) {
        intensity_img.at<float>(u, v) += point.intensity;
        count_img.at<float>(u, v) += 1.0f;

        float radius = point.z;
        float &min_depth = min_depth_img.at<float>(u, v);
        float &max_depth = max_depth_img.at<float>(u, v);
        if (radius < min_depth) {
          min_depth = radius;
        }
        if (radius > max_depth) {
          max_depth = radius;
        }
        depth_valid_mask.at<uchar>(u, v) = 255;
        tags_map[u][v].push_back(i);
      }
    }

    for (int u = 0; u < intensity_img.rows; ++u) {
      for (int v = 0; v < intensity_img.cols; ++v) {
        if (count_img.at<float>(u, v) > 0) {
          intensity_img.at<float>(u, v) /= count_img.at<float>(u, v);
        }
      }
    }

    float nan_value = std::numeric_limits<float>::quiet_NaN();
    min_depth_img.setTo(nan_value, depth_valid_mask == 0);
    max_depth_img.setTo(nan_value, depth_valid_mask == 0);

    cv::Mat depth_for_gradient = min_depth_img.clone();
    depth_for_gradient.setTo(0, depth_valid_mask == 0);
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Mat grad_mag;
    cv::Sobel(depth_for_gradient, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(depth_for_gradient, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);

    /** add the tags_map of this specific pose to maps **/
    tagsMap = tags_map;
    cv::imwrite(debug_dir + "intensity_img.png", intensity_img);


    // 调用Python边缘检测脚本
string cmd_str = "python3 " + debug_dir + "cannyDectector.py";
cout << "----- LiDAR: Running Edge Detection Script -----" << endl;
int status = system(cmd_str.c_str());
if(status != 0) {
    cerr << "Error running Python edge detection script" << endl;
    return;
}

    cv::Mat edge_img = cv::imread(debug_dir + "edge_img.png", cv::IMREAD_UNCHANGED); //读取雷达深度图的边缘图像 
    // pcl::PointCloud<pcl::PointXYZI>::Ptr lidar_edge_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // std::string lidar_edge_cloud_path = debug_dir + "lidar_edge_cloud.pcd";

//void Calibration::generateEdgeCloud(cv::Mat &edge_img, const pcl::PointCloud<pcl::PointXYZI>::Ptr &raw_lidar_cloud_, pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_edge_cloud, const std::string &lidar_edge_cloud_path) 
   cout << "----- LiDAR: GenerateEdgeCloud -----" << endl;

    // 创建一个新的点云对象，用于存储边缘点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_xyzi(new pcl::PointCloud<pcl::PointXYZI>);


// 遍历图像中的每个像素，筛选具有稳定深度跳变与方向的边缘
cout << "----- 遍历图像中的每个像素 -----" << endl;
const float kGradientMagnitudeThreshold = 0.02f;
const float kDepthJumpThreshold = 0.3f;
const float kAssociationTolerance = 0.15f;
const float kSampleStep = 1.0f;

auto sampleDepth = [&](float row, float col) -> float {
  int sample_u = static_cast<int>(std::round(row));
  int sample_v = static_cast<int>(std::round(col));
  if (sample_u < 0 || sample_u >= edge_img.rows || sample_v < 0 ||
      sample_v >= edge_img.cols) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (!depth_valid_mask.at<uchar>(sample_u, sample_v)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return min_depth_img.at<float>(sample_u, sample_v);
};

for (int u = 0; u < edge_img.rows; ++u) {
  for (int v = 0; v < edge_img.cols; ++v) {
    if (edge_img.at<uchar>(u, v) <= 127) {
      continue;
    }
    if (!depth_valid_mask.at<uchar>(u, v)) {
      continue;
    }

    float front_depth = min_depth_img.at<float>(u, v);
    if (!std::isfinite(front_depth)) {
      continue;
    }

    float gradient_mag = grad_mag.at<float>(u, v);
    if (!std::isfinite(gradient_mag) || gradient_mag < kGradientMagnitudeThreshold) {
      continue;
    }

    cv::Vec2f normal(grad_x.at<float>(u, v), grad_y.at<float>(u, v));
    float normal_norm = cv::norm(normal);
    if (normal_norm < 1e-6f) {
      continue;
    }
    normal /= normal_norm;

    float background_depth = std::numeric_limits<float>::quiet_NaN();
    float local_max_depth = max_depth_img.at<float>(u, v);
    if (std::isfinite(local_max_depth) &&
        (local_max_depth - front_depth) > kDepthJumpThreshold) {
      background_depth = local_max_depth;
    } else {
      float forward_depth = sampleDepth(u + normal[1] * kSampleStep,
                                        v + normal[0] * kSampleStep);
      float backward_depth = sampleDepth(u - normal[1] * kSampleStep,
                                         v - normal[0] * kSampleStep);
      if (std::isfinite(forward_depth) && forward_depth > front_depth) {
        background_depth = forward_depth;
      }
      if (std::isfinite(backward_depth) && backward_depth > front_depth) {
        if (!std::isfinite(background_depth) ||
            backward_depth < background_depth) {
          background_depth = backward_depth;
        }
      }
    }

    if (!std::isfinite(background_depth)) {
      continue;
    }
    if ((background_depth - front_depth) < kDepthJumpThreshold) {
      continue;
    }

    cv::Vec2f tangent(-normal[1], normal[0]);
    float tangent_norm = cv::norm(tangent);
    if (tangent_norm < 1e-6f) {
      continue;
    }
    tangent /= tangent_norm;

    const std::vector<int> *pixel_indices = &tags_map[u][v];
    int assoc_u = u;
    int assoc_v = v;
    if (pixel_indices->empty()) {
      float best_diff = std::numeric_limits<float>::infinity();
      for (int du = -1; du <= 1; ++du) {
        for (int dv = -1; dv <= 1; ++dv) {
          int nu = u + du;
          int nv = v + dv;
          if (nu < 0 || nu >= edge_img.rows || nv < 0 || nv >= edge_img.cols) {
            continue;
          }
          if (tags_map[nu][nv].empty()) {
            continue;
          }
          float neighbor_depth = min_depth_img.at<float>(nu, nv);
          if (!std::isfinite(neighbor_depth)) {
            continue;
          }
          float diff = std::abs(neighbor_depth - front_depth);
          if (diff < best_diff) {
            best_diff = diff;
            pixel_indices = &tags_map[nu][nv];
            assoc_u = nu;
            assoc_v = nv;
          }
        }
      }
    }
    if (pixel_indices->empty()) {
      continue;
    }

    float assoc_depth = min_depth_img.at<float>(assoc_u, assoc_v);
    if (!std::isfinite(assoc_depth)) {
      assoc_depth = front_depth;
    }

    bool pushed_point = false;
    for (const auto &index : *pixel_indices) {
      const auto &origin_point = raw_lidar_cloud_->points[index];
      float radius = origin_point.getVector3fMap().norm();
      if (std::fabs(radius - assoc_depth) > kAssociationTolerance) {
        continue;
      }
      edge_xyzi->points.push_back(origin_point);
      DepthEdgeInfo info{Eigen::Vector2f(tangent[0], tangent[1]), assoc_depth,
                         background_depth};
      lidar_depth_edge_info_.push_back(info);
      pushed_point = true;
    }

    if (!pushed_point) {
      float best_depth = std::numeric_limits<float>::infinity();
      int best_index = -1;
      for (const auto &index : *pixel_indices) {
        float radius = raw_lidar_cloud_->points[index].getVector3fMap().norm();
        if (radius < best_depth) {
          best_depth = radius;
          best_index = index;
        }
      }
      if (best_index >= 0) {
        edge_xyzi->points.push_back(raw_lidar_cloud_->points[best_index]);
        DepthEdgeInfo info{Eigen::Vector2f(tangent[0], tangent[1]), best_depth,
                           background_depth};
        lidar_depth_edge_info_.push_back(info);
      }
    }
  }
}
cout << "----- 遍历完成 -----" << endl;

std::cout << "Size of edge_xyzi: " << edge_xyzi->points.size() << std::endl;
std::cout << "Stored depth edge info: " << lidar_depth_edge_info_.size()
          << std::endl;

    // 将处理后的点云复制到另一个点云对象中
    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_discon(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*edge_xyzi, *edge_discon);
    lidar_depth_edge_cloud_ = edge_discon;
    if (lidar_depth_edge_info_.size() != edge_discon->points.size()) {
      ROS_WARN_STREAM("Depth edge info size mismatch: "
                      << lidar_depth_edge_info_.size() << " vs "
                      << edge_discon->points.size());
    }
    
    // 创建一个新的点云用于存储拼接后的点云  这一段用于将非连续边缘点云添加到平面点云中进行之后的优化
    pcl::PointCloud<pcl::PointXYZI>::Ptr concat_cloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr marked_edge_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      // 为体素提取到的边缘点云分配颜色，

    // pcl::io::loadPCDFile(pcd_file, *marked_edge_cloud);  //原始的彩色点云
  for (const auto &point : plane_line_cloud_->points) {
    pcl::PointXYZRGB p;
    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    p.r = 0;  // 体素提取的边缘点云颜色（绿色）
    p.g = 0;
    p.b = 255;
    // 重复添加边缘点，以增加点的大小
    for (int i = 0; i < 5; ++i) {
    marked_edge_cloud->points.push_back(p);
    }
  }
    // 将原始点云复制到新的点云  注释这段代码即只用voxel提取的深度连续边缘
    *concat_cloud = *plane_line_cloud_;
    // 将边缘点云插入到拼接后的点云中
    concat_cloud->insert(concat_cloud->end(), edge_discon->begin(), edge_discon->end());
     // 更新当前的平面点云为拼接后的点云 深度非连续+voxel的深度连续边缘
    plane_line_cloud_ = concat_cloud;

// //只用深度非连续边缘
  //plane_line_cloud_ = edge_discon;
  //与原始点云融合显示
   
      // 将原始点云转换为彩色点云
  for (const auto &point : raw_lidar_cloud_->points) {
    pcl::PointXYZRGB p;
    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    p.r = 255;  // 原始点云颜色（白色）
    p.g = 255;
    p.b = 255;
    marked_edge_cloud->points.push_back(p);
  }
  // 为边缘点云分配颜色，并加粗表示
  for (const auto &point : edge_discon->points) {
    pcl::PointXYZRGB p;
    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    p.r = 255;  // 边缘点云颜色（红色）
    p.g = 0;
    p.b = 0;
    // 重复添加边缘点，以增加点的大小
    // for (int i = 0; i < 10; ++i) {
    marked_edge_cloud->points.push_back(p);
    // }
  }



  // 更新点云的宽度和高度
  marked_edge_cloud->width = static_cast<uint32_t>(marked_edge_cloud->points.size());
  marked_edge_cloud->height = 1;
  marked_edge_cloud->is_dense = true;

  //   // // 将处理后的点云复制到成员变量 lidarEdgeCloud
  // pcl::io::savePCDFileBinary(debug_dir + "all_marked_edge_cloud.pcd", *marked_edge_cloud);          
  }


// Detect edge by canny, and filter by edge length
void Calibration::edgeDetector(
    const int &canny_threshold, const int &edge_threshold,const std::string &debug_dir,
    const cv::Mat &src_img, cv::Mat &edge_img,
    pcl::PointCloud<pcl::PointXYZ>::Ptr &edge_cloud) {
  int gaussian_size = 5;
  cv::GaussianBlur(src_img, src_img, cv::Size(gaussian_size, gaussian_size), 0,
                   0);
  cv::Mat canny_result = cv::Mat::zeros(height_, width_, CV_8UC1);
  cv::Canny(src_img, canny_result, canny_threshold, canny_threshold * 3, 3,
            true);
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(canny_result, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  edge_img = cv::Mat::zeros(height_, width_, CV_8UC1);

  edge_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < contours.size(); i++) {
    if (contours[i].size() > edge_threshold) {
      cv::Mat debug_img = cv::Mat::zeros(height_, width_, CV_8UC1);
      for (size_t j = 0; j < contours[i].size(); j++) {
        pcl::PointXYZ p;
        p.x = contours[i][j].x;
        p.y = -contours[i][j].y;
        p.z = 0;
        edge_img.at<uchar>(-p.y, p.x) = 255;
      }
    }
  }
  for (int x = 0; x < edge_img.cols; x++) {
    for (int y = 0; y < edge_img.rows; y++) {
      if (edge_img.at<uchar>(y, x) == 255) {
        pcl::PointXYZ p;
        p.x = x;
        p.y = -y;
        p.z = 0;
        edge_cloud->points.push_back(p);
      }
    }
  }
  edge_cloud->width = edge_cloud->points.size();
  edge_cloud->height = 1;
  cv::imwrite(debug_dir+"canny_result.png", canny_result);
  cv::imwrite(debug_dir+"edge_result.png", edge_img);
  cv::waitKey();
}

void Calibration::projection(
    const Vector6d &extrinsic_params,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_cloud,
    const ProjectionType projection_type, const bool is_fill_img,
    cv::Mat &projection_img) {
  std::vector<cv::Point3f> pts_3d;
  std::vector<float> intensity_list;
  Eigen::AngleAxisd rotation_vector3;
  rotation_vector3 =
      Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
  for (size_t i = 0; i < lidar_cloud->size(); i++) {
    pcl::PointXYZI point_3d = lidar_cloud->points[i];
    float depth =
        sqrt(pow(point_3d.x, 2) + pow(point_3d.y, 2) + pow(point_3d.z, 2));
    if (depth > min_depth_ && depth < max_depth_) {
      pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
      intensity_list.emplace_back(lidar_cloud->points[i].intensity);
    }
  }
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  cv::Mat distortion_coeff =
      (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
  cv::Mat r_vec =
      (cv::Mat_<double>(3, 1)
           << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
  cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                   extrinsic_params[4], extrinsic_params[5]);
  // project 3d-points into image view
  std::vector<cv::Point2f> pts_2d;
  cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                    pts_2d);
  //spaceToPlane(pts_3d,pts_2d);
  cv::Mat image_project = cv::Mat::zeros(height_, width_, CV_16UC1);
  cv::Mat rgb_image_project = cv::Mat::zeros(height_, width_, CV_8UC3);
  for (size_t i = 0; i < pts_2d.size(); ++i) {
    cv::Point2f point_2d = pts_2d[i];
    if (point_2d.x <= 0 || point_2d.x >= width_ || point_2d.y <= 0 ||
        point_2d.y >= height_) {
      continue;
    } else {
      // test depth and intensity both
      if (projection_type == DEPTH) {
        float depth = sqrt(pow(pts_3d[i].x, 2) + pow(pts_3d[i].y, 2) +
                           pow(pts_3d[i].z, 2));
        float intensity = intensity_list[i];
        float depth_weight = 1;
        float grey = depth_weight * depth / max_depth_ * 65535 +
                     (1 - depth_weight) * intensity / 150 * 65535;
        if (image_project.at<ushort>(point_2d.y, point_2d.x) == 0) {
          image_project.at<ushort>(point_2d.y, point_2d.x) = grey;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[0] =
              depth / max_depth_ * 255;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[1] =
              intensity / 150 * 255;
        } else if (depth < image_project.at<ushort>(point_2d.y, point_2d.x)) {
          image_project.at<ushort>(point_2d.y, point_2d.x) = grey;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[0] =
              depth / max_depth_ * 255;
          rgb_image_project.at<cv::Vec3b>(point_2d.y, point_2d.x)[1] =
              intensity / 150 * 255;
        }

      } else {
        float intensity = intensity_list[i];
        if (intensity > 100) {
          intensity = 65535;
        } else {
          intensity = (intensity / 150.0) * 65535;
        }
        image_project.at<ushort>(point_2d.y, point_2d.x) = intensity;
      }
    }
  }
  cv::Mat grey_image_projection;
  cv::cvtColor(rgb_image_project, grey_image_projection, cv::COLOR_BGR2GRAY);

  image_project.convertTo(image_project, CV_8UC1, 1 / 256.0);
  if (is_fill_img) {
    for (int i = 0; i < 5; i++) {
      image_project = fillImg(image_project, UP, LEFT);
    }
  }
  if (is_fill_img) {
    for (int i = 0; i < 5; i++) {
      grey_image_projection = fillImg(grey_image_projection, UP, LEFT);
    }
  }
  projection_img = image_project.clone();
}

// 填补雷达深度图像
cv::Mat Calibration::fillImg(const cv::Mat &input_img,
                             const Direction first_direct,
                             const Direction second_direct) {
  cv::Mat fill_img = input_img.clone();
  for (int y = 2; y < input_img.rows - 2; y++) {
    for (int x = 2; x < input_img.cols - 2; x++) {
      if (input_img.at<uchar>(y, x) == 0) {
        if (input_img.at<uchar>(y - 1, x) != 0) {
          fill_img.at<uchar>(y, x) = input_img.at<uchar>(y - 1, x);
        } else {
          if ((input_img.at<uchar>(y, x - 1)) != 0) {
            fill_img.at<uchar>(y, x) = input_img.at<uchar>(y, x - 1);
          }
        }
      } else {
        int left_depth = input_img.at<uchar>(y, x - 1);
        int right_depth = input_img.at<uchar>(y, x + 1);
        int up_depth = input_img.at<uchar>(y + 1, x);
        int down_depth = input_img.at<uchar>(y - 1, x);
        int current_depth = input_img.at<uchar>(y, x);
        if ((current_depth - left_depth) > 5 &&
            (current_depth - right_depth) > 5 && left_depth != 0 &&
            right_depth != 0) {
          fill_img.at<uchar>(y, x) = (left_depth + right_depth) / 2;
        } else if ((current_depth - up_depth) > 5 &&
                   (current_depth - down_depth) > 5 && up_depth != 0 &&
                   down_depth != 0) {
          fill_img.at<uchar>(y, x) = (up_depth + down_depth) / 2;
        }
      }
    }
  }
  return fill_img;
}

cv::Mat Calibration::getConnectImg(
    const int dis_threshold,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &rgb_edge_cloud,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &depth_edge_cloud) {
  cv::Mat connect_img = cv::Mat::zeros(height_, width_, CV_8UC3);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  kdtree->setInputCloud(rgb_edge_cloud);
  tree_cloud = rgb_edge_cloud;
  for (size_t i = 0; i < depth_edge_cloud->points.size(); i++) {
    cv::Point2d p2(depth_edge_cloud->points[i].x,
                   -depth_edge_cloud->points[i].y);
    if (checkFov(p2)) {
      pcl::PointXYZ p = depth_edge_cloud->points[i];
      search_cloud->points.push_back(p);
    }
  }

  int line_count = 0;
  // 指定近邻个数
  int K = 1;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < search_cloud->points.size(); i++) {
    pcl::PointXYZ searchPoint = search_cloud->points[i];
    if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      for (int j = 0; j < K; j++) {
        float distance = sqrt(
            pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
            pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
        if (distance < dis_threshold) {
          cv::Scalar color = cv::Scalar(0, 255, 0);
          line_count++;
          if ((line_count % 3) == 0) {
            cv::line(connect_img,
                     cv::Point(search_cloud->points[i].x,
                               -search_cloud->points[i].y),
                     cv::Point(tree_cloud->points[pointIdxNKNSearch[j]].x,
                               -tree_cloud->points[pointIdxNKNSearch[j]].y),
                     color, 1);
          }
        }
      }
    }
  }
  for (size_t i = 0; i < rgb_edge_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[0] = 255;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-rgb_edge_cloud->points[i].y,
                              rgb_edge_cloud->points[i].x)[2] = 0;
  }
  for (size_t i = 0; i < search_cloud->size(); i++) {
    connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                              search_cloud->points[i].x)[0] = 0;
    connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                              search_cloud->points[i].x)[1] = 0;
    connect_img.at<cv::Vec3b>(-search_cloud->points[i].y,
                              search_cloud->points[i].x)[2] = 255;
  }
  int expand_size = 2;
  cv::Mat expand_edge_img;
  expand_edge_img = connect_img.clone();
  for (int x = expand_size; x < connect_img.cols - expand_size; x++) {
    for (int y = expand_size; y < connect_img.rows - expand_size; y++) {
      if (connect_img.at<cv::Vec3b>(y, x)[0] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 255;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 0;
          }
        }
      } else if (connect_img.at<cv::Vec3b>(y, x)[2] == 255) {
        for (int xx = x - expand_size; xx <= x + expand_size; xx++) {
          for (int yy = y - expand_size; yy <= y + expand_size; yy++) {
            expand_edge_img.at<cv::Vec3b>(yy, xx)[0] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[1] = 0;
            expand_edge_img.at<cv::Vec3b>(yy, xx)[2] = 255;
          }
        }
      }
    }
  }
  return connect_img;
}

bool Calibration::checkFov(const cv::Point2d &p) {
  if (p.x > 0 && p.x < width_ && p.y > 0 && p.y < height_) {
    return true;
  } else {
    return false;
  }
}

void Calibration::initVoxel(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    const float voxel_size, std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map) {
  ROS_INFO_STREAM("Building Voxel");
  // for voxel test
  srand((unsigned)time(NULL));
  pcl::PointCloud<pcl::PointXYZRGB> test_cloud;
  for (size_t i = 0; i < input_cloud->size(); i++) {
    const pcl::PointXYZI &p_c = input_cloud->points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end()) {
      voxel_map[position]->cloud->push_back(p_c);
      pcl::PointXYZRGB p_rgb;
      p_rgb.x = p_c.x;
      p_rgb.y = p_c.y;
      p_rgb.z = p_c.z;
      p_rgb.r = voxel_map[position]->voxel_color(0);
      p_rgb.g = voxel_map[position]->voxel_color(1);
      p_rgb.b = voxel_map[position]->voxel_color(2);
      test_cloud.push_back(p_rgb);
    } else {
      Voxel *voxel = new Voxel(voxel_size);
      voxel_map[position] = voxel;
      voxel_map[position]->voxel_origin[0] = position.x * voxel_size;
      voxel_map[position]->voxel_origin[1] = position.y * voxel_size;
      voxel_map[position]->voxel_origin[2] = position.z * voxel_size;
      voxel_map[position]->cloud->push_back(p_c);
      int r = rand() % 256;
      int g = rand() % 256;
      int b = rand() % 256;
      voxel_map[position]->voxel_color << r, g, b;
    }
  }
  // sensor_msgs::PointCloud2 pub_cloud;
  // pcl::toROSMsg(test_cloud, pub_cloud);
  // pub_cloud.header.frame_id = "livox";
  // rgb_cloud_pub_.publish(pub_cloud);
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->cloud->size() > 20) {
      down_sampling_voxel(*(iter->second->cloud), 0.02);
    }
  }
}

void Calibration::LiDAREdgeExtraction(
    const std::unordered_map<VOXEL_LOC, Voxel *> &voxel_map,
    const float ransac_dis_thre, const int plane_size_threshold,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d) {
  ROS_INFO_STREAM("Extracting Lidar Edge");
  ros::Rate loop(5000);
  lidar_line_cloud_3d =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    if (iter->second->cloud->size() > 50) {
      std::vector<Plane> plane_list;
      // 创建一个体素滤波器
      pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filter(
          new pcl::PointCloud<pcl::PointXYZI>);
      pcl::copyPointCloud(*iter->second->cloud, *cloud_filter);
      //创建一个模型参数对象，用于记录结果
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      // inliers表示误差能容忍的点，记录点云序号
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
      //创建一个分割器
      pcl::SACSegmentation<pcl::PointXYZI> seg;
      // Optional,设置结果平面展示的点是分割掉的点还是分割剩下的点
      seg.setOptimizeCoefficients(true);
      // Mandatory-设置目标几何形状
      seg.setModelType(pcl::SACMODEL_PLANE);
      //分割方法：随机采样法
      seg.setMethodType(pcl::SAC_RANSAC);
      //设置误差容忍范围，也就是阈值
      if (iter->second->voxel_origin[0] < 10) {
        seg.setDistanceThreshold(ransac_dis_thre);
      } else {
        seg.setDistanceThreshold(ransac_dis_thre);
      }
      pcl::PointCloud<pcl::PointXYZRGB> color_planner_cloud;
      int plane_index = 0;
      while (cloud_filter->points.size() > 10) {
        pcl::PointCloud<pcl::PointXYZI> planner_cloud;
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        //输入点云
        seg.setInputCloud(cloud_filter);
        seg.setMaxIterations(500);
        //分割点云
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size() == 0) {
          ROS_INFO_STREAM(
              "Could not estimate a planner model for the given dataset");
          break;
        }
        extract.setIndices(inliers);
        extract.setInputCloud(cloud_filter);
        extract.filter(planner_cloud);

        if (planner_cloud.size() > plane_size_threshold) {
          pcl::PointCloud<pcl::PointXYZRGB> color_cloud;
          std::vector<unsigned int> colors;
          colors.push_back(static_cast<unsigned int>(rand() % 256));
          colors.push_back(static_cast<unsigned int>(rand() % 256));
          colors.push_back(static_cast<unsigned int>(rand() % 256));
          pcl::PointXYZ p_center(0, 0, 0);
          for (size_t i = 0; i < planner_cloud.points.size(); i++) {
            pcl::PointXYZRGB p;
            p.x = planner_cloud.points[i].x;
            p.y = planner_cloud.points[i].y;
            p.z = planner_cloud.points[i].z;
            p_center.x += p.x;
            p_center.y += p.y;
            p_center.z += p.z;
            p.r = colors[0];
            p.g = colors[1];
            p.b = colors[2];
            color_cloud.push_back(p);
            color_planner_cloud.push_back(p);
          }
          p_center.x = p_center.x / planner_cloud.size();
          p_center.y = p_center.y / planner_cloud.size();
          p_center.z = p_center.z / planner_cloud.size();
          Plane single_plane;
          single_plane.cloud = planner_cloud;
          single_plane.p_center = p_center;
          single_plane.normal << coefficients->values[0],
              coefficients->values[1], coefficients->values[2];
          single_plane.index = plane_index;
          plane_list.push_back(single_plane);
          plane_index++;
        }
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZI> cloud_f;
        extract.filter(cloud_f);
        *cloud_filter = cloud_f;
      }
      if (plane_list.size() >= 2) {
        sensor_msgs::PointCloud2 planner_cloud2;
        pcl::toROSMsg(color_planner_cloud, planner_cloud2);
        planner_cloud2.header.frame_id = "livox";
        planner_cloud_pub_.publish(planner_cloud2);
        loop.sleep();
      }

      std::vector<pcl::PointCloud<pcl::PointXYZI>> line_cloud_list;
      calcLine(plane_list, voxel_size_, iter->second->voxel_origin,
               line_cloud_list);
      // ouster 5,normal 3
      if (line_cloud_list.size() > 0 && line_cloud_list.size() <= 8) {

        for (size_t cloud_index = 0; cloud_index < line_cloud_list.size();
             cloud_index++) {
          for (size_t i = 0; i < line_cloud_list[cloud_index].size(); i++) {
            pcl::PointXYZI p = line_cloud_list[cloud_index].points[i];
            plane_line_cloud_->points.push_back(p);
            sensor_msgs::PointCloud2 pub_cloud;
            pcl::toROSMsg(line_cloud_list[cloud_index], pub_cloud);
            pub_cloud.header.frame_id = "livox";
            line_cloud_pub_.publish(pub_cloud);
            loop.sleep();
            plane_line_number_.push_back(line_number_);
          }
          line_number_++;
        }
      }
    }
  }
}

void Calibration::calcLine(
    const std::vector<Plane> &plane_list, const double voxel_size,
    const Eigen::Vector3d origin,
    std::vector<pcl::PointCloud<pcl::PointXYZI>> &line_cloud_list) {
  if (plane_list.size() >= 2 && plane_list.size() <= plane_max_size_) {
    pcl::PointCloud<pcl::PointXYZI> temp_line_cloud;
    for (size_t plane_index1 = 0; plane_index1 < plane_list.size() - 1;
         plane_index1++) {
      for (size_t plane_index2 = plane_index1 + 1;
           plane_index2 < plane_list.size(); plane_index2++) {
        float a1 = plane_list[plane_index1].normal[0];
        float b1 = plane_list[plane_index1].normal[1];
        float c1 = plane_list[plane_index1].normal[2];
        float x1 = plane_list[plane_index1].p_center.x;
        float y1 = plane_list[plane_index1].p_center.y;
        float z1 = plane_list[plane_index1].p_center.z;
        float a2 = plane_list[plane_index2].normal[0];
        float b2 = plane_list[plane_index2].normal[1];
        float c2 = plane_list[plane_index2].normal[2];
        float x2 = plane_list[plane_index2].p_center.x;
        float y2 = plane_list[plane_index2].p_center.y;
        float z2 = plane_list[plane_index2].p_center.z;
        float theta = a1 * a2 + b1 * b2 + c1 * c2;
        //
        float point_dis_threshold = 0.00;
        if (theta > theta_max_ && theta < theta_min_) {
          // for (int i = 0; i < 6; i++) {
          if (plane_list[plane_index1].cloud.size() > 0 &&
              plane_list[plane_index2].cloud.size() > 0) {
            float matrix[4][5];
            matrix[1][1] = a1;
            matrix[1][2] = b1;
            matrix[1][3] = c1;
            matrix[1][4] = a1 * x1 + b1 * y1 + c1 * z1;
            matrix[2][1] = a2;
            matrix[2][2] = b2;
            matrix[2][3] = c2;
            matrix[2][4] = a2 * x2 + b2 * y2 + c2 * z2;
            // six types
            std::vector<Eigen::Vector3d> points;
            Eigen::Vector3d point;
            matrix[3][1] = 1;
            matrix[3][2] = 0;
            matrix[3][3] = 0;
            matrix[3][4] = origin[0];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.push_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 1;
            matrix[3][3] = 0;
            matrix[3][4] = origin[1];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.push_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 0;
            matrix[3][3] = 1;
            matrix[3][4] = origin[2];
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.push_back(point);
            }
            matrix[3][1] = 1;
            matrix[3][2] = 0;
            matrix[3][3] = 0;
            matrix[3][4] = origin[0] + voxel_size;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.push_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 1;
            matrix[3][3] = 0;
            matrix[3][4] = origin[1] + voxel_size;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.push_back(point);
            }
            matrix[3][1] = 0;
            matrix[3][2] = 0;
            matrix[3][3] = 1;
            matrix[3][4] = origin[2] + voxel_size;
            calc<float>(matrix, point);
            if (point[0] >= origin[0] - point_dis_threshold &&
                point[0] <= origin[0] + voxel_size + point_dis_threshold &&
                point[1] >= origin[1] - point_dis_threshold &&
                point[1] <= origin[1] + voxel_size + point_dis_threshold &&
                point[2] >= origin[2] - point_dis_threshold &&
                point[2] <= origin[2] + voxel_size + point_dis_threshold) {
              points.push_back(point);
            }
            // std::cout << "points size:" << points.size() << std::endl;
            if (points.size() == 2) {
              pcl::PointCloud<pcl::PointXYZI> line_cloud;
              pcl::PointXYZ p1(points[0][0], points[0][1], points[0][2]);
              pcl::PointXYZ p2(points[1][0], points[1][1], points[1][2]);
              float length = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                                  pow(p1.z - p2.z, 2));
              // 指定近邻个数
              int K = 1;
              // 创建两个向量，分别存放近邻的索引值、近邻的中心距
              std::vector<int> pointIdxNKNSearch1(K);
              std::vector<float> pointNKNSquaredDistance1(K);
              std::vector<int> pointIdxNKNSearch2(K);
              std::vector<float> pointNKNSquaredDistance2(K);
              pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree1(
                  new pcl::search::KdTree<pcl::PointXYZI>());
              pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree2(
                  new pcl::search::KdTree<pcl::PointXYZI>());
              kdtree1->setInputCloud(
                  plane_list[plane_index1].cloud.makeShared());
              kdtree2->setInputCloud(
                  plane_list[plane_index2].cloud.makeShared());
              for (float inc = 0; inc <= length; inc += 0.005) {
                pcl::PointXYZI p;
                p.x = p1.x + (p2.x - p1.x) * inc / length;
                p.y = p1.y + (p2.y - p1.y) * inc / length;
                p.z = p1.z + (p2.z - p1.z) * inc / length;
                p.intensity = 100;
                if ((kdtree1->nearestKSearch(p, K, pointIdxNKNSearch1,
                                             pointNKNSquaredDistance1) > 0) &&
                    (kdtree2->nearestKSearch(p, K, pointIdxNKNSearch2,
                                             pointNKNSquaredDistance2) > 0)) {
                  float dis1 =
                      pow(p.x - plane_list[plane_index1]
                                    .cloud.points[pointIdxNKNSearch1[0]]
                                    .x,
                          2) +
                      pow(p.y - plane_list[plane_index1]
                                    .cloud.points[pointIdxNKNSearch1[0]]
                                    .y,
                          2) +
                      pow(p.z - plane_list[plane_index1]
                                    .cloud.points[pointIdxNKNSearch1[0]]
                                    .z,
                          2);
                  float dis2 =
                      pow(p.x - plane_list[plane_index2]
                                    .cloud.points[pointIdxNKNSearch2[0]]
                                    .x,
                          2) +
                      pow(p.y - plane_list[plane_index2]
                                    .cloud.points[pointIdxNKNSearch2[0]]
                                    .y,
                          2) +
                      pow(p.z - plane_list[plane_index2]
                                    .cloud.points[pointIdxNKNSearch2[0]]
                                    .z,
                          2);
                  if ((dis1 <
                           min_line_dis_threshold_ * min_line_dis_threshold_ &&
                       dis2 <
                           max_line_dis_threshold_ * max_line_dis_threshold_) ||
                      ((dis1 <
                            max_line_dis_threshold_ * max_line_dis_threshold_ &&
                        dis2 < min_line_dis_threshold_ *
                                   min_line_dis_threshold_))) {
                    line_cloud.push_back(p);
                  }
                }
              }
              if (line_cloud.size() > 10) {
                line_cloud_list.push_back(line_cloud);
              }
            }
          }
        }
      }
    }
  }
}

// void                                                                    //适配poly_fisheye模型
// Calibration::spaceToPlane(std::vector<cv::Point3f> &P3list, std::vector<cv::Point2f> &P2list) 
// {
//      Eigen::Matrix4d T_c_l= Eigen::Matrix4d::Identity();
//      T_c_l.block<3,3>(0,0)= init_rotation_matrix_;
//      T_c_l.block<3,1>(0,3)= init_translation_vector_;
//      Eigen::Vector3d P;
//      Eigen::Vector2d p; 
//   cout<<"start spacetoplane "<<endl;
//   std::vector<float>paramlist={fk2_,fk3_,fk4_,fk5_,fk6_,fk7_,fp1_,fp2_,A11_,A12_,A22_,fu0_,fv0_};

//     for (auto P : P3list)
//     {
//     Eigen::Vector3d Pt3(P.x,P.y,P.z);
//     Eigen::Vector4d Pt4(P.x,P.y,P.z,1);
//     Eigen::Vector4d Pt4_T=T_c_l*Pt4;                    //先用初始外参获得变换后的点

//     double theta = acos( Pt4_T( 2 ) / Pt4_T.norm( ) );
//     double inverse_r_P2 = 1.0 / sqrt( Pt4_T( 1 ) * Pt4_T( 1 ) + Pt4_T( 0 ) * Pt4_T( 0 ) );
//     double sin_phi      = Pt4_T( 1 ) * inverse_r_P2;
//     double cos_phi      = Pt4_T( 0 ) * inverse_r_P2;
//     Eigen::Vector2d p_u;
        
//       double u = Pt4_T( 0 ) / Pt4_T( 2 );
//       double v = Pt4_T( 1 ) / Pt4_T( 2 );

//       double r_point = theta
//         + fk2_ * theta * theta
//         + fk3_ * theta * theta * theta
//         + fk4_ * theta * theta * theta * theta
//         + fk5_ * theta * theta * theta * theta * theta
//         + fk6_ * theta * theta * theta * theta * theta * theta
//         + fk7_ * theta * theta * theta * theta * theta * theta * theta; // clang-format on
 
//             p_u = r_point * Eigen::Vector2d( cos_phi, sin_phi ) ;
            
        

//         // Apply generalised projection matrix
//         p << A11_*p_u[0] + A12_*p_u[1] + fu0_, A22_ * p_u[1] + fv0_;
//     // p(0) = A11_ * p_u(0) + A12_ * p_u(1) + fu0_;
//     // p(1) = A22_ * p_u(1) + fv0_;
  

//     cv::Point2d p;
//     p.x=A11_*p_u[0] + A12_*p_u[1] + fu0_;
//     p.y=A22_ * p_u[1] + fv0_;
//     P2list.push_back(p);   
//      }
// }

// 构建视觉-激光雷达的点对点匹配问题，用于后续的优化
void Calibration::buildVPnp(
    const Vector6d &extrinsic_params, const int dis_threshold,
    const bool show_residual,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d,
    std::vector<VPnPData> &pnp_list) {
  pnp_list.clear();
  //创建一个二维向量，用于储存图像中每个像素对应的点云
  std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
  for (int y = 0; y < height_; y++) {
    std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
    for (int x = 0; x < width_; x++) {
      std::vector<pcl::PointXYZI> col_pts_container;
      row_pts_container.push_back(col_pts_container);
    }
    img_pts_container.push_back(row_pts_container);
  }
  // 将lidar三维点转换为二维图像点
  std::vector<cv::Point3d> pts_3d;
  Eigen::AngleAxisd rotation_vector3;
  rotation_vector3 =
      Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
//  3D点云pts_3d
  for (size_t i = 0; i < lidar_line_cloud_3d->size(); i++) {
    pcl::PointXYZI point_3d = lidar_line_cloud_3d->points[i];
    pts_3d.emplace_back(cv::Point3d(point_3d.x, point_3d.y, point_3d.z));
  }
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << fx_, 0.0, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  cv::Mat distortion_coeff =
      (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
  cv::Mat r_vec =
      (cv::Mat_<double>(3, 1)
           << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
  cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                   extrinsic_params[4], extrinsic_params[5]);
  // project 3d-points into image view
  std::vector<cv::Point2d> pts_2d;
  // debug
  // std::cout << "camera_matrix:" << camera_matrix << std::endl;
  // std::cout << "distortion_coeff:" << distortion_coeff << std::endl;
  // std::cout << "r_vec:" << r_vec << std::endl;
  // std::cout << "t_vec:" << t_vec << std::endl;
  // std::cout << "pts 3d size:" << pts_3d.size() << std::endl;
  //  cv::Mat t_vec = (cv::Mat_<double>(3, 1) << 0,0,0);
  // pts_2d 输出
  cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,pts_2d);

  //spaceToPlane(pts_3d,pts_2d);
   //创建一个新的2d点云，存储投影到图像平面的lidar边缘点
  pcl::PointCloud<pcl::PointXYZ>::Ptr line_edge_cloud_2d(
      new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<int> line_edge_cloud_2d_number;   //创建一个向量，储存每个点的编号
  for (size_t i = 0; i < pts_2d.size(); i++) {   //遍历所有2d点
    pcl::PointXYZ p;   //创建2d点储存坐标
    p.x = pts_2d[i].x;
    p.y = -pts_2d[i].y;
    p.z = 0;
    pcl::PointXYZI pi_3d;   //创建一个新的点，储存3D坐标和intensity
    pi_3d.x = pts_3d[i].x;
    pi_3d.y = pts_3d[i].y;
    pi_3d.z = pts_3d[i].z;
    pi_3d.intensity = 1;
    if (p.x > 0 && p.x < width_ && pts_2d[i].y > 0 && pts_2d[i].y < height_) {
      if (img_pts_container[pts_2d[i].y][pts_2d[i].x].size() == 0) { // 如果这个像素还没有对应的点，那么就添加这个点到2D点云和编号向量中，并将3D点添加到图像点容器中
        line_edge_cloud_2d->points.push_back(p);
        line_edge_cloud_2d_number.push_back(plane_line_number_[i]);
        img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
      } else {                                                         // 如果这个像素已经有对应的点，那么只需要将3D点添加到图像点容器中
        img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
      }
    }
  }
  if (show_residual) {
    cv::Mat residual_img =
        getConnectImg(dis_threshold, cam_edge_cloud_2d, line_edge_cloud_2d);
    static int nn=-1;
    nn++;
    cv::namedWindow("residual",0);                     //修改，创建窗口
    cv::resizeWindow("residual", 1345, 1014);    
    cv::imshow("residual", residual_img);
    cv::imwrite("/home/yty/zhitai/yuancalib_ws/src/MFCalib/result/residualimage/residual"+std::to_string(nn)+".png", residual_img);
    cv::waitKey(100);
  }
  //创建KDtree，用于在2d图像寻找最近邻
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_lidar(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud_lidar =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  kdtree->setInputCloud(cam_edge_cloud_2d);
  kdtree_lidar->setInputCloud(line_edge_cloud_2d);
  tree_cloud = cam_edge_cloud_2d;
  tree_cloud_lidar = line_edge_cloud_2d;
  search_cloud = line_edge_cloud_2d;
  // 指定近邻个数
  int K = 5;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  std::vector<int> pointIdxNKNSearchLidar(K);
  std::vector<float> pointNKNSquaredDistanceLidar(K);
  int match_count = 0;
  double mean_distance;
  int line_count = 0;
  std::vector<cv::Point2d> lidar_2d_list;
  std::vector<cv::Point2d> img_2d_list;
  std::vector<Eigen::Vector2d> camera_direction_list;
  std::vector<Eigen::Vector2d> lidar_direction_list;
  std::vector<int> lidar_2d_number;
  //遍历激光雷达的2D点云，并对每个点进行最近邻搜索
  for (size_t i = 0; i < search_cloud->points.size(); i++) {
    pcl::PointXYZ searchPoint = search_cloud->points[i];
    if ((kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0) &&
        (kdtree_lidar->nearestKSearch(searchPoint, K, pointIdxNKNSearchLidar,
                                      pointNKNSquaredDistanceLidar) > 0)) {
      bool dis_check = true;
      for (int j = 0; j < K; j++) {
        float distance = sqrt(
            pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
            pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
        if (distance > dis_threshold) {
          dis_check = false;
        }
      }
       // 如果一个点在图像中的最近邻都在一定的距离阈值内，那么这个点就被认为是一个匹配点，并被添加到匹配列表中
     // 如果距离检查通过
      if (dis_check) {
        // 创建一个新的点，用于存储激光雷达点的2D坐标
        cv::Point p_l_2d(search_cloud->points[i].x, -search_cloud->points[i].y);
        // 创建一个新的点，用于存储相机点的2D坐标
        cv::Point p_c_2d(tree_cloud->points[pointIdxNKNSearch[0]].x,
                         -tree_cloud->points[pointIdxNKNSearch[0]].y);
        // 创建一个新的向量，用于存储相机点的方向
        Eigen::Vector2d direction_cam(0, 0);
        // 创建一个新的向量，用于存储相机点的坐标
        std::vector<Eigen::Vector2d> points_cam;
        // 遍历所有的相机点
        for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
          // 创建一个新的点，用于存储相机点的坐标
          Eigen::Vector2d p(tree_cloud->points[pointIdxNKNSearch[i]].x,
                            -tree_cloud->points[pointIdxNKNSearch[i]].y);
          // 将点添加到向量中
          points_cam.push_back(p);
        }
        // 计算相机点的方向
        calcDirection(points_cam, direction_cam);
        // 创建一个新的向量，用于存储激光雷达点的方向
        Eigen::Vector2d direction_lidar(0, 0);
        // 创建一个新的向量，用于存储激光雷达点的坐标
        std::vector<Eigen::Vector2d> points_lidar;
        // 遍历所有的激光雷达点
        for (size_t i = 0; i < pointIdxNKNSearch.size(); i++) {
          // 创建一个新的点，用于存储激光雷达点的坐标
          Eigen::Vector2d p(
              tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].x,
              -tree_cloud_lidar->points[pointIdxNKNSearchLidar[i]].y);
          // 将点添加到向量中
          points_lidar.push_back(p);
        }
        // 计算激光雷达点的方向
        calcDirection(points_lidar, direction_lidar);
        // 检查激光雷达点是否在视野中
        if (checkFov(p_l_2d)) {
          // 如果在视野中，那么就将激光雷达点的2D坐标、相机点的2D坐标、相机点的方向、激光雷达点的方向和激光雷达点的编号添加到对应的列表中
          lidar_2d_list.push_back(p_l_2d);
          img_2d_list.push_back(p_c_2d);
          camera_direction_list.push_back(direction_cam);
          lidar_direction_list.push_back(direction_lidar);
          lidar_2d_number.push_back(line_edge_cloud_2d_number[i]);
        }
      }
    }
  }

  // 遍历所有的匹配点，并计算每个匹配点的3D坐标和2D坐标，以及相机和激光雷达的方向
  for (size_t i = 0; i < lidar_2d_list.size(); i++) {
    int y = lidar_2d_list[i].y;
    int x = lidar_2d_list[i].x;
    int pixel_points_size = img_pts_container[y][x].size();
    if (pixel_points_size > 0) {
      VPnPData pnp;
      pnp.x = 0;
      pnp.y = 0;
      pnp.z = 0;
      pnp.u = img_2d_list[i].x;
      pnp.v = img_2d_list[i].y;
      for (size_t j = 0; j < pixel_points_size; j++) {
        pnp.x += img_pts_container[y][x][j].x;
        pnp.y += img_pts_container[y][x][j].y;
        pnp.z += img_pts_container[y][x][j].z;
      }
      pnp.x = pnp.x / pixel_points_size;
      pnp.y = pnp.y / pixel_points_size;
      pnp.z = pnp.z / pixel_points_size;
      pnp.direction = camera_direction_list[i];
      pnp.direction_lidar = lidar_direction_list[i];
      pnp.number = lidar_2d_number[i];
      float theta = pnp.direction.dot(pnp.direction_lidar);
      if (theta > direction_theta_min_ || theta < direction_theta_max_) {
        pnp_list.push_back(pnp);
      }
    }
  }
}

void Calibration::buildPnp(
    const Vector6d &extrinsic_params, const int dis_threshold,
    const bool show_residual,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cam_edge_cloud_2d,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &lidar_line_cloud_3d,
    std::vector<PnPData> &pnp_list) {
  std::vector<std::vector<std::vector<pcl::PointXYZI>>> img_pts_container;
  for (int y = 0; y < height_; y++) {
    std::vector<std::vector<pcl::PointXYZI>> row_pts_container;
    for (int x = 0; x < width_; x++) {
      std::vector<pcl::PointXYZI> col_pts_container;
      row_pts_container.push_back(col_pts_container);
    }
    img_pts_container.push_back(row_pts_container);
  }
  std::vector<cv::Point3f> pts_3d;
  Eigen::AngleAxisd rotation_vector3;
  rotation_vector3 =
      Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
  for (size_t i = 0; i < lidar_line_cloud_3d->size(); i++) {
    pcl::PointXYZI point_3d = lidar_line_cloud_3d->points[i];
    pts_3d.emplace_back(cv::Point3f(point_3d.x, point_3d.y, point_3d.z));
  }
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << fx_, s_, cx_, 0.0, fy_, cy_, 0.0, 0.0, 1.0);
  cv::Mat distortion_coeff =
      (cv::Mat_<double>(1, 5) << k1_, k2_, p1_, p2_, k3_);
  cv::Mat r_vec =
      (cv::Mat_<double>(3, 1)
           << rotation_vector3.angle() * rotation_vector3.axis().transpose()[0],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[1],
       rotation_vector3.angle() * rotation_vector3.axis().transpose()[2]);
  cv::Mat t_vec = (cv::Mat_<double>(3, 1) << extrinsic_params[3],
                   extrinsic_params[4], extrinsic_params[5]);
  // project 3d-points into image view
  std::vector<cv::Point2f> pts_2d;
  cv::projectPoints(pts_3d, r_vec, t_vec, camera_matrix, distortion_coeff,
                     pts_2d);
  //spaceToPlane(pts_3d,pts_2d);
  pcl::PointCloud<pcl::PointXYZ>::Ptr line_edge_cloud_2d(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < pts_2d.size(); i++) {
    pcl::PointXYZ p;
    p.x = pts_2d[i].x;
    p.y = -pts_2d[i].y;
    p.z = 0;
    pcl::PointXYZI pi_3d;
    pi_3d.x = pts_3d[i].x;
    pi_3d.y = pts_3d[i].y;
    pi_3d.z = pts_3d[i].z;
    pi_3d.intensity = 1;
    if (p.x > 0 && p.x < width_ && pts_2d[i].y > 0 && pts_2d[i].y < height_) {
      line_edge_cloud_2d->points.push_back(p);
      img_pts_container[pts_2d[i].y][pts_2d[i].x].push_back(pi_3d);
    }
  }
  if (show_residual) {
    cv::Mat residual_img =
        getConnectImg(dis_threshold, cam_edge_cloud_2d, line_edge_cloud_2d);
    cv::imshow("residual", residual_img);
    cv::waitKey(100);
  }
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr search_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud =
      pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  kdtree->setInputCloud(cam_edge_cloud_2d);
  tree_cloud = cam_edge_cloud_2d;
  search_cloud = line_edge_cloud_2d;
  // 指定近邻个数
  int K = 1;
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  int match_count = 0;
  double mean_distance;
  int line_count = 0;
  std::vector<cv::Point2d> lidar_2d_list;
  std::vector<cv::Point2d> img_2d_list;
  for (size_t i = 0; i < search_cloud->points.size(); i++) {
    pcl::PointXYZ searchPoint = search_cloud->points[i];
    if (kdtree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                               pointNKNSquaredDistance) > 0) {
      for (int j = 0; j < K; j++) {
        float distance = sqrt(
            pow(searchPoint.x - tree_cloud->points[pointIdxNKNSearch[j]].x, 2) +
            pow(searchPoint.y - tree_cloud->points[pointIdxNKNSearch[j]].y, 2));
        if (distance < dis_threshold) {

          cv::Point p_l_2d(search_cloud->points[i].x,
                           -search_cloud->points[i].y);
          cv::Point p_c_2d(tree_cloud->points[pointIdxNKNSearch[j]].x,
                           -tree_cloud->points[pointIdxNKNSearch[j]].y);
          if (checkFov(p_l_2d)) {
            lidar_2d_list.push_back(p_l_2d);
            img_2d_list.push_back(p_c_2d);
          }
        }
      }
    }
  }
  pnp_list.clear();
  for (size_t i = 0; i < lidar_2d_list.size(); i++) {
    int y = lidar_2d_list[i].y;
    int x = lidar_2d_list[i].x;
    int pixel_points_size = img_pts_container[y][x].size();
    if (pixel_points_size > 0) {
      PnPData pnp;
      pnp.x = 0;
      pnp.y = 0;
      pnp.z = 0;
      pnp.u = img_2d_list[i].x;
      pnp.v = img_2d_list[i].y;
      for (size_t j = 0; j < pixel_points_size; j++) {
        pnp.x += img_pts_container[y][x][j].x;
        pnp.y += img_pts_container[y][x][j].y;
        pnp.z += img_pts_container[y][x][j].z;
      }
      pnp.x = pnp.x / pixel_points_size;
      pnp.y = pnp.y / pixel_points_size;
      pnp.z = pnp.z / pixel_points_size;
      pnp_list.push_back(pnp);
    }
  }
}

void Calibration::loadImgAndPointcloud(
    const std::string path, pcl::PointCloud<pcl::PointXYZI>::Ptr &origin_cloud,
    cv::Mat &rgb_img) {
  origin_cloud =
      pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::fstream file_;
  file_.open(path, ios::in);
  if (!file_) {
    cout << "File " << path << " does not exit" << endl;
    return;
  }
  ROS_INFO("Loading the rosbag %s", path.c_str());
  rosbag::Bag bag;
  try {
    bag.open(path, rosbag::bagmode::Read);
  } catch (rosbag::BagException e) {
    ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
    return;
  }

  std::vector<string> lidar_topic;
  lidar_topic.push_back(lidar_topic_name_);
  rosbag::View view(bag, rosbag::TopicQuery(lidar_topic));

  int cloudCount = 0;
  for (const rosbag::MessageInstance &m : view) {
    if (is_use_custom_msg_) {
      livox_ros_driver::CustomMsg livox_cloud_msg =
          *(m.instantiate<livox_ros_driver::CustomMsg>()); // message type

      for (uint i = 0; i < livox_cloud_msg.point_num; ++i) {
        pcl::PointXYZI p;
        p.x = livox_cloud_msg.points[i].x;
        p.y = livox_cloud_msg.points[i].y;
        p.z = livox_cloud_msg.points[i].z;
        p.intensity = livox_cloud_msg.points[i].reflectivity;
        origin_cloud->points.push_back(p);
      }
    } else {
      sensor_msgs::PointCloud2 livox_cloud;
      livox_cloud =
          *(m.instantiate<sensor_msgs::PointCloud2>()); // message type
      pcl::PointCloud<pcl::PointXYZI> cloud;
      pcl::PCLPointCloud2 pcl_pc;
      pcl_conversions::toPCL(livox_cloud, pcl_pc);
      pcl::fromPCLPointCloud2(pcl_pc, cloud);
      for (uint i = 0; i < cloud.size(); ++i) {
        origin_cloud->points.push_back(cloud.points[i]);
      }
    }

    ++cloudCount;
    // maxinum msg num 1000
    // if (cloudCount > 1000) {
    //   break;
    // }
  }
  std::vector<string> img_topic;
  img_topic.push_back(image_topic_name_);
  rosbag::View img_view(bag, rosbag::TopicQuery(img_topic));
  int cnt = 0;
  for (const rosbag::MessageInstance &m : img_view) {
    cnt++;
    if (cnt == 1) {
      sensor_msgs::Image image;
      image = *(m.instantiate<sensor_msgs::Image>()); // message type
      cv_bridge::CvImagePtr img_ptr =
          cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BGR8);
      img_ptr->image.copyTo(rgb_img);
    }
  }
  ROS_INFO("Sucessfully load Point Cloud and Image");
}

void Calibration::calcDirection(const std::vector<Eigen::Vector2d> &points,
                                Eigen::Vector2d &direction) {
  Eigen::Vector2d mean_point(0, 0);
  for (size_t i = 0; i < points.size(); i++) {
    mean_point(0) += points[i](0);
    mean_point(1) += points[i](1);
  }
  mean_point(0) = mean_point(0) / points.size();
  mean_point(1) = mean_point(1) / points.size();
  Eigen::Matrix2d S;
  S << 0, 0, 0, 0;
  for (size_t i = 0; i < points.size(); i++) {
    Eigen::Matrix2d s =
        (points[i] - mean_point) * (points[i] - mean_point).transpose();
    S += s;
  }
  Eigen::EigenSolver<Eigen::Matrix<double, 2, 2>> es(S);
  Eigen::MatrixXcd evecs = es.eigenvectors();
  Eigen::MatrixXcd evals = es.eigenvalues();
  Eigen::MatrixXd evalsReal;
  evalsReal = evals.real();
  Eigen::MatrixXf::Index evalsMax;
  evalsReal.rowwise().sum().maxCoeff(&evalsMax); //得到最大特征值的位置
  direction << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax);
}

cv::Mat Calibration::getProjectionImg(const Vector6d &extrinsic_params) {
  cv::Mat depth_projection_img;
  projection(extrinsic_params, raw_lidar_cloud_, INTENSITY, false,
             depth_projection_img);
  cv::Mat map_img = cv::Mat::zeros(height_, width_, CV_8UC3);
  for (int x = 0; x < map_img.cols; x++) {
    for (int y = 0; y < map_img.rows; y++) {
      uint8_t r, g, b;
      float norm = depth_projection_img.at<uchar>(y, x) / 256.0;
      mapJet(norm, 0, 1, r, g, b);
      map_img.at<cv::Vec3b>(y, x)[0] = b;
      map_img.at<cv::Vec3b>(y, x)[1] = g;
      map_img.at<cv::Vec3b>(y, x)[2] = r;
    }
  }
  cv::Mat merge_img;
  if (image_.type() == CV_8UC3) {
    merge_img = 0.5 * map_img + 0.8 * image_;
  } else {
    cv::Mat src_rgb;
    cv::cvtColor(image_, src_rgb, cv::COLOR_GRAY2BGR);
    merge_img = 0.5 * map_img + 0.8 * src_rgb;
  }
  return merge_img;
}

void Calibration::calcResidual(const Vector6d &extrinsic_params,
                               const std::vector<VPnPData> vpnp_list,
                               std::vector<float> &residual_list) {
  residual_list.clear();
  Eigen::Vector3d euler_angle(extrinsic_params[0], extrinsic_params[1],
                              extrinsic_params[2]);
  Eigen::Matrix3d rotation_matrix;
  rotation_matrix =
      Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
  Eigen::Vector3d transation(extrinsic_params[3], extrinsic_params[4],
                             extrinsic_params[5]);
  for (size_t i = 0; i < vpnp_list.size(); i++) {
    Eigen::Vector2d residual;
    Eigen::Matrix2f var;
    calcCovarance(extrinsic_params, vpnp_list[i], 1, 0.02, 0.05, var);
    VPnPData vpnp_point = vpnp_list[i];
    float fx = fx_;
    float cx = cx_;
    float fy = fy_;
    float cy = cy_;
    Eigen::Matrix3d inner;
    inner << fx, 0, cx, 0, fy, cy, 0, 0, 1;
    Eigen::Vector4d distor;
    distor << k1_, k2_, p1_, p2_;
    Eigen::Vector3d p_l(vpnp_point.x, vpnp_point.y, vpnp_point.z);
    Eigen::Vector3d p_c = rotation_matrix * p_l + transation;
    Eigen::Vector3d p_2 = inner * p_c;
    float uo = p_2[0] / p_2[2];
    float vo = p_2[1] / p_2[2];
    float xo = (uo - cx) / fx;
    float yo = (vo - cy) / fy;
    float r2 = xo * xo + yo * yo;
    float r4 = r2 * r2;
    float distortion = 1.0 + distor[0] * r2 + distor[1] * r4;
    float xd = xo * distortion + (distor[2] * xo * yo + distor[2] * xo * yo) +
               distor[3] * (r2 + xo * xo + xo * xo);
    float yd = yo * distortion + distor[2] * xo * yo + distor[2] * xo * yo +
               distor[2] * (r2 + yo * yo + yo * yo);
    float ud = fx * xd + cx;
    float vd = fy * yd + cy;
    residual[0] = ud - vpnp_point.u;
    residual[1] = vd - vpnp_point.v;
    // if (vpnp_point.direction(0) == 0 && vpnp_point.direction(1) == 0) {
    //   residual[0] = ud - vpnp_point.u;
    //   residual[1] = vd - vpnp_point.v;
    // } else {
    //   residual[0] = ud - vpnp_point.u;
    //   residual[1] = vd - vpnp_point.v;
    //   Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
    //   Eigen::Vector2d n = vpnp_point.direction;
    //   Eigen::Matrix2d V = n * n.transpose();
    //   Eigen::Matrix2d R;
    //   R << residual[0], 0, 0, residual[1];
    //   V = I - V;
    //   R = V * R * V.transpose();
    //   residual[0] = R(0, 0);
    //   residual[1] = R(1, 1);
    // }
    // Eigen::Vector2d v(-vpnp_point.direction(1), vpnp_point.direction(0));
    // if (v(0) < 0) {
    //   v = -v;
    // }
    float cost = sqrt(residual[0] * residual[0] + residual[1] * residual[1]);
    residual_list.push_back(cost);
  }
}

void Calibration::calcCovarance(const Vector6d &extrinsic_params,
                                const VPnPData &vpnp_point,
                                const float pixel_inc, const float range_inc,
                                const float degree_inc,
                                Eigen::Matrix2f &covarance) {
  Eigen::Matrix3d rotation;
  rotation = Eigen::AngleAxisd(extrinsic_params[0], Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(extrinsic_params[1], Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(extrinsic_params[2], Eigen::Vector3d::UnitX());
  Eigen::Vector3d transation(extrinsic_params[3], extrinsic_params[4],
                             extrinsic_params[5]);
  float fx = fx_;
  float cx = cx_;
  float fy = fy_;
  float cy = cy_;
  Eigen::Vector3f p_l(vpnp_point.x, vpnp_point.y, vpnp_point.z);
  Eigen::Vector3f p_c = rotation.cast<float>() * p_l + transation.cast<float>();

  Eigen::Matrix2f var_camera_pixel;
  var_camera_pixel << pow(pixel_inc, 2), 0, 0, pow(pixel_inc, 2);

  Eigen::Matrix2f var_lidar_pixel;
  float range = sqrt(vpnp_point.x * vpnp_point.x + vpnp_point.y * vpnp_point.y +
                     vpnp_point.z * vpnp_point.z);
  Eigen::Vector3f direction(vpnp_point.x, vpnp_point.y, vpnp_point.z);
  direction.normalize();
  Eigen::Matrix3f direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0,
      -direction(0), -direction(1), direction(0), 0;
  float range_var = range_inc * range_inc;
  Eigen::Matrix2f direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
      pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3f base_vector1(1, 1,
                               -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3f base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<float, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<float, 3, 2> A = range * direction_hat * N;
  Eigen::Matrix3f lidar_position_var =
      direction * range_var * direction.transpose() +
      A * direction_var * A.transpose();
  Eigen::Matrix3f lidar_position_var_camera =
      rotation.cast<float>() * lidar_position_var *
      rotation.transpose().cast<float>();
  Eigen::Matrix2f lidar_pixel_var_2d;
  Eigen::Matrix<float, 2, 3> B;
  B << fx / p_c(2), 0, fx * p_c(0) / pow(p_c(2), 2), 0, fy / p_c(2),
      fy * p_c(1) / pow(p_c(2), 2);
  var_lidar_pixel = B * lidar_position_var * B.transpose();
  covarance = var_camera_pixel + var_lidar_pixel;
}

#endif