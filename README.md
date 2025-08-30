# MFCalib
**MFCalib** is an innovative extrinsic calibration tool for  LiDAR and camera that operates automatically in **targetless environments** with a single data capture. By leveraging a rich set of **edge information**, MFCalib significantly enhances calibration accuracy and robustness.

## Related paper
Related paper available on IEEE Xplore:  
[MFCalib: Single-shot and Automatic Extrinsic Calibration for LiDAR and Camera in Targetless Environments Based on Multi-Feature Edge](https://ieeexplore.ieee.org/document/10802395?denied=)

## 1. Prerequisites
### 1.1 **Ubuntu** and **ROS**
Ubuntu 64-bit 18.04 or 20.04.
ROS Melodic or Noetic. [ROS Installation](http://wiki.ros.org/ROS/Installation) and its additional ROS pacakge:

```
    sudo apt-get install ros-XXX-cv-bridge ros-xxx-pcl-conversions
```

### 1.2 **Eigen**
Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page). (Our code is tested with Eigen 1.10)

### 1.3 **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html).(Our code is tested with Ceres 1.10)

### 1.4 **PCL**
Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html). (Our code is tested with PCL 1.10)

## 2. Build
Clone the repository and catkin_make:

```
cd ~/catkin_ws/src
git clone https://github.com/Es1erda/MFCalib.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. Run our example
The example dataset can be download from [**BaiduNetDisk(百度网盘)**](https://pan.baidu.com/s/1b-6Eu5ChOirJ-DnPXoV5pA?pwd=g6eq)
### 3.1 Single scene calibration
Download example dataset to your local path, and then change the file path in **calib.yaml** to your data path. Then directly run
```
roslaunch MFCalib calib.launch
```
You will get the result. 
