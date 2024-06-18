#ifndef __RGBD_INERTIAL_NODE_HPP__
#define __RGBD_INERTIAL_NODE_HPP__

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include <cv_bridge/cv_bridge.h>

#include "System.h"
#include "Frame.h"
#include "Map.h"
#include "Tracking.h"

#include "utility.hpp"

using ImuMsg = sensor_msgs::msg::Imu;
using ImageMsg = sensor_msgs::msg::Image;

class RgbdInertialNode : public rclcpp::Node
{
public:
    RgbdInertialNode(ORB_SLAM3::System* pSLAM);
    ~RgbdInertialNode();

private:
    void GrabImu(const ImuMsg::SharedPtr msg);
    void GrabRgb(const ImageMsg::SharedPtr msg);
    void GrabDepth(const ImageMsg::SharedPtr msg);
    cv::Mat GetImage(const ImageMsg::SharedPtr msg);
    cv::Mat GetDepthImage(const ImageMsg::SharedPtr msg);
    void SyncWithImu();

    rclcpp::Subscription<ImuMsg>::SharedPtr   subImu_;
    rclcpp::Subscription<ImageMsg>::SharedPtr subRgb_;
    rclcpp::Subscription<ImageMsg>::SharedPtr subDepth_;

    ORB_SLAM3::System *SLAM_;
    std::thread *syncThread_;

    // IMU
    queue<ImuMsg::SharedPtr> imuBuf_;
    std::mutex bufMutex_;

    // Image
    queue<ImageMsg::SharedPtr> rgbBuf_, depthBuf_;
    std::mutex bufMutexRgb_, bufMutexDepth_;
};

#endif
