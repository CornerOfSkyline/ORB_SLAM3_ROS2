#include "rgbd-inertial-node.hpp"

#include <opencv2/core/core.hpp>

using std::placeholders::_1;

RgbdInertialNode::RgbdInertialNode(ORB_SLAM3::System *SLAM) :
    Node("ORB_SLAM3_ROS2"),
    SLAM_(SLAM)
{
    subImu_ = this->create_subscription<ImuMsg>("imu", 1000, std::bind(&RgbdInertialNode::GrabImu, this, _1));
    subRgb_ = this->create_subscription<ImageMsg>("camera/rgb", 100, std::bind(&RgbdInertialNode::GrabRgb, this, _1));
    subDepth_ = this->create_subscription<ImageMsg>("camera/depth", 100, std::bind(&RgbdInertialNode::GrabDepth, this, _1));

    syncThread_ = new std::thread(&RgbdInertialNode::SyncWithImu, this);
}

RgbdInertialNode::~RgbdInertialNode()
{
    // Delete sync thread
    syncThread_->join();
    delete syncThread_;

    // Stop all threads
    SLAM_->Shutdown();

    // Save camera trajectory
    SLAM_->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
}

void RgbdInertialNode::GrabImu(const ImuMsg::SharedPtr msg)
{
    bufMutex_.lock();
    imuBuf_.push(msg);
    bufMutex_.unlock();
}

void RgbdInertialNode::GrabRgb(const ImageMsg::SharedPtr msg)
{
    bufMutexRgb_.lock();

    if (!rgbBuf_.empty())
        rgbBuf_.pop();
    rgbBuf_.push(msg);

    bufMutexRgb_.unlock();
}

void RgbdInertialNode::GrabDepth(const ImageMsg::SharedPtr msgRight)
{
    bufMutexDepth_.lock();

    if (!depthBuf_.empty())
        depthBuf_.pop();
    depthBuf_.push(msgRight);

    bufMutexDepth_.unlock();
}

cv::Mat RgbdInertialNode::GetImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }

    if (cv_ptr->image.type() == 0)
    {
        return cv_ptr->image.clone();
    }
    else
    {
        std::cerr << "Error image type" << std::endl;
        return cv_ptr->image.clone();
    }
}

cv::Mat RgbdInertialNode::GetDepthImage(const ImageMsg::SharedPtr msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;

    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);

    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    }

    return cv_ptr->image.clone();

}

void RgbdInertialNode::SyncWithImu()
{
    const double maxTimeDiff = 0.05;

    while (1)
    {
        cv::Mat rgb, depth;
        double tRgb = 0, tDepth = 0;
        if (!rgbBuf_.empty() && !depthBuf_.empty() && !imuBuf_.empty())
        {
            tRgb = Utility::StampToSec(rgbBuf_.front()->header.stamp);
            tDepth = Utility::StampToSec(depthBuf_.front()->header.stamp);

            bufMutexDepth_.lock();
            while ((tRgb - tDepth) > maxTimeDiff && depthBuf_.size() > 1)
            {
                depthBuf_.pop();
                tDepth = Utility::StampToSec(depthBuf_.front()->header.stamp);
            }
            bufMutexDepth_.unlock();

            bufMutexRgb_.lock();
            while ((tDepth - tRgb) > maxTimeDiff && rgbBuf_.size() > 1)
            {
                rgbBuf_.pop();
                tRgb = Utility::StampToSec(rgbBuf_.front()->header.stamp);
            }
            bufMutexRgb_.unlock();

            if ((tRgb - tDepth) > maxTimeDiff || (tDepth - tRgb) > maxTimeDiff)
            {
                // std::cout << "big time difference" << (tRgb - tDepth) << std::endl;
                // continue;
            }
            if (tRgb > Utility::StampToSec(imuBuf_.back()->header.stamp))
                continue;

            bufMutexRgb_.lock();
            rgb = GetImage(rgbBuf_.front());
            rgbBuf_.pop();
            bufMutexRgb_.unlock();

            bufMutexDepth_.lock();
            depth = GetDepthImage(depthBuf_.front());
            depthBuf_.pop();
            bufMutexDepth_.unlock();

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            bufMutex_.lock();
            if (!imuBuf_.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while (!imuBuf_.empty() && Utility::StampToSec(imuBuf_.front()->header.stamp) <= tRgb)
                {
                    double t = Utility::StampToSec(imuBuf_.front()->header.stamp);
                    cv::Point3f acc(imuBuf_.front()->linear_acceleration.x, imuBuf_.front()->linear_acceleration.y, imuBuf_.front()->linear_acceleration.z);
                    cv::Point3f gyr(imuBuf_.front()->angular_velocity.x, imuBuf_.front()->angular_velocity.y, imuBuf_.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    imuBuf_.pop();
                }
            }
            bufMutex_.unlock();

            SLAM_->TrackRGBD(rgb, depth, tRgb, vImuMeas);

            std::chrono::milliseconds tSleep(1);
            std::this_thread::sleep_for(tSleep);
        }
    }
}
