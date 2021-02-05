/*
 * YoloObjectDetector.h
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

#pragma once

// c++
#include <pthread.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <condition_variable>

// ROS
#include <actionlib/server/simple_action_server.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>

// OpenCv
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

// darknet_ros_msgs
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <darknet_ros_msgs/CheckForObjectsAction.h>
#include <darknet_ros_msgs/CheckForObjects.h>
#include <darknet_ros_msgs/ObjectCount.h>

// Darknet.
#ifdef GPU
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "curand.h"
#endif

extern "C" {
#include <sys/time.h>
#include "box.h"
#include "cost_layer.h"
#include "darknet_ros/image_interface.h"
#include "detection_layer.h"
#include "network.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
}

extern "C" void ipl_into_image(IplImage* src, image im);
extern "C" image ipl_to_image(IplImage* src);
extern "C" void show_image_cv(image p, const char* name, IplImage* disp);

namespace darknet_ros {

//! Bounding box of the detected object.
typedef struct {
  float x, y, w, h, prob;
  int num, Class;
} RosBox_;

typedef struct {
  IplImage* image;
  std_msgs::Header header;
} IplImageWithHeader_;


class YoloObjectDetector {

 public:
  /*!
   * Constructor.
   */
  explicit YoloObjectDetector(ros::NodeHandle nh);


  /*!
   * \brief yolo
   */
  void workerYolo();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   * @return true if successful.
   */
  bool readParameters();

  /*!
   * Initialize the ROS connections.
   */
  void init();

  /*!
   * Callback of camera.
   * @param[in] msg image pointer.
   */
  void cameraCallback(const sensor_msgs::ImageConstPtr& msg);

  /*!
   * Check for objects action goal callback.
   */
  void checkForObjectsActionGoalCB();

  /*!
   * Check for objects service callback.
   */
  bool checkForObjectsServiceCB(darknet_ros_msgs::CheckForObjects::Request &req, darknet_ros_msgs::CheckForObjects::Response &res);

  /*!
   * \brief processes an image
   * \param cam_image
   * \return found bounding boxes
   */
  darknet_ros_msgs::BoundingBoxes processImage(const cv_bridge::CvImagePtr& cam_image);

  /*!
   * \brief extracts the ROI information of the detections
   * \param dets
   * \param nboxes
   * \return ROI boxes
   */
  std::vector<RosBox_> extractDetectionROIs(const detection *dets, const int nboxes);

  /*!
   * \brief fills the ROS information into a ROS msg
   * \param roi_boxes
   * \return ROS msg
   */
  darknet_ros_msgs::BoundingBoxes fillROIsIntoROSmsg(const std::vector<RosBox_> &roi_boxes);

  /*!
   * \brief drawDetections
   * \param img
   * \param boxes
   * \return img with bounding boxes
   */
  cv::Mat drawDetections(const cv::Mat &img, const darknet_ros_msgs::BoundingBoxes &boxes);

  /*!
   * \brief visualize a cv image in a with adjustable image size
   * \param img
   * \param window_name
   * \param resize_img_height
   */
  void visualizeCVImage(const cv::Mat &img, const std::string &window_name);

  /*!
   * \brief publish number of found objects in image
   * \param num_det
   */
  void publishNumberOfDetections(const int num_det);

  /*!
   * \brief publishDetectionImage
   * \param cv_img
   */
  void publishDetectionImage(const cv::Mat& cv_img);

  //! Yolo Functions
  detection* avgPredictions(network* net, int* nboxes, int img_width, int img_height);

  int sizeNetwork(network* net);

  void rememberNetwork(network* net);

  void setupNetwork(char* cfgfile, char* weightfile, char* datafile, float thresh, int classes, int avg_frames, float hier);


  //! Using.
  using CheckForObjectsActionServer = actionlib::SimpleActionServer<darknet_ros_msgs::CheckForObjectsAction>;
  using CheckForObjectsActionServerPtr = std::shared_ptr<CheckForObjectsActionServer>;

  //! ROS node handle.
  ros::NodeHandle nodeHandle_;

  //! Number of classes.
  int numClasses_;

  //! ROS Paremeter
  bool show_opencv_;
  bool publish_detection_image_;
  std::vector<std::string> classLabels_;

  //! Check for objects action server.
  CheckForObjectsActionServerPtr checkForObjectsActionServer_;

  //! Check for objects service server.
  ros::ServiceServer checkForObjectsServiceServer_;

  //! Advertise and subscribe to image topics.
  image_transport::ImageTransport imageTransport_;

  //! ROS subscriber and publisher.
  image_transport::Subscriber imageSubscriber_;
  ros::Publisher objectPublisher_;
  ros::Publisher boundingBoxesPublisher_;
  ros::Publisher detectionImagePublisher_;

  //! Camera related parameters.
  int frameWidth_;
  int frameHeight_;

  //! Darknet.
  int demoClasses_;
  network* net_;

  float demoThresh_ = 0;
  float demoHier_ = .5;

  int demoFrame_ = 3;
  float** predictions_;
  int demoIndex_ = 0;
  float* avg_;
  int demoTotal_ = 0;
};

} /* namespace darknet_ros*/
