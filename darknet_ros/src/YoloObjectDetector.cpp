/*
 * YoloObjectDetector.cpp
 *
 *  Created on: Dec 19, 2016
 *      Author: Marko Bjelonic
 *   Institute: ETH Zurich, Robotic Systems Lab
 */

// yolo object detector
#include "darknet_ros/YoloObjectDetector.hpp"

// Check for xServer
#include <X11/Xlib.h>

#ifdef DARKNET_FILE_PATH
static std::string darknetFilePath_ = DARKNET_FILE_PATH;
#else
#error Path of darknet repository is not defined in CMakeLists.txt.
#endif

namespace darknet_ros {

static char* cfg;
static char* weights;
static char* data;
static char** detectionNames;

YoloObjectDetector::YoloObjectDetector(ros::NodeHandle nh)
    : nodeHandle_(nh), imageTransport_(nodeHandle_) {

  // Read parameters from config file.
  if (!readParameters()) {
    ROS_ERROR("Cloud not read all required parameters!");
    ros::requestShutdown();
  }

  init();

  ROS_INFO("Successfully started [YoloObjectDetector] Node.");
}


bool YoloObjectDetector::readParameters() {

  // Set vector sizes.
  return nodeHandle_.getParam("yolo_model/detection_classes/names", classLabels_) &&
      nodeHandle_.getParam("show_opencv", show_opencv_) &&
      nodeHandle_.getParam("publish_detection_image", publish_detection_image_);
}


void YoloObjectDetector::init() {
  ROS_INFO("[YoloObjectDetector] init().");

  numClasses_ = static_cast<int>(classLabels_.size());

  // Initialize deep network of darknet.
  std::string weightsPath;
  std::string configPath;
  std::string dataPath;
  std::string configModel;
  std::string weightsModel;

  // Threshold of object detection.
  float thresh;
  nodeHandle_.param("yolo_model/threshold/value", thresh, 0.3f);

  // Path to weights file.
  nodeHandle_.param("yolo_model/weight_file/name", weightsModel, std::string("yolov2-tiny.weights"));
  nodeHandle_.param("weights_path", weightsPath, std::string("/default"));
  weightsPath += "/" + weightsModel;
  weights = new char[weightsPath.length() + 1];
  strcpy(weights, weightsPath.c_str());

  // Path to config file.
  nodeHandle_.param("yolo_model/config_file/name", configModel, std::string("yolov2-tiny.cfg"));
  nodeHandle_.param("config_path", configPath, std::string("/default"));
  configPath += "/" + configModel;
  cfg = new char[configPath.length() + 1];
  strcpy(cfg, configPath.c_str());

  // Path to data folder.
  dataPath = darknetFilePath_;
  dataPath += "/data";
  data = new char[dataPath.length() + 1];
  strcpy(data, dataPath.c_str());

  // Get classes.
  detectionNames = (char**)realloc((void*)detectionNames, static_cast<size_t>(numClasses_ + 1) * sizeof(char*));
  for (size_t i = 0; i < static_cast<size_t>(numClasses_); i++) {
    detectionNames[i] = new char[classLabels_[i].length() + 1];
    strcpy(detectionNames[i], classLabels_[i].c_str());
  }

  // Load network.
  setupNetwork(cfg, weights, data, thresh, numClasses_, 1, 0.5);

  // Initialize publisher and subscriber.
  imageSubscriber_ = imageTransport_.subscribe("/camera/image_raw", 1, &YoloObjectDetector::cameraCallback, this);
  objectPublisher_ = nodeHandle_.advertise<darknet_ros_msgs::ObjectCount>("found_object", 1);
  boundingBoxesPublisher_ = nodeHandle_.advertise<darknet_ros_msgs::BoundingBoxes>("bounding_boxes", 1);
  detectionImagePublisher_ = nodeHandle_.advertise<sensor_msgs::Image>("detection_image", 1);
  // Action servers.
  checkForObjectsActionServer_.reset(new CheckForObjectsActionServer(nodeHandle_, "check_for_objects", false));
  checkForObjectsActionServer_->registerGoalCallback(boost::bind(&YoloObjectDetector::checkForObjectsActionGoalCB, this));
  checkForObjectsActionServer_->start();
  // Service servers.
  checkForObjectsServiceServer_ = nodeHandle_.advertiseService("check_for_objects", &YoloObjectDetector::checkForObjectsServiceCB, this);
}


void YoloObjectDetector::cameraCallback(const sensor_msgs::ImageConstPtr& msg) {
  ROS_DEBUG("[YoloObjectDetector] USB image received.");

  cv_bridge::CvImagePtr cam_image;
  try {
    cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  frameWidth_ = cam_image->image.size().width;
  frameHeight_ = cam_image->image.size().height;

  darknet_ros_msgs::BoundingBoxes bb_msg = processImage(cam_image);
  bb_msg.header.stamp = ros::Time::now();
  bb_msg.header.frame_id = "detection";
  bb_msg.image_header = msg->header;

  boundingBoxesPublisher_.publish(bb_msg);
}


void YoloObjectDetector::checkForObjectsActionGoalCB() {
  ROS_DEBUG("[YoloObjectDetector] Start check for objects action.");

  boost::shared_ptr<const darknet_ros_msgs::CheckForObjectsGoal> imageActionPtr = checkForObjectsActionServer_->acceptNewGoal();
  sensor_msgs::Image imageAction = imageActionPtr->image;

  cv_bridge::CvImagePtr cam_image;

  try {
    cam_image = cv_bridge::toCvCopy(imageAction, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  frameWidth_ = cam_image->image.size().width;
  frameHeight_ = cam_image->image.size().height;

  darknet_ros_msgs::BoundingBoxes bb_msg = processImage(cam_image);
  bb_msg.header.stamp = ros::Time::now();
  bb_msg.header.frame_id = "detection";
  bb_msg.image_header = imageAction.header;

  darknet_ros_msgs::CheckForObjectsResult objectsActionResult;
  objectsActionResult.id = imageActionPtr->id;
  objectsActionResult.bounding_boxes = bb_msg;
  checkForObjectsActionServer_->setSucceeded(objectsActionResult, "Send bounding boxes.");
}


bool YoloObjectDetector::checkForObjectsServiceCB(darknet_ros_msgs::CheckForObjects::Request &req,
                                                  darknet_ros_msgs::CheckForObjects::Response &res) {
  ROS_DEBUG("[YoloObjectDetector] Start check for objects service.");

  cv_bridge::CvImagePtr cam_image;
  try {
    cam_image = cv_bridge::toCvCopy(req.image, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return false;
  }

  frameWidth_ = cam_image->image.size().width;
  frameHeight_ = cam_image->image.size().height;

  darknet_ros_msgs::BoundingBoxes bb_msg = processImage(cam_image);
  bb_msg.header.stamp = ros::Time::now();
  bb_msg.header.frame_id = "detection";
  bb_msg.image_header = req.image.header;

  darknet_ros_msgs::CheckForObjects::Response serviceResponse;
  serviceResponse.bounding_boxes = bb_msg;
  serviceResponse.id = req.id;
  res = serviceResponse;

  return true;
}


darknet_ros_msgs::BoundingBoxes YoloObjectDetector::processImage(const cv_bridge::CvImagePtr& cam_image){

  cv::Mat camImageCopy = cam_image->image.clone();
  IplImage* ROS_img = new IplImage(camImageCopy);
  image img;
  img = ipl_to_image(ROS_img);
  image img_letter = letterbox_image(img, net_->w, net_->h);

  //detect
  layer l = net_->layers[net_->n - 1];
  float* X = img_letter.data;
  network_predict(net_, X);
  rememberNetwork(net_);

  detection* dets;
  int nboxes = 0;
  dets = avgPredictions(net_, &nboxes, img.w, img.h);
  do_nms_obj(dets, nboxes, l.classes, 0.4f);

  //convert into ROS message
  std::vector<RosBox_> roi_boxes = extractDetectionROIs(dets, nboxes);
  darknet_ros_msgs::BoundingBoxes bb_msg = fillROIsIntoROSmsg(roi_boxes);

  if(show_opencv_ || publish_detection_image_){
    cv::Mat visu_img = drawDetections(camImageCopy, bb_msg);

    if(show_opencv_) visualizeCVImage(visu_img, "Yolo_Detections");
    if(publish_detection_image_) publishDetectionImage(visu_img);
  }

  //cleanup
  free_image(img);
  free_detections(dets, nboxes);
  delete ROS_img;
  delete X;

  return bb_msg;
}


std::vector<RosBox_> YoloObjectDetector::extractDetectionROIs(const detection *dets, const int nboxes){

  int count = 0;
  std::vector<RosBox_> roi_boxes;

  for (int i = 0; i < nboxes; ++i) {
    float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.f;
    float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.f;
    float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.f;
    float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.f;


    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > 1) xmax = 1;
    if (ymax > 1) ymax = 1;

    // iterate through possible boxes and collect the bounding boxes
    for (int j = 0; j < demoClasses_; ++j) {
      if (dets[i].prob[j]>0) {
        float x_center = (xmin + xmax) / 2;
        float y_center = (ymin + ymax) / 2;
        float BoundingBox_width = xmax - xmin;
        float BoundingBox_height = ymax - ymin;

        // define bounding box
        // BoundingBox must be 1% size of frame (3.2x2.4 pixels)
        if (BoundingBox_width > 0.01f && BoundingBox_height > 0.01f) {
          RosBox_ roi_box;
          roi_box.x = x_center;
          roi_box.y = y_center;
          roi_box.w = BoundingBox_width;
          roi_box.h = BoundingBox_height;
          roi_box.Class = j;
          roi_box.prob = dets[i].prob[j];
          count++;
          roi_boxes.push_back(roi_box);
        }
      }
    }
  }

  // create array to store found bounding boxes
  // if no object detected, make sure that ROS knows that num = 0
  if (count == 0){
    RosBox_ roi_box;
    roi_box.num = 0;
    roi_boxes.push_back(roi_box);
  }
  else
    roi_boxes.begin()->num = count;

  return roi_boxes;
}


darknet_ros_msgs::BoundingBoxes YoloObjectDetector::fillROIsIntoROSmsg(const std::vector<RosBox_> &roi_boxes){

  //fill return msg
  std::vector<std::vector<RosBox_> > rosBoxes(static_cast<size_t>(numClasses_));
  std::vector<int> rosBoxCounter(static_cast<size_t>(numClasses_));
  darknet_ros_msgs::BoundingBoxes boundingBoxesResults;

  int num = roi_boxes.begin()->num;

  if (num < 0 || num >= 100){
    ROS_WARN_STREAM("Found " << num << " detections. Too many or to less!");
    return  boundingBoxesResults;
  }

  for (size_t i = 0; i < static_cast<size_t>(num); i++) {
    for (size_t j = 0; j < static_cast<size_t>(numClasses_); j++) {
      if (roi_boxes.at(i).Class == static_cast<int>(j)) {
        rosBoxes.at(j).push_back(roi_boxes.at(i));
        rosBoxCounter.at(j) = rosBoxCounter.at(j) + 1;
      }
    }
  }

  for (size_t i = 0; i < static_cast<size_t>(numClasses_); i++) {
    if (rosBoxCounter[i] > 0) {
      darknet_ros_msgs::BoundingBox boundingBox;

      for (size_t j = 0; j < static_cast<size_t>(rosBoxCounter[i]); j++) {
        int xmin = static_cast<int>((rosBoxes[i][j].x - rosBoxes[i][j].w / 2) * frameWidth_);
        int ymin = static_cast<int>((rosBoxes[i][j].y - rosBoxes[i][j].h / 2) * frameHeight_);
        int xmax = static_cast<int>((rosBoxes[i][j].x + rosBoxes[i][j].w / 2) * frameWidth_);
        int ymax = static_cast<int>((rosBoxes[i][j].y + rosBoxes[i][j].h / 2) * frameHeight_);

        boundingBox.Class = classLabels_[i];
        boundingBox.id = static_cast<int8_t>(i);
        boundingBox.probability = static_cast<double>(rosBoxes[i][j].prob);
        boundingBox.xmin = xmin;
        boundingBox.ymin = ymin;
        boundingBox.xmax = xmax;
        boundingBox.ymax = ymax;
        boundingBoxesResults.bounding_boxes.push_back(boundingBox);
      }
    }
  }

  return boundingBoxesResults;
}


void YoloObjectDetector::publishNumberOfDetections(const int num_det){

  //publish number of detections
  darknet_ros_msgs::ObjectCount msg;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "detection";
  msg.count = static_cast<signed char>(num_det);
  objectPublisher_.publish(msg);
}


void YoloObjectDetector::publishDetectionImage(const cv::Mat &cv_img){

  cv_bridge::CvImagePtr cv_ptr{new cv_bridge::CvImage};

  cv_ptr->header.frame_id = "detections";
  cv_ptr->header.stamp = ros::Time::now();
  cv_ptr->encoding = sensor_msgs::image_encodings::BGR8;
  cv_ptr->image = cv_img;

  sensor_msgs::Image::ConstPtr img_msg;
  try{
    img_msg = cv_ptr->toImageMsg();
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  detectionImagePublisher_.publish(img_msg);
}


cv::Mat YoloObjectDetector::drawDetections(const cv::Mat &img, const darknet_ros_msgs::BoundingBoxes &boxes){

  double resize_factor = static_cast<double>(500) / img.rows;
  int width = static_cast<int>(img.cols * resize_factor);
  int height  = static_cast<int>(img.rows * resize_factor);

  cv::Mat img_sized;
  img.copyTo(img_sized);
  cv::resize(img_sized,img_sized,cv::Size(width, height));

  for(const auto& box : boxes.bounding_boxes){

    int thickness = 2;
    double font_scale = 0.5;
    //bounding box
    int xmin = static_cast<int>(box.xmin) * width / img.cols;
    int xmax = static_cast<int>(box.xmax) * width / img.cols;
    int ymin = static_cast<int>(box.ymin) * height / img.rows;
    int ymax = static_cast<int>(box.xmax) * height / img.rows;



    cv::Point pt1 = cv::Point(xmax, ymax);
    cv::Point pt2 = cv::Point(xmin, ymin);
    cv::rectangle(img_sized, pt1, pt2, cv::Scalar(255,0,0), thickness);

    std::string class_name = box.Class;
    cv::Point pt_text = cv::Point(xmin, ymin-10);
    cv::putText(img_sized, class_name, pt_text, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255,0,0), thickness);
  }

  return img_sized;
}


void YoloObjectDetector::visualizeCVImage(const cv::Mat &img, const std::string &window_name){

  cv::namedWindow( window_name, cv::WINDOW_AUTOSIZE );
  cv::imshow( window_name, img );
  cv::waitKey(1);
}


int YoloObjectDetector::sizeNetwork(network* net) {

  int count = 0;
  for (int i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      count += l.outputs;
    }
  }

  return count;
}


void YoloObjectDetector::rememberNetwork(network* net) {

  int count = 0;
  for (int i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      memcpy(predictions_[demoIndex_] + count, net->layers[i].output, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }

}


detection* YoloObjectDetector::avgPredictions(network* net, int* nboxes, int img_width, int img_height) {

  int count = 0;
  fill_cpu(demoTotal_, 0, avg_, 1);
  for (int j = 0; j < demoFrame_; ++j) {
    axpy_cpu(demoTotal_, 1.f / demoFrame_, predictions_[j], 1, avg_, 1);
  }
  for (int i = 0; i < net->n; ++i) {
    layer l = net->layers[i];
    if (l.type == YOLO || l.type == REGION || l.type == DETECTION) {
      memcpy(l.output, avg_ + count, sizeof(float) * l.outputs);
      count += l.outputs;
    }
  }
  detection* dets = get_network_boxes(net, img_width, img_height, demoThresh_, demoHier_, 0, 1, nboxes);

  return dets;
}


void YoloObjectDetector::setupNetwork(char* cfgfile, char* weightfile, char* datafile,
                                        float thresh, int classes, int avg_frames, float hier) {

  demoFrame_ = avg_frames;
  load_alphabet_with_file(datafile);
  demoClasses_ = classes;
  demoThresh_ = thresh;
  demoHier_ = hier;
  printf("YOLO V3\n");
  net_ = load_network(cfgfile, weightfile, 0);
  set_batch_network(net_, 1);
}


void YoloObjectDetector::workerYolo() {

  ROS_INFO_STREAM("Entering Yolo Worker.");

  demoTotal_ = sizeNetwork(net_);
  predictions_ = (float**)calloc(static_cast<size_t>(demoFrame_), sizeof(float*));
  for (int i = 0; i < demoFrame_; ++i) {
    predictions_[i] = (float*)calloc(static_cast<size_t>(demoTotal_), sizeof(float));
  }
  avg_ = (float*)calloc(static_cast<size_t>(demoTotal_), sizeof(float));

  ros::spin();
}


} /* namespace darknet_ros*/
