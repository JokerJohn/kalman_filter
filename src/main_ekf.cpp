

#include "kalman_filter/fusion_ekf.h"
#include "kalman_filter/ground_truth_package.h"
#include "kalman_filter/measurement_package.h"

#include "Eigen/Dense"
#include "ros/ros.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

void check_arguments(int argc, char *argv[]) {
  string usage_instructions = "Usage instructions: ";
  usage_instructions += argv[0];
  usage_instructions += " path/to/input.txt output.txt";

  bool has_valid_args = false;

  // make sure the user has provided input and output files
  if (argc == 1) {
    cerr << usage_instructions << endl;
  } else if (argc == 2) {
    cerr << "Please include an output file.\n" << usage_instructions << endl;
  } else if (argc == 3) {
    has_valid_args = true;
  } else if (argc > 3) {
    cerr << "Too many arguments.\n" << usage_instructions << endl;
  }

  if (!has_valid_args) {
    exit(EXIT_FAILURE);
  }
}

void check_files(ifstream &in_file, string &in_name,
                 ofstream &out_file, string &out_name) {
  if (!in_file.is_open()) {
    cerr << "Cannot open input file: " << in_name << endl;
    exit(EXIT_FAILURE);
  }

  if (!out_file.is_open()) {
    cerr << "Cannot open output file: " << out_name << endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "kalmanfilter");
  ros::NodeHandle node_handle;

  // check参数
//  check_arguments(argc, argv);
//  string in_file_name_ = argv[1];
//  string out_file_name_ = argv[2];

  string in_file_name_ =
      "/home/xchu/workspace/hdmap_testws/src/kalman_filter/data/sample-laser-radar-measurement-data-1.txt";
  string out_file_name_ = "/home/xchu/workspace/hdmap_testws/src/kalman_filter/data/ekf_out.txt";

  ifstream in_file_(in_file_name_.c_str(), ifstream::in);
  ofstream out_file_(out_file_name_.c_str(), ofstream::out);

  // 校验文件路径
  check_files(in_file_, in_file_name_, out_file_, out_file_name_);

  // measurement_pack_list 中包含一个时间戳和两种类型的数据，按时间戳顺序排列。用来存储读取的数据文件
  vector<MeasurementPackage> measurement_pack_list;
  vector<GroundTruthPackage> gt_pack_list;

  // 逐行读取毫米波和激光雷达数据
  string line;
  while (getline(in_file_, line)) {
    string sensor_type;
    MeasurementPackage meas_package;
    GroundTruthPackage gt_package;
    istringstream iss(line);
    long long timestamp;

    // 读取激光雷达、毫米波、groundTruth数据
    iss >> sensor_type;  // 数据的第一行
    if (sensor_type.compare("L") == 0) {
      // LASER MEASUREMENT
      // 激光雷达数据, 只有px,py
      // read measurements at this timestamp
      meas_package.sensor_type_ = MeasurementPackage::LASER;
      meas_package.raw_measurements_ = VectorXd(2);
      float x;
      float y;
      iss >> x;
      iss >> y;
      meas_package.raw_measurements_ << x, y;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    } else if (sensor_type.compare("R") == 0) {
      // RADAR MEASUREMENT
      // 毫米波数据, 包含ro, phi, ro_dot
      meas_package.sensor_type_ = MeasurementPackage::RADAR;
      meas_package.raw_measurements_ = VectorXd(3);
      float ro;
      float phi;
      float ro_dot;
      iss >> ro;
      iss >> phi;
      iss >> ro_dot;
      meas_package.raw_measurements_ << ro, phi, ro_dot;
      iss >> timestamp;
      meas_package.timestamp_ = timestamp;
      measurement_pack_list.push_back(meas_package);
    }

    // ground truth data
    float x_gt;
    float y_gt;
    float vx_gt;
    float vy_gt;
    iss >> x_gt;
    iss >> y_gt;
    iss >> vx_gt;
    iss >> vy_gt;
    gt_package.gt_values_ = VectorXd(4);
    gt_package.gt_values_ << x_gt, y_gt, vx_gt, vy_gt;
    gt_pack_list.push_back(gt_package);
  }

  // 初始化EKF参数
  // EKF主要包括两个矩阵, 两个函数
  // 状态预测函数,需要计算状态变换的F矩阵；
  // 测量更新函数,需要计算测量变换矩阵H
  FusionEKF fusionEKF;

  vector<VectorXd> estimations;
  vector<VectorXd> ground_truth;

  //Call the EKF-based fusion
  size_t N = measurement_pack_list.size();
  for (size_t k = 0; k < N; ++k) {

    // 第一帧数据初始化预测的X,P和转换矩阵F
    // 从第二帧数据起开始进行融合 (the speed is unknown in the first frame)
    fusionEKF.ProcessMeasurement(measurement_pack_list[k]);

    // 将预测状态量结果输出到文件
    out_file_ << fusionEKF.ekf_.x_(0) << "\t";
    out_file_ << fusionEKF.ekf_.x_(1) << "\t";
    out_file_ << fusionEKF.ekf_.x_(2) << "\t";
    out_file_ << fusionEKF.ekf_.x_(3) << "\t";

    // 将观测结果输出到文件
    if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::LASER) {
      // output the estimation
      out_file_ << measurement_pack_list[k].raw_measurements_(0) << "\t";
      out_file_ << measurement_pack_list[k].raw_measurements_(1) << "\t";
    } else if (measurement_pack_list[k].sensor_type_ == MeasurementPackage::RADAR) {
      // output the estimation in the cartesian coordinates
      float ro = measurement_pack_list[k].raw_measurements_(0);
      float phi = measurement_pack_list[k].raw_measurements_(1);
      out_file_ << ro * cos(phi) << "\t"; // p1_meas
      out_file_ << ro * sin(phi) << "\t"; // ps_meas
    }

    // 输出ground truth
    out_file_ << gt_pack_list[k].gt_values_(0) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(1) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(2) << "\t";
    out_file_ << gt_pack_list[k].gt_values_(3) << "\n";

    estimations.push_back(fusionEKF.ekf_.x_);
    ground_truth.push_back(gt_pack_list[k].gt_values_);
  }

  // 计算RMSE
  Tools tools;
  cout << "Accuracy - RMSE:" << endl << tools.CalculateRMSE(estimations, ground_truth) << endl;

  // close files
  if (out_file_.is_open()) {
    out_file_.close();
  }
  if (in_file_.is_open()) {
    in_file_.close();
  }

  return 0;
}
