#include "kalman_filter/FusionEKF.h"
#include "kalman_filter/tools.h"
#include "Eigen/Dense"
#include <iostream>
#define EPS 0.0001 // A very small number

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  is_initialized_ = false;
  previous_timestamp_ = 0;
  // Initializing matrices
  R_laser_ = MatrixXd(2, 2);  // 激光雷达与毫米波的观测协方差矩阵R
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);  // 激光雷达、毫米波状态空间->观测空间的变换矩阵H
  H_laser_ << 1, 0, 0, 0,
			  0, 1, 0, 0;
  Hj_ = MatrixXd(3, 4);      // 毫米波的状态变换矩阵（雅克比）
  // There is no need to tune R for this project because it is given in the task.
  // Measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // Measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   * TODO: Initialization
   *  利用第一帧数据，初始化协方差矩阵P，状态量（px, py, vx, vy）ekf_.x_
   *  如果第一帧是毫米波数据，需要将毫米波数据，从极坐标系转换到笛卡尔系
   *  第二帧之后进来的数据跳过此流程
   ****************************************************************************/
  if (!is_initialized_) {
    ekf_.x_ = VectorXd(4);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // 利用第一帧毫米波数据初始化状态量
      float rho = measurement_pack.raw_measurements_[0]; // range
	  float phi = measurement_pack.raw_measurements_[1]; // bearing
	  float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
	  // Coordinates convertion from polar to cartesian
	  float x = rho * cos(phi); 
	  float y = rho * sin(phi);
	  float vx = rho_dot * cos(phi);
	  float vy = rho_dot * sin(phi);
	  ekf_.x_ << x, y, vx , vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // 利用第一帧激光数据初始化状态量，因为无法直接根据激光数据得到速度，设为0
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    // 状态量xy太小的话设为EPS
    if (fabs(ekf_.x_(0)) < EPS and fabs(ekf_.x_(1)) < EPS){
		ekf_.x_(0) = EPS;
		ekf_.x_(1) = EPS;
	}
	// 初始化协方差矩阵P,4*4，因为我们不知道速度，所以将vx,vy方差设置大一些
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
			   0, 1, 0, 0,
			   0, 0, 1000, 0,
			   0, 0, 0, 1000;
    // Print the initialization results
    cout << "EKF init: " << ekf_.x_ << endl;
    // 更新时间戳，后面需要计算delta_t
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }

  /**
   TODO: Prediction
     * 根据上一帧的状态预测当前帧的状态
     * 计算状态转换矩阵F->当前预测的状态量x
     * 当前预测的过程噪声Q
   */
  // 计算两帧的时间差delta_t
  float dt = (measurement_pack.timestamp_ - previous_timestamp_);
  dt /= 1000000.0; // convert micros to s
  previous_timestamp_ = measurement_pack.timestamp_;
  
  // 根据匀速运动模型，设置状态变换的F矩阵
  // px = px_1 + vx*dt
  // py = py_1 + vy*dt
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, dt, 0,
			0, 1, 0, dt,
			0, 0, 1, 0,
			0, 0, 0, 1;
  // 计算随机加速度噪声协方差，需要根据实际的运动模型进行估计
  float noise_ax = 9.0;
  float noise_ay = 9.0;
  // 根据上一时刻状态->预测当前状态,过程噪声协方差矩阵Q
  float dt_2 = dt * dt; //dt^2
  float dt_3 = dt_2 * dt; //dt^3
  float dt_4 = dt_3 * dt; //dt^4
  float dt_4_4 = dt_4 / 4; //dt^4/4
  float dt_3_2 = dt_3 / 2; //dt^3/2
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4_4 * noise_ax, 0, dt_3_2 * noise_ax, 0,
	         0, dt_4_4 * noise_ay, 0, dt_3_2 * noise_ay,
	         dt_3_2 * noise_ax, 0, dt_2 * noise_ax, 0,
 	         0, dt_3_2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict(); // 当前预测x_和P_

  /**
   TODO: Update
     * 根据当前不同类型的观测数据和上一帧预测的状态，来计算残差和卡尔曼增益.
     * 获取当前的状态量以及协方差矩阵.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // 当前你观测是毫米波数据的话，H(X)是从状态空间4->测量空间的转换3
	// H(X)是非线性的，观测空间->状态空间的变换之后，已经不一定服从高斯分布了
	// 所以对H(X)用泰勒一阶展开,因此H矩阵可以由雅克比矩阵J代替
	ekf_.H_ = tools.CalculateJacobian(ekf_.x_); // 测量变量3个，状态变量4个，雅克比3*4
	ekf_.R_ = R_radar_;
	ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // 当前观测是激光雷达数据的话，就是线性变换了
    // 按卡尔曼滤波模型更新测量
	ekf_.H_ = H_laser_;
	ekf_.R_ = R_laser_;
	ekf_.Update(measurement_pack.raw_measurements_);
  }
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
