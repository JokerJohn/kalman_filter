#include "kalman_filter/fusion_ekf.h"
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
    * 系统状态量：px, py, vx, vy
    * 观测量：
    *   1.激光雷达：px, py ,可直接作为系统状态量使用, 无需转换(H矩阵)
    *   2.毫米波雷达：rho(距离), phi(角度), rho_dot(角度变化率), 需要先转换成系统状态量使用(非线性过程)
    *  两个步骤：
    *    1.预测：根据运动模型预测下一时刻的系统状态 (px py vx vy), x_k = F(x)*x_k-1+Q
    *           系统上一时刻状态->系统当前时刻状态, 不涉及状态空间与观测空间的转换, 即不考虑传感器类型
    *           这个过程是线性的, 其状态转换矩阵定义为F(x), 过程噪声定义为Q
    *    2.测量更新： 根据当前的观测数据, 去修正预测的结果
    *           这里首先需要将预测数据转换到测量空间下, 这个过程涉及到具体的传感器模型, 以及观测带来的传感器本身的噪声R
    *           z_k = H(x)*x_k + R  H矩阵则是描述这个状态空间到观测空间的转换
    *           这样计算残差 delta_x = z_k - x_k
    *
    *
  */
  is_initialized_ = false;
  previous_timestamp_ = 0;
  // Initializing matrices
  // 激光雷达与毫米波的观测协方差矩阵R
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  // 激光雷达、毫米波状态空间->观测空间的变换矩阵H
  H_laser_ = MatrixXd(2, 4);
  H_laser_ << 1, 0, 0, 0,
			  0, 1, 0, 0;
  // 毫米波的状态变换矩阵(雅克比), 因为是非线性
  Hj_ = MatrixXd(3, 4);

  // 激光雷达和毫米波的测量噪声R是设备固有的,在说明中已给出
  R_laser_ << 0.0225, 0,
              0, 0.0225;
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
   *  利用第一帧数据, 初始化系统状态变量 ekf_.x_,协方差矩阵P,状态量(px, py, vx, vy)
   *  这里传感器不同, 可以初始化的变量就不同, 比如激光雷达只能初始化px和py
   *  第二帧之后进来的数据则直接进入预测流程
   ****************************************************************************/
  if (!is_initialized_) {
    ekf_.x_ = VectorXd(4);
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // 利用第一帧毫米波数据初始化状态量
      float rho = measurement_pack.raw_measurements_[0]; // range
	  float phi = measurement_pack.raw_measurements_[1]; // bearing
	  float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
	  // 毫米波数据可同时初始化所有的系统状态变量
	  float x = rho * cos(phi); 
	  float y = rho * sin(phi);
	  float vx = rho_dot * cos(phi);
	  float vy = rho_dot * sin(phi);
	  ekf_.x_ << x, y, vx , vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // 激光只能初始化px和py
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    // 这里校验状态量, 状态量xy太小的话设为EPS
    if (fabs(ekf_.x_(0)) < EPS and fabs(ekf_.x_(1)) < EPS){
		ekf_.x_(0) = EPS;
		ekf_.x_(1) = EPS;
	}
	// 初始化系统4*4的协方差矩阵P,因为我们不知道速度,所以将vx,vy方差设置大一些
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
     * 根据上一帧的系统状态(px_1, py_1, vx_1, vy_1)预测当前帧的状态(px py vx vy)
     * 计算状态转换矩阵F->当前预测的状态量x
     * 预测的过程噪声Q, 一部分噪声是由匀速运动模型带来的
   */
  // 计算两帧的时间差dt
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

  ekf_.Predict(); // 输入转换矩阵F，过程噪声Q

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
