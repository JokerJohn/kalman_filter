#include "kalman_filter/kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in; // object state
  P_ = P_in; // object covariance matrix
  F_ = F_in; // state transition matrix
  H_ = H_in; // measurement matrix
  R_ = R_in; // measurement covariance matrix
  Q_ = Q_in; // process covariance matrix
}

// The Kalman filter predict function. The same for linear and extended Kalman filter
void KalmanFilter::Predict() {
  x_ = F_ * x_; // There is no external motion, so, we do not have to add "+u"
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Kalman filter update step. Equations from the lectures
  VectorXd y = z - H_ * x_; // 残差：当前观测 - H(X)*根据上一帧预测的状态量 = 当前观测 - 预测得到的观测
  KF(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  // 预测的状态量->预测的观测
  double rho = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  double theta = atan(x_(1) / x_(0));
  double rho_dot = (x_(0) * x_(2) + x_(1) * x_(3)) / rho;
  VectorXd h = VectorXd(3); // h(x_)
  h << rho, theta, rho_dot;

  VectorXd y = z - h; // 观测空间： 计算状态量残差 = 观测 - 预测
  // Calculations are essentially the same to the Update function
  KF(y);
}

// Universal update Kalman Filter step. Equations from the lectures
void KalmanFilter::KF(const VectorXd &y) {
  // 观测空间
  // 根据观测的残差计算残差的协方差矩阵、卡尔曼增益
  MatrixXd Ht = H_.transpose(); // H-1
  MatrixXd S = H_ * P_ * Ht + R_; // 观测残差的协方差矩阵
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;  // 卡尔安增益， 预测的协方差/观测残差的协方差，即预测的权重占比

  // 更新当前系统状态
  x_ = x_ + (K * y);  // 当前状态 = 预测状态 + 权重 * 预测在残差权重的占比
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;   // 当前协方差 = （1-K）* 预测的协方差
}
