#include "kalman_filter/fusion_ukf.h"
#include "kalman_filter/tools.h"
#include "Eigen/Dense"
#include <iostream>

#define EPS 0.001 // Just a small number

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // 是否融合这些观测数据
  use_laser_ = true;
  use_radar_ = true;

  // 系统的状态量和不确定度
  x_ = VectorXd(5);
  P_ = MatrixXd(5, 5);

  // 纵向加速度上的过程噪声标准差  m/s^2
  std_a_ = 1.5;
  //  yaw加速度上的过程噪声标准差 in rad/s^2
  std_yawdd_ = 0.57;
  // 激光雷达在px上测测量噪声标准差 m
  std_laspx_ = 0.15;
  // 激光雷达在py上测测量噪声标准差 m
  std_laspy_ = 0.15;
  // 毫米波测量的rho标准差 m
  std_radr_ = 0.3;
  // 毫米波测量的phi角度标准差
  std_radphi_ = 0.03;
  // 毫米波测量的角度变化率rho_phi in m/s
  std_radrd_ = 0.3;

  n_x_ = x_.size();   // 系统状态维度
  n_aug_ = n_x_ + 2; // 采样点的状态维度, 这里直接7个维度
  n_sig_ = 2 * n_aug_ + 1; // sigma点个数, 15个

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);  // 预测的sigma点集状态矩阵 5*15
  lambda_ = 3 - n_aug_;  // sigma点扩散参数
  weights_ = VectorXd(n_sig_); // sigma点权重

  // 初始化测量毫米波和激光的噪声矩阵R
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;

  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 *  Angle normalization to [-Pi, Pi]
 */
void UKF::NormAng(double *ang) {
  while (*ang > M_PI) *ang -= 2. * M_PI;
  while (*ang < -M_PI) *ang += 2. * M_PI;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  // CTRV Model, x_ is [px, py, vel, ang, ang_rate]
  if (!is_initialized_) {
    // 初始化系统协方差矩阵P
    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // 毫米波雷达数据的话
      float rho = measurement_pack.raw_measurements_[0]; // range
      float phi = measurement_pack.raw_measurements_[1]; // bearing
      float rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho
      // 求系统状态并作为初值
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);
      float v = sqrt(vx * vx + vy * vy);
      x_ << px, py, v, 0, 0; // 另外两个数据没有，设为0
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // 激光雷达数据，只有位置px,py, 其他初始化为0
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
      // Deal with the special case initialisation problems
      if (fabs(x_(0)) < EPS and fabs(x_(1)) < EPS) {
        x_(0) = EPS;
        x_(1) = EPS;
      }
    }

    // 初始化sigma点权重,一共15个sigma点
    weights_(0) = lambda_ / (lambda_ + n_aug_);  // 第一个点的权重lamda/(n+lamda)
    for (int i = 1; i < weights_.size(); i++) {  // 之后每个点的权重 1/2(n+lamda)
      weights_(i) = 0.5 / (n_aug_ + lambda_);
    }

    time_us_ = measurement_pack.timestamp_;  // 更新时间戳
    is_initialized_ = true;
    //cout << "Init" << endl;
    //cout << "x_" << x_ << endl;
    return;
  }

  // 计算delta_t并更新
  double dt = (measurement_pack.timestamp_ - time_us_);
  dt /= 1000000.0; // convert micros to s
  time_us_ = measurement_pack.timestamp_;

  // 根据上一帧状态，预测当前状态，计算转换后的sigma点，并拟合高斯
  Prediction(dt);
  //cout << "predict:" << endl;
  //cout << "x_" << endl << x_ << endl;

  // 根据传感器数据类型去获取融合结果
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    //cout << "Radar " << measurement_pack.raw_measurements_[0] << " " << measurement_pack.raw_measurements_[1] << endl;
    UpdateRadar(measurement_pack);
  }
  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    //cout << "Lidar " << measurement_pack.raw_measurements_[0] << " " << measurement_pack.raw_measurements_[1] << endl;
    UpdateLidar(measurement_pack);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // 匀加速运动模型，这里是dt^2
  double delta_t2 = delta_t * delta_t;

  // 扩展到7维的状态
  VectorXd x_aug = VectorXd(n_aug_);    // Augmented mean vector 7×1
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);    // Augmented state covarience matrix 7×7
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);    // Sigma point matrix 7*15

  // Fill the matrices
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;
  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;

  P_aug(5, 5) = std_a_ * std_a_;  // 剩下两个维度对加速度和yaw变化率噪声进行估计
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;
  MatrixXd L = P_aug.llt().matrixL();  // Square root of P matrix

  // 创建sigma点 7*15
  Xsig_aug.col(0) = x_aug;
  double sqrt_lambda_n_aug = sqrt(lambda_ + n_aug_); // Save some computations
  VectorXd sqrt_lambda_n_aug_L;
  for (int i = 0; i < n_aug_; i++) {
    // 每一行赋值
    sqrt_lambda_n_aug_L = sqrt_lambda_n_aug * L.col(i);
    Xsig_aug.col(i + 1) = x_aug + sqrt_lambda_n_aug_L;
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt_lambda_n_aug_L;
  }

  // 预测sigma点的状态，根据匀加速运动模型来计算
  for (int i = 0; i < n_sig_; i++) {
    // Extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);
    // Precalculate sin and cos for optimization
    double sin_yaw = sin(yaw);
    double cos_yaw = cos(yaw);
    double arg = yaw + yawd * delta_t;

    // Predicted state values
    double px_p, py_p;
    // Avoid division by zero
    if (fabs(yawd) > EPS) {
      double v_yawd = v / yawd;
      px_p = p_x + v_yawd * (sin(arg) - sin_yaw);
      py_p = p_y + v_yawd * (cos_yaw - cos(arg));
    } else {
      double v_delta_t = v * delta_t;
      px_p = p_x + v_delta_t * cos_yaw;
      py_p = p_y + v_delta_t * sin_yaw;
    }
    double v_p = v;
    double yaw_p = arg;
    double yawd_p = yawd;

    // Add noise
    px_p += 0.5 * nu_a * delta_t2 * cos_yaw;
    py_p += 0.5 * nu_a * delta_t2 * sin_yaw;
    v_p += nu_a * delta_t;
    yaw_p += 0.5 * nu_yawdd * delta_t2;
    yawd_p += nu_yawdd * delta_t;

    // Write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  // 计算系统的加权均值以及不确定度
  x_ = Xsig_pred_ * weights_; // vectorised sum
  // Predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    NormAng(&(x_diff(3)));
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  int n_z = 3;   // 观测变量只有3维
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  // 将预测的7维状态变量的sigma点转换到3维观测空间
  for (int i = 0; i < n_sig_; i++) {
    // 列为状态空间维度，行为每一个sigma点
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);
    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;
    // Measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);          //r
    Zsig(1, i) = atan2(p_y, p_x);                   //phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / Zsig(0, i);   //r_dot
  }
  UpdateUKF(meas_package, Zsig, n_z);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  int n_z = 2;  //  设置观测维度
  // 将7*15的sigma 点集合转换到测量空间
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);
  UpdateUKF(meas_package, Zsig, n_z);
}

// Universal update function
void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, int n_z) {

  VectorXd z_pred = VectorXd(n_z); // 观测状态量 3*1或者2*1
  z_pred = Zsig * weights_;   // sigma点集加权均值
  MatrixXd S = MatrixXd(n_z, n_z);  //  观测不确定度 3*3或者2*2
  S.fill(0.0);

  // 计算状态量的加权均值
  for (int i = 0; i < n_sig_; i++) {
    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;  // 计算每个点状态量的残差
    NormAng(&(z_diff(1)));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // 测量噪声也加入进来
  MatrixXd R = MatrixXd(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar
    R = R_radar_;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) { // Lidar
    R = R_lidar_;
  }
  S = S + R;

  // 计算关联矩阵T 5*3或者5*2
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar
      NormAng(&(z_diff(1)));        // Angle normalization
    }
    // State difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // Angle normalization
    NormAng(&(x_diff(3)));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  // Measurements
  VectorXd z = meas_package.raw_measurements_;
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  // Residual
  VectorXd z_diff = z - z_pred;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar
    // Angle normalization
    NormAng(&(z_diff(1)));
  }
  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();
  // Calculate NIS
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) { // Radar
    NIS_radar_ = z.transpose() * S.inverse() * z;
  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) { // Lidar
    NIS_laser_ = z.transpose() * S.inverse() * z;
  }
}
