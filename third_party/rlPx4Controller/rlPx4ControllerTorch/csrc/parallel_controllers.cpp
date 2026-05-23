#include "parallel_controllers.h"

#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include <ATen/ATen.h>

namespace rlpx4controller_torch {
namespace {

constexpr float kGravity{9.80665F};
constexpr float kPi{3.14159265358979323846F};
constexpr float kHoverThrust{0.1533F};
constexpr float kDerivativeCutoff{10.0F};
constexpr float kHoverThrMin{0.1F};
constexpr float kHoverThrMax{0.9F};
constexpr float kHoverNoise{0.1F};
constexpr float kProcessNoise{0.0036F};
constexpr float kInitialHoverThrust{0.4F};
constexpr float kInitialAccVar{5.0F};
constexpr float kNoiseLearningTimeConstant{2.0F};
constexpr float kLpfTimeConstant{1.0F};

constexpr std::array<float, 4> kRollScale{
    -0.707107F, 0.707107F, 0.707107F, -0.707107F};
constexpr std::array<float, 4> kPitchScale{
    -0.707107F, 0.707107F, -0.707107F, 0.707107F};
constexpr std::array<float, 4> kYawScale{-1.0F, -1.0F, 1.0F, 1.0F};
constexpr std::array<float, 4> kThrustScale{1.0F, 1.0F, 1.0F, 1.0F};

void check_matrix(
    const torch::Tensor& tensor,
    const char* name,
    const std::int64_t rows,
    const std::int64_t cols) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.dim() == 2, name, " must have shape [N, M]");
    TORCH_CHECK(tensor.size(0) == rows, name, " row count must match envs_num");
    TORCH_CHECK(tensor.size(1) == cols, name, " column count must be ", cols);
}

void check_dt(const double dt) {
    TORCH_CHECK(std::isfinite(dt), "dt must be finite");
    TORCH_CHECK(dt > 0.0, "dt must be positive");
}

torch::Tensor as_runtime_tensor(
    const torch::Tensor& tensor,
    const torch::TensorOptions& options) {
    return tensor.to(options, false, false).contiguous();
}

torch::Tensor scalar_tensor(
    const double value,
    const torch::TensorOptions& options) {
    return torch::full({1}, value, options);
}

torch::Tensor tensor_from_array3(
    const std::array<float, 3>& values,
    const torch::TensorOptions& options) {
    return torch::tensor({values[0], values[1], values[2]}, options);
}

torch::Tensor tensor_from_array4(
    const std::array<float, 4>& values,
    const torch::TensorOptions& options) {
    return torch::tensor({values[0], values[1], values[2], values[3]}, options);
}

torch::Tensor sign_zero_preserving(const torch::Tensor& tensor) {
    return tensor.gt(0).to(tensor.scalar_type()) -
           tensor.lt(0).to(tensor.scalar_type());
}

torch::Tensor clamp_tensor(
    const torch::Tensor& tensor,
    const double min_value,
    const double max_value) {
    return torch::clamp(tensor, min_value, max_value);
}

torch::Tensor stack_wxyz(
    const torch::Tensor& w,
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& z) {
    return torch::stack({w, x, y, z}, 1);
}

torch::Tensor normalize_quaternion(const torch::Tensor& q) {
    const auto norm =
        torch::sqrt(torch::sum(q * q, 1, true)).clamp_min(1.0e-12);
    return q / norm;
}

torch::Tensor inverse_quaternion(const torch::Tensor& q) {
    return stack_wxyz(
        q.select(1, 0),
        -q.select(1, 1),
        -q.select(1, 2),
        -q.select(1, 3));
}

torch::Tensor canonical_quaternion(const torch::Tensor& q) {
    const auto sign = sign_zero_preserving(q.select(1, 0)).unsqueeze(1);
    return q * sign;
}

torch::Tensor multiply_quaternion(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs) {
    const auto w1 = lhs.select(1, 0);
    const auto x1 = lhs.select(1, 1);
    const auto y1 = lhs.select(1, 2);
    const auto z1 = lhs.select(1, 3);

    const auto w2 = rhs.select(1, 0);
    const auto x2 = rhs.select(1, 1);
    const auto y2 = rhs.select(1, 2);
    const auto z2 = rhs.select(1, 3);

    const auto w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    const auto x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    const auto y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    const auto z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;

    return stack_wxyz(w, x, y, z);
}

torch::Tensor quaternion_from_rpy(
    const torch::Tensor& roll,
    const torch::Tensor& pitch,
    const torch::Tensor& yaw) {
    const auto half = scalar_tensor(0.5, roll.options()).squeeze(0);
    const auto half_roll = roll * half;
    const auto half_pitch = pitch * half;
    const auto half_yaw = yaw * half;

    const auto cr = torch::cos(half_roll);
    const auto sr = torch::sin(half_roll);
    const auto cp = torch::cos(half_pitch);
    const auto sp = torch::sin(half_pitch);
    const auto cy = torch::cos(half_yaw);
    const auto sy = torch::sin(half_yaw);

    const auto w = cr * cp * cy + sr * sp * sy;
    const auto x = sr * cp * cy - cr * sp * sy;
    const auto y = cr * sp * cy + sr * cp * sy;
    const auto z = cr * cp * sy - sr * sp * cy;

    return stack_wxyz(w, x, y, z);
}

torch::Tensor rotation_col2(const torch::Tensor& q) {
    const auto w = q.select(1, 0);
    const auto x = q.select(1, 1);
    const auto y = q.select(1, 2);
    const auto z = q.select(1, 3);

    const auto two = scalar_tensor(2.0, q.options()).squeeze(0);
    const auto col0 = two * (x * z + y * w);
    const auto col1 = two * (y * z - x * w);
    const auto col2 = 1.0 - two * (x * x + y * y);
    return torch::stack({col0, col1, col2}, 1);
}

torch::Tensor world_to_body_rate(
    const torch::Tensor& q,
    const torch::Tensor& rate_world) {
    const auto w = q.select(1, 0);
    const auto x = q.select(1, 1);
    const auto y = q.select(1, 2);
    const auto z = q.select(1, 3);

    const auto rx = rate_world.select(1, 0);
    const auto ry = rate_world.select(1, 1);
    const auto rz = rate_world.select(1, 2);

    const auto xx = x * x;
    const auto yy = y * y;
    const auto zz = z * z;

    const auto out_x =
        (1.0 - 2.0 * yy - 2.0 * zz) * rx + (2.0 * x * y + 2.0 * z * w) * ry +
        (2.0 * x * z - 2.0 * y * w) * rz;
    const auto out_y =
        (2.0 * x * y - 2.0 * z * w) * rx + (1.0 - 2.0 * xx - 2.0 * zz) * ry +
        (2.0 * y * z + 2.0 * x * w) * rz;
    const auto out_z =
        (2.0 * x * z + 2.0 * y * w) * rx + (2.0 * y * z - 2.0 * x * w) * ry +
        (1.0 - 2.0 * xx - 2.0 * yy) * rz;

    return torch::stack({out_x, out_y, out_z}, 1);
}

torch::Tensor yaw_from_quaternion(const torch::Tensor& q) {
    const auto w = q.select(1, 0);
    const auto x = q.select(1, 1);
    const auto y = q.select(1, 2);
    const auto z = q.select(1, 3);

    const auto numerator = 2.0 * (x * y + w * z);
    const auto denominator = w * w + x * x - y * y - z * z;
    return torch::atan2(numerator, denominator);
}

torch::Tensor get_attitude_error(
    const torch::Tensor& src,
    const torch::Tensor& dst) {
    const auto src_x = src.select(1, 0);
    const auto src_y = src.select(1, 1);
    const auto src_z = src.select(1, 2);
    const auto dst_x = dst.select(1, 0);
    const auto dst_y = dst.select(1, 1);
    const auto dst_z = dst.select(1, 2);

    auto cross = torch::stack(
        {src_y * dst_z - src_z * dst_y,
         src_z * dst_x - src_x * dst_z,
         src_x * dst_y - src_y * dst_x},
        1);
    const auto dot = torch::sum(src * dst, 1);
    const auto norm =
        torch::sqrt(torch::sum(cross * cross, 1)).clamp_min(0.0);
    const auto corner_case = norm.lt(1.0e-5) & dot.lt(0.0);

    if (corner_case.any().item<bool>()) {
        const auto abs_src = src.abs();
        const auto abs_x = abs_src.select(1, 0);
        const auto abs_y = abs_src.select(1, 1);
        const auto abs_z = abs_src.select(1, 2);

        const auto choose_x = (abs_x < abs_y) & (abs_x < abs_z);
        const auto choose_y = (~choose_x) & (abs_y < abs_z);
        const auto choose_z = ~(choose_x | choose_y);

        const auto basis = torch::stack(
            {choose_x.to(src.scalar_type()),
             choose_y.to(src.scalar_type()),
             choose_z.to(src.scalar_type())},
            1);

        const auto basis_x = basis.select(1, 0);
        const auto basis_y = basis.select(1, 1);
        const auto basis_z = basis.select(1, 2);

        auto corner_cross = torch::stack(
            {src_y * basis_z - src_z * basis_y,
             src_z * basis_x - src_x * basis_z,
             src_x * basis_y - src_y * basis_x},
            1);
        cross = torch::where(corner_case.unsqueeze(1), corner_cross, cross);
    }

    const auto q0 = torch::where(
        corner_case,
        torch::zeros_like(dot),
        dot + torch::sqrt(torch::sum(src * src, 1) * torch::sum(dst * dst, 1)));

    return normalize_quaternion(
        stack_wxyz(q0, cross.select(1, 0), cross.select(1, 1), cross.select(1, 2)));
}

torch::Tensor position_update(
    const torch::Tensor& pos,
    const torch::Tensor& vel,
    const torch::Tensor& quat,
    const torch::Tensor& hover_thrust,
    torch::Tensor& thrust_sp,
    const control_mode mode,
    const torch::Tensor& pos_sp,
    const torch::Tensor& vel_sp,
    const torch::Tensor& acc_sp,
    const torch::Tensor& yaw_sp) {
    auto des_acc = torch::zeros_like(pos);
    constexpr std::array<float, 3> kGains{1.5F, 1.5F, 1.5F};
    const auto gain = tensor_from_array3(kGains, pos.options()).view({1, 3});

    if (mode == control_mode::pos_only) {
        des_acc = acc_sp + gain * (pos_sp - pos);
    } else if (mode == control_mode::vel_only) {
        des_acc = acc_sp + gain * (vel_sp - vel);
    } else {
        des_acc = acc_sp + gain * (vel_sp - vel) + gain * (pos_sp - pos);
    }

    des_acc = clamp_tensor(des_acc, -4.0, 4.0);
    thrust_sp = des_acc.select(1, 2) * (hover_thrust / kGravity) + hover_thrust;

    const auto yaw_odom = yaw_from_quaternion(quat);
    const auto sin_yaw = torch::sin(yaw_odom);
    const auto cos_yaw = torch::cos(yaw_odom);
    const auto roll =
        (des_acc.select(1, 0) * sin_yaw - des_acc.select(1, 1) * cos_yaw) /
        kGravity;
    const auto pitch =
        (des_acc.select(1, 0) * cos_yaw + des_acc.select(1, 1) * sin_yaw) /
        kGravity;

    const auto q_sp = quaternion_from_rpy(roll, pitch, yaw_sp);
    return torch::cat({q_sp, thrust_sp.unsqueeze(1)}, 1);
}

torch::Tensor attitude_update(
    const torch::Tensor& q_sp,
    const torch::Tensor& q) {
    const auto q_current = normalize_quaternion(q);
    auto q_desired = normalize_quaternion(q_sp);
    const auto e_z = rotation_col2(q_current);
    const auto e_z_d = rotation_col2(q_desired);

    auto qd_red = get_attitude_error(e_z, e_z_d);
    const auto abs_x = qd_red.select(1, 1).abs();
    const auto abs_y = qd_red.select(1, 2).abs();
    const auto opposite = abs_x.gt(1.0 - 1.0e-5) | abs_y.gt(1.0 - 1.0e-5);
    const auto blended_red =
        multiply_quaternion(qd_red, q_current);
    qd_red = torch::where(opposite.unsqueeze(1), q_desired, blended_red);

    auto q_mix = multiply_quaternion(inverse_quaternion(qd_red), q_desired);
    q_mix = canonical_quaternion(q_mix);
    q_mix = stack_wxyz(
        clamp_tensor(q_mix.select(1, 0), -1.0, 1.0),
        q_mix.select(1, 1),
        q_mix.select(1, 2),
        clamp_tensor(q_mix.select(1, 3), -1.0, 1.0));

    constexpr std::array<float, 3> kAttitudeGain{8.0F, 8.0F, 2.5F};
    constexpr std::array<float, 3> kRateLimit{
        1600.0F / 57.3F, 1600.0F / 57.3F, 1000.0F / 57.3F};
    constexpr float kYawWeight{0.4F};
    const auto q_yaw = stack_wxyz(
        torch::cos(kYawWeight * torch::acos(q_mix.select(1, 0))),
        torch::zeros_like(q_mix.select(1, 0)),
        torch::zeros_like(q_mix.select(1, 0)),
        torch::sin(kYawWeight * torch::asin(q_mix.select(1, 3))));
    q_desired = multiply_quaternion(qd_red, q_yaw);

    auto q_error =
        multiply_quaternion(inverse_quaternion(q_current), q_desired);
    q_error = canonical_quaternion(q_error);
    auto eq = q_error.slice(1, 1, 4) * 2.0;

    const auto gain = tensor_from_array3(kAttitudeGain, q.options()).view({1, 3});
    auto rate_setpoint = eq * gain;
    const auto rate_limit = tensor_from_array3(kRateLimit, q.options()).view({1, 3});
    rate_setpoint = torch::maximum(
        torch::minimum(rate_setpoint, rate_limit), -rate_limit);
    return rate_setpoint;
}

torch::Tensor rate_update(
    const torch::Tensor& q_world,
    torch::Tensor& rate_int,
    const torch::Tensor& rate_sp,
    const torch::Tensor& rate,
    const torch::Tensor& angular_accel,
    const double dt) {
    constexpr std::array<float, 3> kGainP{0.5F, 0.5F, 0.2F};
    constexpr std::array<float, 3> kGainI{0.08F, 0.08F, 0.05F};
    constexpr std::array<float, 3> kGainD{0.001F, 0.001F, 0.0F};
    constexpr std::array<float, 3> kLimitInt{0.3F, 0.3F, 0.3F};

    const auto gain_p = tensor_from_array3(kGainP, q_world.options()).view({1, 3});
    const auto gain_i = tensor_from_array3(kGainI, q_world.options()).view({1, 3});
    const auto gain_d = tensor_from_array3(kGainD, q_world.options()).view({1, 3});
    const auto limit_int = tensor_from_array3(kLimitInt, q_world.options()).view({1, 3});

    const auto body_rate = world_to_body_rate(q_world, rate);
    const auto rate_error = rate_sp - body_rate;
    const auto torque = gain_p * rate_error + rate_int - gain_d * angular_accel;

    const auto radians_400 = 400.0 * kPi / 180.0;
    auto i_factor = rate_error / radians_400;
    i_factor = torch::maximum(torch::zeros_like(i_factor), 1.0 - i_factor * i_factor);
    rate_int = rate_int + i_factor * gain_i * rate_error * dt;
    rate_int = torch::maximum(torch::minimum(rate_int, limit_int), -limit_int);

    return torque;
}

torch::Tensor compute_desaturation_gain(
    const torch::Tensor& desaturation_vector,
    const torch::Tensor& outputs,
    const double min_output,
    const double max_output) {
    auto k_min = torch::zeros({outputs.size(0)}, outputs.options());
    auto k_max = torch::zeros({outputs.size(0)}, outputs.options());
    constexpr float kEpsilon = std::numeric_limits<float>::epsilon();

    for (std::int64_t index = 0; index < 4; ++index) {
        const auto desaturation = desaturation_vector.select(1, index);
        const auto output = outputs.select(1, index);
        const auto active = desaturation.abs().ge(kEpsilon);

        const auto under = active & output.lt(min_output);
        const auto over = active & output.gt(max_output);

        const auto under_gain = torch::where(
            under,
            (min_output - output) / desaturation,
            torch::zeros_like(output));
        const auto over_gain = torch::where(
            over,
            (max_output - output) / desaturation,
            torch::zeros_like(output));

        k_min = torch::minimum(k_min, under_gain);
        k_min = torch::minimum(k_min, over_gain);
        k_max = torch::maximum(k_max, under_gain);
        k_max = torch::maximum(k_max, over_gain);
    }

    return k_min + k_max;
}

void minimize_saturation(
    const torch::Tensor& desaturation_vector,
    torch::Tensor& outputs,
    const double min_output,
    const double max_output,
    const bool reduce_only) {
    const auto k1 =
        compute_desaturation_gain(desaturation_vector, outputs, min_output, max_output);
    if (reduce_only) {
        const auto should_update = k1.le(0.0);
        const auto expanded_mask = should_update.unsqueeze(1);
        outputs = torch::where(
            expanded_mask,
            outputs + k1.unsqueeze(1) * desaturation_vector,
            outputs);
        const auto k2 = 0.5 *
                        compute_desaturation_gain(
                            desaturation_vector, outputs, min_output, max_output);
        outputs = torch::where(
            expanded_mask,
            outputs + k2.unsqueeze(1) * desaturation_vector,
            outputs);
        return;
    } else {
        outputs = outputs + k1.unsqueeze(1) * desaturation_vector;
    }

    const auto k2 = 0.5 *
                    compute_desaturation_gain(
                        desaturation_vector, outputs, min_output, max_output);
    outputs = outputs + k2.unsqueeze(1) * desaturation_vector;
}

torch::Tensor mixer_update(const torch::Tensor& torque_thrust) {
    auto roll = clamp_tensor(torque_thrust.select(1, 0), -1.0, 1.0);
    auto pitch = clamp_tensor(torque_thrust.select(1, 1), -1.0, 1.0);
    auto yaw = clamp_tensor(torque_thrust.select(1, 2), -1.0, 1.0);
    auto thrust = clamp_tensor(torque_thrust.select(1, 3), 0.0, 1.0);

    const auto options = torque_thrust.options();
    const auto roll_scale = tensor_from_array4(kRollScale, options).view({1, 4});
    const auto pitch_scale = tensor_from_array4(kPitchScale, options).view({1, 4});
    const auto yaw_scale = tensor_from_array4(kYawScale, options).view({1, 4});
    const auto thrust_scale = tensor_from_array4(kThrustScale, options).view({1, 4});

    auto outputs = roll.unsqueeze(1) * roll_scale + pitch.unsqueeze(1) * pitch_scale +
                   thrust.unsqueeze(1) * thrust_scale;
    minimize_saturation(thrust_scale.expand_as(outputs), outputs, 0.0, 1.0, true);
    minimize_saturation(roll_scale.expand_as(outputs), outputs, 0.0, 1.0, false);
    minimize_saturation(pitch_scale.expand_as(outputs), outputs, 0.0, 1.0, false);

    outputs = outputs + yaw.unsqueeze(1) * yaw_scale;
    minimize_saturation(yaw_scale.expand_as(outputs), outputs, 0.0, 1.15, false);
    minimize_saturation(thrust_scale.expand_as(outputs), outputs, 0.0, 1.0, true);
    return clamp_tensor(outputs, 0.0, 1.0);
}

}  // namespace

ParallelPositionVelocityController::ParallelPositionVelocityController(
    const std::int64_t envs_num,
    std::string device,
    const control_mode mode)
    : envs_num_(envs_num),
      device_(std::move(device)),
      float_options_(torch::TensorOptions().dtype(torch::kFloat32).device(device_)),
      bool_options_(torch::TensorOptions().dtype(torch::kBool).device(device_)),
      mode_(mode),
      pos_(torch::zeros({envs_num_, 3}, float_options_)),
      vel_(torch::zeros({envs_num_, 3}, float_options_)),
      quat_(torch::zeros({envs_num_, 4}, float_options_)),
      ang_vel_(torch::zeros({envs_num_, 3}, float_options_)),
      hover_thrust_(torch::full({envs_num_}, kHoverThrust, float_options_)),
      hover_thr_state_(torch::full({envs_num_}, kInitialHoverThrust, float_options_)),
      thrust_sp_(torch::zeros({envs_num_}, float_options_)),
      prev_vel_z_(torch::zeros({envs_num_}, float_options_)),
      vel_z_lpf_state_(torch::zeros({envs_num_}, float_options_)),
      derivative_initialized_(torch::zeros({envs_num_}, bool_options_)),
      ekf_state_var_(
          torch::full({envs_num_}, kHoverNoise * kHoverNoise, float_options_)),
      ekf_acc_var_(torch::full({envs_num_}, kInitialAccVar, float_options_)),
      ekf_residual_lpf_(torch::zeros({envs_num_}, float_options_)),
      ekf_signed_innov_ratio_lpf_(torch::zeros({envs_num_}, float_options_)),
      rate_q_world_(torch::zeros({envs_num_, 4}, float_options_)),
      rate_int_(torch::zeros({envs_num_, 3}, float_options_)) {
    TORCH_CHECK(envs_num_ > 0, "envs_num must be positive");
}

void ParallelPositionVelocityController::set_status(
    const torch::Tensor& pos,
    const torch::Tensor& q_matrix,
    const torch::Tensor& vel,
    const torch::Tensor& ang_vel,
    const double dt) {
    check_dt(dt);
    check_matrix(pos, "pos", envs_num_, 3);
    check_matrix(q_matrix, "q_matrix", envs_num_, 4);
    check_matrix(vel, "vel", envs_num_, 3);
    check_matrix(ang_vel, "ang_vel", envs_num_, 3);

    pos_ = as_runtime_tensor(pos, float_options_);
    quat_ = as_runtime_tensor(q_matrix, float_options_);
    vel_ = as_runtime_tensor(vel, float_options_);
    ang_vel_ = as_runtime_tensor(ang_vel, float_options_);
    rate_q_world_ = quat_;
    dt_ = static_cast<float>(dt);

    const auto thrust_z = rotation_col2(quat_) * thrust_sp_.unsqueeze(1);
    ekf_state_var_ += (kProcessNoise * kProcessNoise) * dt_ * dt_;

    const auto vel_z = vel_.select(1, 2);
    const auto derivative = (vel_z - prev_vel_z_) / dt_;
    const auto low_pass_gain =
        2.0 * kPi * kDerivativeCutoff * dt_ /
        (1.0 + 2.0 * kPi * kDerivativeCutoff * dt_);
    const auto updated_lpf =
        low_pass_gain * derivative + (1.0 - low_pass_gain) * vel_z_lpf_state_;
    const auto acc_z = torch::where(
        derivative_initialized_,
        updated_lpf,
        torch::zeros_like(vel_z));
    vel_z_lpf_state_ = torch::where(
        derivative_initialized_,
        updated_lpf,
        torch::zeros_like(updated_lpf));
    derivative_initialized_ = torch::ones({envs_num_}, bool_options_);
    prev_vel_z_ = vel_z;

    const auto thrust_z_world = thrust_z.select(1, 2);
    const auto hover_sq = hover_thr_state_ * hover_thr_state_;
    const auto h = -kGravity * thrust_z_world / hover_sq;
    const auto p = ekf_state_var_;
    const auto innov_var =
        torch::maximum(h * p * h + ekf_acc_var_, ekf_acc_var_);
    const auto predicted_acc_z = kGravity * thrust_z_world / hover_thr_state_ - kGravity;
    const auto innov = acc_z - predicted_acc_z;
    const auto kalman_gain = p * h / innov_var;
    hover_thr_state_ = clamp_tensor(
        hover_thr_state_ + kalman_gain * innov,
        kHoverThrMin,
        kHoverThrMax);
    ekf_state_var_ = clamp_tensor((1.0 - kalman_gain * h) * ekf_state_var_, 1.0e-10, 1.0);

    const auto residual =
        acc_z - (kGravity * thrust_z_world / hover_thr_state_ - kGravity);
    const auto alpha_residual = dt_ / (kLpfTimeConstant + dt_);
    const auto signed_innov_ratio =
        sign_zero_preserving(innov) * innov * innov / (9.0 * innov_var);
    ekf_residual_lpf_ =
        (1.0 - alpha_residual) * ekf_residual_lpf_ + alpha_residual * residual;
    ekf_signed_innov_ratio_lpf_ =
        (1.0 - alpha_residual) * ekf_signed_innov_ratio_lpf_ +
        alpha_residual * clamp_tensor(signed_innov_ratio, -1.0, 1.0);
    const auto alpha_noise = dt_ / (kNoiseLearningTimeConstant + dt_);
    const auto residual_no_bias = residual - ekf_residual_lpf_;
    ekf_acc_var_ = clamp_tensor(
        (1.0 - alpha_noise) * ekf_acc_var_ +
            alpha_noise * (residual_no_bias * residual_no_bias + h * ekf_state_var_ * h),
        1.0,
        400.0);

    hover_thrust_ = torch::full({envs_num_}, kHoverThrust, float_options_);
}

torch::Tensor ParallelPositionVelocityController::update(const torch::Tensor& actions) {
    check_matrix(actions, "actions", envs_num_, 4);
    const auto runtime_actions = as_runtime_tensor(actions, float_options_);
    const auto zero_vector = torch::zeros({envs_num_, 3}, float_options_);
    const auto yaw_sp = runtime_actions.select(1, 3);

    torch::Tensor atti_thrust;
    if (mode_ == control_mode::vel_only) {
        atti_thrust = position_update(
            pos_,
            vel_,
            quat_,
            hover_thrust_,
            thrust_sp_,
            mode_,
            zero_vector,
            runtime_actions.slice(1, 0, 3),
            zero_vector,
            yaw_sp);
    } else {
        atti_thrust = position_update(
            pos_,
            vel_,
            quat_,
            hover_thrust_,
            thrust_sp_,
            mode_,
            runtime_actions.slice(1, 0, 3),
            zero_vector,
            zero_vector,
            yaw_sp);
    }

    const auto rate_sp = attitude_update(atti_thrust.slice(1, 0, 4), quat_);
    const auto torque_sp = rate_update(
        rate_q_world_,
        rate_int_,
        rate_sp,
        ang_vel_,
        zero_vector,
        dt_);
    const auto torque_thrust = torch::cat(
        {torque_sp, atti_thrust.select(1, 4).unsqueeze(1)},
        1);
    return mixer_update(torque_thrust);
}

ParallelPosControl::ParallelPosControl(
    const std::int64_t envs_num,
    std::string device)
    : ParallelPositionVelocityController(
          envs_num,
          std::move(device),
          control_mode::all) {}

ParallelVelControl::ParallelVelControl(
    const std::int64_t envs_num,
    std::string device)
    : ParallelPositionVelocityController(
          envs_num,
          std::move(device),
          control_mode::vel_only) {}

ParallelAttiControl::ParallelAttiControl(
    const std::int64_t envs_num,
    std::string device)
    : envs_num_(envs_num),
      device_(std::move(device)),
      float_options_(torch::TensorOptions().dtype(torch::kFloat32).device(device_)),
      quat_(torch::zeros({envs_num_, 4}, float_options_)),
      ang_vel_(torch::zeros({envs_num_, 3}, float_options_)),
      rate_q_world_(torch::zeros({envs_num_, 4}, float_options_)),
      rate_int_(torch::zeros({envs_num_, 3}, float_options_)) {
    TORCH_CHECK(envs_num_ > 0, "envs_num must be positive");
}

void ParallelAttiControl::set_status(
    const torch::Tensor& pos,
    const torch::Tensor& q_matrix,
    const torch::Tensor& vel,
    const torch::Tensor& ang_vel,
    const double dt) {
    check_dt(dt);
    check_matrix(pos, "pos", envs_num_, 3);
    check_matrix(q_matrix, "q_matrix", envs_num_, 4);
    check_matrix(vel, "vel", envs_num_, 3);
    check_matrix(ang_vel, "ang_vel", envs_num_, 3);

    quat_ = as_runtime_tensor(q_matrix, float_options_);
    ang_vel_ = as_runtime_tensor(ang_vel, float_options_);
    rate_q_world_ = quat_;
    dt_ = static_cast<float>(dt);
}

torch::Tensor ParallelAttiControl::update(const torch::Tensor& actions) {
    check_matrix(actions, "actions", envs_num_, 5);
    const auto runtime_actions = as_runtime_tensor(actions, float_options_);
    const auto zero_vector = torch::zeros({envs_num_, 3}, float_options_);
    const auto rate_sp = attitude_update(runtime_actions.slice(1, 0, 4), quat_);
    const auto torque_sp = rate_update(
        rate_q_world_,
        rate_int_,
        rate_sp,
        ang_vel_,
        zero_vector,
        dt_);
    const auto torque_thrust = torch::cat(
        {torque_sp, runtime_actions.select(1, 4).unsqueeze(1)},
        1);
    return mixer_update(torque_thrust);
}

ParallelRateControl::ParallelRateControl(
    const std::int64_t envs_num,
    std::string device)
    : envs_num_(envs_num),
      device_(std::move(device)),
      float_options_(torch::TensorOptions().dtype(torch::kFloat32).device(device_)),
      q_world_(torch::zeros({envs_num_, 4}, float_options_)),
      rate_int_(torch::zeros({envs_num_, 3}, float_options_)) {
    TORCH_CHECK(envs_num_ > 0, "envs_num must be positive");
}

void ParallelRateControl::set_q_world(const torch::Tensor& q_world) {
    check_matrix(q_world, "q_world", envs_num_, 4);
    q_world_ = as_runtime_tensor(q_world, float_options_);
}

torch::Tensor ParallelRateControl::update(
    const torch::Tensor& actions,
    const torch::Tensor& rate,
    const double dt) {
    check_dt(dt);
    check_matrix(actions, "actions", envs_num_, 4);
    check_matrix(rate, "rate", envs_num_, 3);

    const auto runtime_actions = as_runtime_tensor(actions, float_options_);
    const auto runtime_rate = as_runtime_tensor(rate, float_options_);
    const auto zero_vector = torch::zeros({envs_num_, 3}, float_options_);
    const auto torque_sp = rate_update(
        q_world_,
        rate_int_,
        runtime_actions.slice(1, 0, 3),
        runtime_rate,
        zero_vector,
        dt);
    const auto torque_thrust = torch::cat(
        {torque_sp, runtime_actions.select(1, 3).unsqueeze(1)},
        1);
    return mixer_update(torque_thrust);
}

}  // namespace rlpx4controller_torch
