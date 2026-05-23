#ifndef RLPX4CONTROLLERTORCH_PARALLEL_CONTROLLERS_H_
#define RLPX4CONTROLLERTORCH_PARALLEL_CONTROLLERS_H_

#include <cstdint>
#include <string>

#include <torch/extension.h>

namespace rlpx4controller_torch {

enum class control_mode : std::uint8_t {
    all,
    pos_only,
    vel_only,
};

class ParallelPositionVelocityController {
public:
    explicit ParallelPositionVelocityController(
        std::int64_t envs_num,
        std::string device,
        control_mode mode);

    void set_status(
        const torch::Tensor& pos,
        const torch::Tensor& q_matrix,
        const torch::Tensor& vel,
        const torch::Tensor& ang_vel,
        double dt);

    torch::Tensor update(const torch::Tensor& actions);

protected:
    std::int64_t envs_num_;
    c10::Device device_;
    torch::TensorOptions float_options_;
    torch::TensorOptions bool_options_;
    control_mode mode_;

    torch::Tensor pos_;
    torch::Tensor vel_;
    torch::Tensor quat_;
    torch::Tensor ang_vel_;

    torch::Tensor hover_thrust_;
    torch::Tensor hover_thr_state_;
    torch::Tensor thrust_sp_;

    torch::Tensor prev_vel_z_;
    torch::Tensor vel_z_lpf_state_;
    torch::Tensor derivative_initialized_;

    torch::Tensor ekf_state_var_;
    torch::Tensor ekf_acc_var_;
    torch::Tensor ekf_residual_lpf_;
    torch::Tensor ekf_signed_innov_ratio_lpf_;

    torch::Tensor rate_q_world_;
    torch::Tensor rate_int_;

    float dt_{0.01F};
};

class ParallelPosControl final : public ParallelPositionVelocityController {
public:
    explicit ParallelPosControl(std::int64_t envs_num, std::string device);
};

class ParallelVelControl final : public ParallelPositionVelocityController {
public:
    explicit ParallelVelControl(std::int64_t envs_num, std::string device);
};

class ParallelAttiControl {
public:
    explicit ParallelAttiControl(std::int64_t envs_num, std::string device);

    void set_status(
        const torch::Tensor& pos,
        const torch::Tensor& q_matrix,
        const torch::Tensor& vel,
        const torch::Tensor& ang_vel,
        double dt);

    torch::Tensor update(const torch::Tensor& actions);

private:
    std::int64_t envs_num_;
    c10::Device device_;
    torch::TensorOptions float_options_;

    torch::Tensor quat_;
    torch::Tensor ang_vel_;
    torch::Tensor rate_q_world_;
    torch::Tensor rate_int_;
    float dt_{0.01F};
};

class ParallelRateControl {
public:
    explicit ParallelRateControl(std::int64_t envs_num, std::string device);

    void set_q_world(const torch::Tensor& q_world);

    torch::Tensor update(
        const torch::Tensor& actions,
        const torch::Tensor& rate,
        double dt);

private:
    std::int64_t envs_num_;
    c10::Device device_;
    torch::TensorOptions float_options_;

    torch::Tensor q_world_;
    torch::Tensor rate_int_;
};

}  // namespace rlpx4controller_torch

#endif  // RLPX4CONTROLLERTORCH_PARALLEL_CONTROLLERS_H_
