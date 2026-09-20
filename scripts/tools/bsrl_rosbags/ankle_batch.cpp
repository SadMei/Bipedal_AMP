#include "AnkleKinematics.hpp"
#include <cstddef>

// Rosbag diagnostics only: use the user's unmodified kinematics implementation.
extern "C" void ankle_forward_batch(int right, std::size_t count,
    const double* joints, double* motors, double* jacobians, int* statuses) {
  using namespace bsrl::ethercat;
  AnkleKinematics model(defaultBsrlAnkleGeometry(
      right ? AnkleSide::kRight : AnkleSide::kLeft));
  for (std::size_t i = 0; i < count; ++i) {
    Vector2 motor{};
    Matrix2 jacobian{};
    statuses[i] = static_cast<int>(model.jointToMotor(
        {joints[2*i], joints[2*i+1]}, motor, &jacobian));
    for (std::size_t r = 0; r < 2; ++r) {
      motors[2*i+r] = motor[r];
      for (std::size_t c = 0; c < 2; ++c)
        jacobians[4*i+2*r+c] = jacobian[r][c];
    }
  }
}
