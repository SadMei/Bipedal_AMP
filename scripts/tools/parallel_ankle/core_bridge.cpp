#include "AnkleKinematics.hpp"
#include <cstddef>

// Parallel-ankle test/runtime ABI around the unmodified, uploaded robot solver.
using namespace bsrl::ethercat;
extern "C" int ankle_forward(int right, const double* q, double* motor, double* j) {
  static const AnkleKinematics left(defaultBsrlAnkleGeometry(AnkleSide::kLeft));
  static const AnkleKinematics rhs(defaultBsrlAnkleGeometry(AnkleSide::kRight));
  Vector2 m{}; Matrix2 jac{};
  auto status = (right ? rhs : left).jointToMotor({q[0], q[1]}, m, &jac);
  for (int a=0; a<2; ++a) {
    motor[a]=m[a];
    for (int b=0; b<2; ++b) j[2*a+b]=jac[a][b];
  }
  return static_cast<int>(status);
}
extern "C" int ankle_inverse(int right, const double* motor, const double* initial,
                              double* q) {
  static const AnkleKinematics left(defaultBsrlAnkleGeometry(AnkleSide::kLeft));
  static const AnkleKinematics rhs(defaultBsrlAnkleGeometry(AnkleSide::kRight));
  Vector2 out{};
  auto status = (right ? rhs : left).motorToJoint(
      {motor[0],motor[1]}, {initial[0],initial[1]}, {-.6,-.52}, {.6,.52}, out);
  q[0]=out[0]; q[1]=out[1];
  return static_cast<int>(status);
}
extern "C" void ankle_geometry(int right, double* out) {
  const auto g=defaultBsrlAnkleGeometry(right ? AnkleSide::kRight : AnkleSide::kLeft);
  int k=0;
  for (auto values : {g.l1,g.l2,g.l3,g.l4,g.radius}) for (auto v:values) out[k++]=v;
  for (auto row:g.d0) for(auto v:row) out[k++]=v;
  out[k++]=g.paper_x_robot_y_sign;
  for(auto v:g.lateral_sign) out[k++]=v;
  for(auto v:g.output_direction) out[k++]=v;
}
