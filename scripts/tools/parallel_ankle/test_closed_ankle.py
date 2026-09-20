"""CPU regression tests for robot geometry, closure, signs and policy compatibility."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

if __package__:
    from .build_models import SERIAL_XML, full_robot_xml
    from .check_mounting import audit, cad_mesh
    from .real_ankle import Geometry, RobotGeometry, RobotCore, closure_gaps, standalone_xml
    from .replay_policy import BASELINE_RUN, JOINTS, Replay, clip_motor
else:
    from build_models import SERIAL_XML, full_robot_xml
    from check_mounting import audit, cad_mesh
    from real_ankle import Geometry, RobotGeometry, RobotCore, closure_gaps, standalone_xml
    from replay_policy import BASELINE_RUN, JOINTS, Replay, clip_motor


@pytest.fixture(scope="module")
def core():
    return RobotCore()


@pytest.mark.parametrize("side", ["right", "left"])
def test_installation_matches_independent_urdf_cad(core, side):
    geometry = Geometry(side, core)
    report = audit(geometry, cad_mesh(side, "shank"))
    assert max(report["foot_endpoint_offset_mm"]) < 1e-6
    assert max(report["crank_endpoint_offset_mm"]) < 1e-6
    toe = cad_mesh(side, "ankle_roll")
    assert toe.bounds[1, 0] > .15  # shoe points +X; rods must remain on -X
    np.testing.assert_array_equal(geometry.installation_rotation @ [0., 0., 1.], [0., 0., 1.])
    np.testing.assert_allclose(np.linalg.det(geometry.installation_rotation), 1.)
    motor, _ = geometry.forward(np.zeros(2))
    np.testing.assert_array_equal(motor, np.zeros(2))
    # The previous self-consistent-but-unmounted model must fail this CAD check.
    geometry.mounts = geometry.core_mounts.copy()
    with pytest.raises(AssertionError):
        audit(geometry, cad_mesh(side, "shank"))


@pytest.mark.parametrize("side", ["right", "left"])
def test_full_joint_workspace_closes_against_cad_constraints(core, side):
    g=Geometry(side,core)
    model=mujoco.MjModel.from_xml_string(standalone_xml(g)); data=mujoco.MjData(model)
    worst=0.
    for p in np.linspace(-.6,.6,21):
        for r in np.linspace(-.52,.52,21):
            q=np.array([p,r])
            motor,jac=g.forward(q)
            independent=g.output_direction*(g.raw_angles(q)-g.alpha0)
            np.testing.assert_allclose(motor,independent,atol=2e-14,rtol=0)
            np.testing.assert_allclose(g.inverse(motor,q),q,atol=1e-8,rtol=0)
            mujoco.mj_resetData(model,data)
            data.joint("ankle_pitch").qpos[0]=p
            data.joint("ankle_roll").qpos[0]=r
            g.initialize(model,data,q,"")
            worst=max(worst,float(closure_gaps(model,data,[""]).max()))
            assert np.linalg.cond(jac)<4
    assert worst<1e-12


@pytest.mark.parametrize("side", ["right", "left"])
def test_legacy_cpp_preserved_and_cad_jacobian_finite_difference(core, side):
    legacy, cad = RobotGeometry(side, core), Geometry(side, core)
    for q in (np.array([.4, .3]), np.array([-.4, -.3])):
        old_m, old_j = core.forward(-q, side)
        m, j = legacy.forward(q)
        np.testing.assert_array_equal(m, old_m)
        np.testing.assert_array_equal(j, -old_j)
        _, actual = cad.forward(q)
        numerical = np.column_stack([(cad.forward(q+delta)[0]-cad.forward(q-delta)[0])/2e-6
                                     for delta in np.eye(2)*1e-6])
        np.testing.assert_allclose(actual, numerical, atol=2e-8, rtol=0)


@pytest.mark.parametrize("side", ["right", "left"])
def test_mujoco_constraint_jacobian_matches_robot_solver(core,side):
    g=Geometry(side,core)
    m=mujoco.MjModel.from_xml_string(standalone_xml(g)); d=mujoco.MjData(m)
    for q in (np.zeros(2),np.array([-.2,0]),np.array([.4,.3]),np.array([-.4,-.3])):
        mujoco.mj_resetData(m,d)
        d.joint("ankle_pitch").qpos[0],d.joint("ankle_roll").qpos[0]=q
        g.initialize(m,d,q,""); mujoco.mj_forward(m,d)
        constraint=[]
        for i in (1,2):
            rod=np.zeros((3,m.nv)); foot=np.zeros((3,m.nv)); unused=np.zeros_like(rod)
            mujoco.mj_jacSite(m,d,rod,unused,m.site(f"rod_end_{i}").id)
            mujoco.mj_jacSite(m,d,foot,unused,m.site(f"foot_end_{i}").id)
            constraint.append(rod-foot)
        constraint=np.vstack(constraint)
        ankles=np.array([m.joint(n).dofadr[0] for n in ("ankle_pitch","ankle_roll")])
        remaining=np.array([i for i in range(m.nv) if i not in ankles])
        implied=np.linalg.lstsq(constraint[:,remaining],-constraint[:,ankles],rcond=None)[0]
        motor_rows=[list(remaining).index(m.joint(f"motor_{i}").dofadr[0]) for i in (1,2)]
        motor,jac=g.forward(q)
        np.testing.assert_allclose(g.inverse(motor,np.zeros(2)),q,atol=1e-8,rtol=0)
        np.testing.assert_allclose(implied[motor_rows],jac,atol=2e-8,rtol=0)


def test_passive_ankles_zero_pose_original_mass_and_observations(core):
    serial=Replay(BASELINE_RUN,"serial",core,[.6,0,0])
    closed=Replay(BASELINE_RUN,"closed",core,[.6,0,0])
    assert closed.model.nu==12 and closed.model.neq==4
    assert len(closed.direct_ids)==8
    assert closed.initial_gaps.max()<1e-12
    for body_id in range(1,serial.model.nbody):
        name=serial.model.body(body_id).name
        old,new=serial.model.body(name),closed.model.body(name)
        np.testing.assert_array_equal(old.mass,new.mass)
        np.testing.assert_array_equal(old.inertia,new.inertia)
        np.testing.assert_allclose(serial.data.body(name).xpos,closed.data.body(name).xpos,atol=1e-12)
        np.testing.assert_allclose(serial.data.body(name).xquat,closed.data.body(name).xquat,atol=1e-12)
    np.testing.assert_allclose(closed.model.body_mass.sum()-serial.model.body_mass.sum(),.008,atol=1e-12)
    np.testing.assert_array_equal(serial.observation(),closed.observation())
    closed.target[closed.ankle_indices["right"]]+=np.array([.1,.05])
    closed.apply(); mujoco.mj_forward(closed.model,closed.data)
    for ix in closed.ankle_indices.values():
        assert np.all(closed.data.qfrc_actuator[closed._qvel_adr[ix]]==0)
    assert np.any(closed.data.actuator_force!=0)


def test_joint_pd_roundtrip_and_coupled_motor_limits(core):
    for side in ("right","left"):
        _,j=Geometry(side,core).forward([-.2,0])
        # Enough combined pitch/roll demand to saturate under the corrected J.
        joint=np.array([100.,80.])
        motor=np.linalg.solve(j.T,joint)
        np.testing.assert_allclose(j.T@motor,joint,atol=1e-12)
        assert np.abs(motor).max()>112
        clipped=clip_motor(motor,np.zeros(2),112,26.18,1250)
        assert np.abs(clipped).max()<=112
        assert not np.allclose(j.T@clipped,joint)
        qdot=np.array([.4,-.3])
        np.testing.assert_allclose(clipped@(j@qdot),(j.T@clipped)@qdot,atol=1e-12)
    # Overspeed can brake, but cannot accelerate farther out of the envelope.
    np.testing.assert_array_equal(clip_motor([10,-10],np.array([30.,-30.]),112,26.18,1250),[0,0])
    np.testing.assert_array_equal(clip_motor([-10,10],np.array([30.,-30.]),112,26.18,1250),[-10,10])
