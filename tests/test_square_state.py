import numpy as np
import pytest

from robot_dynamics.square_state import fixture_poses, rotations_xyzw


def fixture_xml(joint=""):
    return f"""<mujoco><worldbody><body pos="1 2 3" quat="0.7071067811865476 0 0 0.7071067811865476">
    {joint}<body name="peg1" pos="1 0 0"><geom type="box" size=".01 .01 .1"/></body>
    <body name="table" pos="0 0 -.2"><geom name="table_collision" type="box" size=".4 .4 .2"/></body>
    </body></worldbody></mujoco>"""


def test_nested_fixture_transform_and_table_surface():
    poses = fixture_poses(fixture_xml())
    np.testing.assert_allclose(poses["peg1"]["position"], [1, 3, 3])
    np.testing.assert_allclose(poses["table"]["surface_position"], [1, 2, 3])
    np.testing.assert_allclose(poses["table"]["normal"], [0, 0, 1])


def test_dynamic_fixture_cannot_be_recovered_from_xml_only():
    with pytest.raises(ValueError, match="joint"):
        fixture_poses(fixture_xml('<joint type="slide"/>'))


def test_xyzw_rotation_and_invalid_quaternion():
    np.testing.assert_allclose(rotations_xyzw([[0, 0, 0, 1]])[0], np.eye(3))
    with pytest.raises(ValueError, match="unit length"):
        rotations_xyzw([[0, 0, 0, 2]])
