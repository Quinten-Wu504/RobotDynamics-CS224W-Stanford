# State and action fields

The decoder targets the pinned MimicGen **core** Square D0/D1 release. It checks the Panda OSC_POSE metadata, array dimensions, finite values, action bounds, quaternion norms, and timestamps before writing a cache. It is not a general decoder for arbitrary robosuite datasets.

Each cache has one group per original episode. Time-varying fields have N rows; fixture fields are stored once per episode. Float64 values are retained. Positions use metres in the MuJoCo world frame. Rotation matrices map local vectors to world vectors. The original HDF5 checksum, extraction commit, dirty-tree flag, environment metadata, timestep, and schema version are stored at the file root.

| Cache field | Shape | Meaning / source |
| --- | --- | --- |
| `time` | N | Simulation time in seconds, `states[:, 0]` |
| `actions` | N × 7 | Original normalized pose/gripper commands |
| `translation_command` | N × 3 | `actions[:, :3] * 0.05`, world-axis target displacement in m |
| `rotation_command` | N × 3 | `actions[:, 3:6] * 0.5`, axis-angle increment in rad |
| `eef_position` | N × 3 | `robot0_eef_pos`, the gripper site's world position |
| `eef_rotation` | N × 3 × 3 | Rotation of the robot's end-effector **body**, converted from `robot0_eef_quat` in xyzw order |
| `nut_position` | N × 3 | Square nut body origin, `object[:, :3]` |
| `nut_rotation` | N × 3 × 3 | Square nut body rotation, from `object[:, 3:7]` in xyzw order |
| `gripper_qpos` | N × 2 | Signed finger joint coordinates in m; not a single aperture value |
| `robot_joint_qpos` | N × 7 | Panda joint angles in rad |
| `sim_states` | N × 45 | Original simulator state for recovery only; mixed units, not a model-ready vector |
| `peg1/position`, `peg1/rotation` | 3, 3 × 3 | Target peg body pose, composed through its XML ancestors |
| `peg1/geom_position`, `peg1/geom_rotation`, `peg1/half_size` | 3, 3 × 3, 3 | Target collision box pose and half-extents |
| `table/position`, `table/rotation` | 3, 3 × 3 | Table body pose |
| `table/geom_position`, `table/geom_rotation`, `table/half_size` | 3, 3 × 3, 3 | Table collision box pose and half-extents |
| `table/surface_position`, `table/normal` | 3, 3 | Top-face centre and world unit normal |

The end-effector position and orientation come from different robosuite references: the position is at the grip site, while the quaternion describes the hand body. The hand-body and site orientations have a fixed offset in this robot model. `eef_rotation` retains the observation convention; it must not be described as the controller site's rotation. Simulator verification checks the position against `site_xpos` and the orientation against the named end-effector body's `body_xmat`.

D0 has a 14-value `object` observation: nut position (3), nut xyzw quaternion (4), nut position relative to the end effector (3), and relative quaternion (4). D1 appends the target peg world position (3), giving 17 values. The cache takes only the first seven values for the nut pose and obtains fixture geometry from each episode's XML. All D1 peg observations must agree with that XML pose. No zero padding or shared peg position is used. The inactive round nut and second peg are not task nodes.

The XML decoder supports the explicit static position/quaternion/box geometry in this release. It composes ancestor transforms and rejects fixtures under moving joints or unsupported transform encodings. The selected complete trajectories also check these poses against the fully restored simulator.

## Controller interpretation

Both datasets specify Panda, fixed-impedance OSC_POSE, delta control, normalized inputs in [-1, 1], translation limits ±0.05 m, and rotation limits ±0.5 rad. Position and orientation limits are unset, with no interpolation. Translation changes the world-coordinate controller target; rotation is left-multiplied as `R_delta @ R_current`, so its axis-angle is expressed in world axes. These are target commands, not the measured displacement achieved by the next state. The controller retains its orientation goal for an exactly zero rotation command; decoded increments alone do not reconstruct hidden controller history.

The seventh action component controls the Panda gripper: negative opens, positive closes. It is not a finger position or aperture in metres. The cache retains it in the raw action array, alongside the observed finger coordinates.

## Time alignment and unavailable labels

Metadata specifies 20 Hz and the recorded timestamps advance by 0.05 s. The upstream postprocessing routine associates `obs[t]` with `states[t]` and `actions[t]` with the transition to `states[t+1]`. The released files omit `next_obs`. For N recorded states there are N−1 observed one-step transitions; the final action is kept for provenance but has no recorded successor label. Multi-step windows must end at an existing state, with actions `[t:t+k]` and target state `t+k`. At this rate 0.25 s and 0.5 s correspond to 5 and 10 steps.

The simulator checks restore every state in three fixed episodes per distribution. Their short action replays are diagnostics, not replacement labels: the saved state does not contain the full solver/controller history, and fresh dynamics need not reproduce recorded trajectories exactly. The source format and state-to-observation agreement establish the indexing convention.

Recorded rewards and done flags describe the post-transition state in upstream postprocessing. They are not included as current-state model inputs. Recorded velocities are also omitted from the canonical cache; any later velocity feature will use causal history with an explicitly documented convention. The cache adds neither future-derived features nor split-dependent normalization. Episode splits and windows belong to the next stage.

## Source checks

- [MimicGen Square variants](https://github.com/NVlabs/mimicgen/blob/c0fbff22ba40060f05a917a80817f3f5c20731e7/mimicgen/envs/robosuite/nut_assembly.py)
- [robosuite nut observation ordering](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/environments/manipulation/nut_assembly.py)
- [End-effector observation references](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/robots/single_arm.py)
- [OSC goal construction](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/controllers/osc.py) and [rotation composition](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/utils/control_utils.py)
- [Panda gripper command](https://github.com/ARISE-Initiative/robosuite/blob/b9d8d3de5e3dfd1724f4a0e6555246c460407daa/robosuite/models/grippers/panda_gripper.py)
- [State-to-observation postprocessing and terminal handling](https://github.com/ARISE-Initiative/robomimic/blob/d0b37cf214bd24fb590d182edb6384333f67b661/robomimic/scripts/dataset_states_to_obs.py)
