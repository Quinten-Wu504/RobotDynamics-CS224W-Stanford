import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def xml_transform(element):
    if any(
        key in element.attrib
        for key in ("euler", "axisangle", "xyaxes", "zaxis", "class", "childclass")
    ):
        raise ValueError("Fixture transform needs an explicit position/quaternion")
    position = np.fromstring(element.get("pos", "0 0 0"), sep=" ")
    quaternion = np.fromstring(element.get("quat", "1 0 0 0"), sep=" ")
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
    ):
        raise ValueError("Invalid XML pose")
    return position, Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def fixture_poses(xml):
    root = ET.fromstring(xml)
    found = {}

    def walk(parent, position, rotation, moving):
        for body in parent.findall("body"):
            # Only parse transforms on paths to the two fixtures.
            names = {item.get("name") for item in body.iter("body")}
            if not names.intersection({"peg1", "table"}):
                continue
            local, local_rotation = xml_transform(body)
            world = position + rotation @ local
            world_rotation = rotation @ local_rotation
            dynamic = (
                moving
                or body.find("joint") is not None
                or body.find("freejoint") is not None
            )
            name = body.get("name")
            if name in {"peg1", "table"}:
                if dynamic:
                    raise ValueError(
                        f"{name} has a joint; XML alone cannot recover its pose"
                    )
                geom = (
                    body.find("geom[@name='table_collision']")
                    if name == "table"
                    else next(
                        (g for g in body.findall("geom") if g.get("group", "0") == "0"),
                        None,
                    )
                )
                if geom is None or geom.get("type") != "box":
                    raise ValueError(f"Expected explicit box geometry for {name}")
                offset, geom_rotation = xml_transform(geom)
                size = np.fromstring(geom.attrib["size"], sep=" ")
                if size.shape != (3,) or not np.all(size > 0):
                    raise ValueError("Invalid box half-size")
                found[name] = {
                    "position": world,
                    "rotation": world_rotation,
                    "geom_position": world + world_rotation @ offset,
                    "geom_rotation": world_rotation @ geom_rotation,
                    "half_size": size,
                }
            walk(body, world, world_rotation, dynamic)

    walk(root.find("worldbody"), np.zeros(3), np.eye(3), False)
    if set(found) != {"peg1", "table"}:
        raise ValueError("Missing peg1 or table in episode XML")
    table = found["table"]
    table["surface_position"] = (
        table["geom_position"] + table["geom_rotation"][:, 2] * table["half_size"][2]
    )
    table["normal"] = table["geom_rotation"][:, 2]
    return found


def rotations_xyzw(quaternion):
    quaternion = np.asarray(quaternion)
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=1e-5):
        raise ValueError("Observation quaternion is not unit length")
    return Rotation.from_quat(quaternion).as_matrix()
