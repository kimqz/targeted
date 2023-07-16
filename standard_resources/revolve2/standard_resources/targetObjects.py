# whole file added by
# Qinwan

"""Standard terrains."""

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from noise import pnoise2
from pyrr import Quaternion, Vector3
from revolve2.core.physics import Terrain
from revolve2.core.physics.running import geometry

# adding -- Qinwan

from revolve2.core.physics import TargetObject


def targetA(position: Vector3 = Vector3(), size: Vector3 = Vector3([0.1, 0.1, 0.1]), color: Vector3 = Vector3([1.0, 0, 0])) -> TargetObject:

    return TargetObject(
        static_geometry=[
            geometry.Cube(
                position=position, # changed from Vector3()
                #position=Vector3(5, 5, size.z / 2.0 ),
                orientation=Quaternion(),
                size=size,
                color=color
            )
        ]
    )

