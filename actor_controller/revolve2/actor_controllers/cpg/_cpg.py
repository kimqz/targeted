#Qinwan

from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt

from revolve2.actor_controller import ActorController
from revolve2.serialization import SerializeError, StaticData

# Qinwan
import math
from pyrr import Quaternion, Matrix33, Vector3
#"


class CpgActorController(ActorController):
    _state: npt.NDArray[np.float_]
    _num_output_neurons: int
    _weight_matrix: npt.NDArray[np.float_]  # nxn matrix matching number of neurons
    _dof_ranges: npt.NDArray[np.float_]

    def __init__(
        self,
        state: npt.NDArray[np.float_],
        num_output_neurons: int,
        weight_matrix: npt.NDArray[np.float_],
        dof_ranges: npt.NDArray[np.float_],
        # Qinwan "
        jointsLeft: List,
        jointsRight: List,
        jointsMid: List
        #"
    ):
        """
        First num_output_neurons will be dof targets
        """
        assert state.ndim == 1
        assert weight_matrix.ndim == 2
        assert weight_matrix.shape[0] == weight_matrix.shape[1]
        assert state.shape[0] == weight_matrix.shape[0]

        self._state = state
        self._num_output_neurons = num_output_neurons
        self._weight_matrix = weight_matrix
        self._dof_ranges = dof_ranges

        # Qinwan "
        self._jointsLeft = jointsLeft
        self._jointsRight = jointsRight
        self._jointsMid = jointsMid

        self.targetAngle = 0
        self.ori = Quaternion()

    def step(self, dt: float) -> None:
        self._state = self._rk45(self._state, self._weight_matrix, dt)


    @staticmethod
    def _rk45(
        state: npt.NDArray[np.float_], A: npt.NDArray[np.float_], dt: float
    ) -> npt.NDArray[np.float_]:
        # TODO The scipy implementation of this function is very slow for some reason.
        # investigate the performance and accuracy differences
        A1: npt.NDArray[np.float_] = np.matmul(
            A, state
        )  # TODO matmul doesn't seem to be properly typed.
        A2: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A1))
        A3: npt.NDArray[np.float_] = np.matmul(A, (state + dt / 2 * A2))
        A4: npt.NDArray[np.float_] = np.matmul(A, (state + dt * A3))
        return state + dt / 6 * (A1 + 2 * (A2 + A3) + A4)

    def get_dof_targets(self) -> List[float]:
        #Qinwan "
        outPuts = self._state[0 : self._num_output_neurons].copy()
        adjustment_factor = abs(self.targetAngle) / math.pi
        reduction_factor = 1 - adjustment_factor

        #current version:
        left_count = len(self._jointsLeft)
        mid_count = len(self._jointsMid)
        right_count = len(self._jointsRight)

        # if only one side has hinges, then ignore
        if left_count + right_count == 0 or right_count + mid_count == 0 or left_count + mid_count == 0:
            pass

        else:
            if self.targetAngle < 0:
                for i in self._jointsLeft:
                    outPuts[i] = outPuts[i] * reduction_factor
            elif self.targetAngle > 0:
                for i in self._jointsRight:
                    outPuts[i] = outPuts[i] * reduction_factor

        return list(
            np.clip(
                outPuts,
                a_min=-self._dof_ranges,
                a_max=self._dof_ranges,
            )
         )

    def serialize(self) -> StaticData:
        return {
            "state": self._state.tolist(),
            "num_output_neurons": self._num_output_neurons,
            "weight_matrix": self._weight_matrix.tolist(),
            "dof_ranges": self._dof_ranges.tolist(),
        }

    @classmethod
    def deserialize(cls, data: StaticData) -> CpgActorController:
        if (
            not type(data) == dict
            or not "state" in data
            or not type(data["state"]) == list
            or not all(type(s) == float for s in data["state"])
            or not "num_output_neurons" in data
            or not type(data["num_output_neurons"]) is int
            or not "weight_matrix" in data
            or not all(
                type(r) == list and all(type(c) == float for c in r)
                for r in data["weight_matrix"]
            )
            or not "dof_ranges" in data
            or not all(type(r) == float for r in data["dof_ranges"])
        ):
            raise SerializeError()

        return CpgActorController(
            np.array(data["state"]),
            data["num_output_neurons"],
            np.array(data["weight_matrix"]),
            np.array(data["dof_ranges"]),
        )

    # Qinwan "
    def passInfo(self,*args):
        actorState = args[0][0]
        tarpos = args[1]
        self.ori = actorState.orientation
        self.pos = actorState.position
        self.tarpos = tarpos

        self.targetAngle = self.calculate_tar_angle()

    def calculate_tar_angle(self):
        # calculate direction vector from robot to target
        direction_vector = np.array([self.tarpos.y - self.pos.y, self.tarpos.x - self.pos.x])

        # convert quaternion to rotation matrix
        self.rotation_matrix = Matrix33.from_quaternion(self.ori)

        # extract robot forward vector from rotation matrix
        robot_forward_vector = np.array([self.rotation_matrix[1, 0], self.rotation_matrix[0, 0]])

        # calculate angle between robot forward vector and direction vector
        angle = math.atan2(direction_vector[0], direction_vector[1]) - math.atan2(robot_forward_vector[0],
                                                                                  robot_forward_vector[1])

        # normalize the angle to the range of -pi to pi
        angle = math.atan2(math.sin(angle), math.cos(angle))

        return (angle)




