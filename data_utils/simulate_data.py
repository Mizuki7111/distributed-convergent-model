from typing import Optional, Union, Callable

import numpy as np
import networkx as nx
from numpy import ndarray, random as nr


def distance_angular_graph(
    pos: ndarray, vel: ndarray, dist: float, angle: Optional[float] = None
):
    """
    Create a graph based on the distance and angle between the nodes.
    Remark: The graph may be directed if the angle is not None.
    The angle is None for the experiment in the paper."""
    relative_pos = pos[:, None] - pos[None, :]
    pos_norm = np.linalg.norm(relative_pos, axis=-1)
    if type(angle) is float:
        if (angle >= np.pi) | (angle <= 0):
            angle = None
    if angle is None:
        out = np.where(pos_norm <= dist)
    else:
        vel_norm = np.linalg.norm(vel, axis=-1)
        cos = np.divide(
            np.sum(relative_pos * vel[:, None], axis=-1),
            pos_norm * vel_norm[:, None],
            where=vel_norm[:, None] != 0,
        )
        out = np.where((pos_norm <= dist) & (cos >= np.cos(angle)))
    return np.stack((out[0][out[0] != out[1]], out[1][out[0] != out[1]]))


def rotate_2d(theta: float):
    """
    Generate a 2D rotation matrix."""
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rotate_3d(theta: float, phi: float, psi: float):
    """
    Generate a 3D rotation matrix."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    return np.array(
        [
            [
                cos_phi * cos_psi - sin_phi * cos_theta * sin_psi,
                sin_phi * cos_psi + cos_phi * cos_theta * sin_psi,
                sin_theta * sin_psi,
            ],
            [
                -cos_phi * sin_psi - sin_phi * cos_theta * cos_psi,
                -sin_phi * sin_psi + cos_phi * cos_theta * cos_psi,
                sin_theta * cos_psi,
            ],
            [sin_phi * sin_theta, -cos_phi * sin_theta, cos_theta],
        ]
    )


def random_graph(n: int, p: float):
    """
    Generate a static graph randomly."""
    random_graph = nx.gnp_random_graph(n, p)
    e = np.concatenate(
        (np.array(random_graph.edges()).T, np.array(random_graph.edges()).T[::-1]),
        axis=1,
    )
    return e


class Simulator:
    def __init__(self) -> None:
        pass

    def __call__(
        self, steps: int, x0: ndarray, dt: float = 0.01, *args, **kwds
    ) -> ndarray:
        return self.simulate(steps, x0, dt, *args, **kwds)

    def simulate(self, steps: int, x0: ndarray, dt: float, *args, **kwds) -> ndarray:
        """
        Simulate the system for a given number of steps.
        """
        shape = x0.shape
        result = np.zeros((steps, *shape))
        transform = kwds["transform"] if "transform" in kwds else lambda x: x
        result[0] = transform(x0.copy())
        for stp in range(steps - 1):
            result[stp + 1] = transform(
                result[stp] + self.step(result[stp], stp, *args, **kwds) * dt
            ).copy()
        return result

    def step(self, x: ndarray, stp: int, *args, **kwds):
        """
        update the state of the system from the current state.
        """
        return x


class AggregationStaticGraph(Simulator):
    """
    A simulator for the aggregation model with a static graph."""

    def __init__(
        self, coefficient: float = 1.0, conv_range: list = [-2.0, 2.0]
    ) -> None:
        super().__init__()
        self.coefficient = coefficient
        self.conv_range = conv_range

    def step(self, x: ndarray, stp: float, stable_graph: ndarray):
        relative_pos = x[stable_graph[1]] - x[stable_graph[0]]
        vel = np.zeros_like(x)
        np.add.at(vel, stable_graph[0], relative_pos)
        vel = self.coefficient * vel.copy()
        conve_neg_mask = x < self.conv_range[0]
        vel[conve_neg_mask] = vel[conve_neg_mask] - (
            x[conve_neg_mask] - self.conv_range[0]
        )
        conve_pos_mask = x > self.conv_range[1]
        vel[conve_pos_mask] = vel[conve_pos_mask] - (
            x[conve_pos_mask] - self.conv_range[1]
        )
        return vel


class BoidSimulation(Simulator):
    """
    A simulator for the boid model."""

    def __init__(
        self,
        a_align: float,
        a_cohesion: float,
        a_separation: float,
        d_align: float,
        d_cohesion: float,
        d_separation: float,
        angle_align: Optional[float] = None,
        angle_cohesion: Optional[float] = None,
        angle_separation: Optional[float] = None,
        forces: list[Callable] = [],
        vel_range: list[float] = [0.0, np.inf],
        *args,
        **kwargs
    ):
        super().__init__()
        self.a_align = a_align
        self.a_cohesion = a_cohesion
        self.a_separation = a_separation
        self.d_align = d_align
        self.d_cohesion = d_cohesion
        self.d_separation = d_separation
        self.angle_align = angle_align
        self.angle_cohesion = angle_cohesion
        self.angle_separation = angle_separation
        self.forces = forces
        self.vel_range = vel_range

    def step(self, x: ndarray, stp: int, *args, **kwds):
        x_dot = self.boid_step(x, stp)
        for f in self.forces:
            force = f(x, stp)
            x_dot += np.concatenate((np.zeros_like(force), force), axis=-1)
        return x_dot

    def boid_step(self, x: ndarray, _: int):
        dim = x.shape[1] // 2
        pos, vel = x[:, :dim], x[:, dim:]
        if self.vel_range is not None:
            vel_norm = np.linalg.norm(vel, axis=-1, keepdims=True)
            vel_dir = np.divide(vel, vel_norm, where=vel_norm != 0)
            vel = vel_dir * np.clip(vel_norm, *self.vel_range)
        g_align = distance_angular_graph(pos, vel, self.d_align, self.angle_align)
        g_cohesion = distance_angular_graph(
            pos, vel, self.d_cohesion, self.angle_cohesion
        )
        g_separation = distance_angular_graph(
            pos, vel, self.d_separation, self.angle_separation
        )
        align = self.average_fuctor(vel, g_align)
        cohesion = self.average_fuctor(pos, g_cohesion)
        separation = -self.inverse_fuctor(pos, g_separation)
        acc = (
            self.a_align * align
            + self.a_cohesion * cohesion
            + self.a_separation * separation
        )
        return np.concatenate((vel, acc), axis=-1)

    @staticmethod
    def average_fuctor(z, graph):
        row, col = graph
        relative = z[col] - z[row]
        degree = np.zeros((len(z), 1))
        np.add.at(degree, row, 1)
        out = np.zeros_like(z)
        np.add.at(out, row, relative)
        out = out.copy()
        out = np.divide(out, degree, where=degree != 0)
        return out

    @staticmethod
    def inverse_fuctor(z, graph):
        row, col = graph
        relative = z[col] - z[row]
        norm2 = np.sum(relative**2, axis=-1, keepdims=True)
        inv = np.divide(relative, norm2, where=norm2 != 0)
        out = np.zeros_like(z)
        np.add.at(out, row, inv)
        return out.copy()


class RectangleBarrierForce:
    def __init__(
        self, bounds: ndarray, dist: float, epsilon: float = 1e-3, *args, **kwargs
    ):
        self.bounds = bounds
        self.dist = dist
        self.epsilon = epsilon
        self.dim = len(bounds)

    def __call__(self, x: ndarray, _: int, *args, **kwds):
        pos = x[:, : self.dim]
        force = np.zeros_like(pos)
        min_dist = pos - self.bounds[None, :, 0]
        min_mask = (min_dist < self.dist) & (min_dist > 0)
        force[min_mask] = self.epsilon * (1 / min_dist[min_mask])
        max_dist = self.bounds[None, :, 1] - pos
        max_mask = (max_dist < self.dist) & (max_dist > 0)
        force[max_mask] = -self.epsilon * (1 / max_dist[max_mask])
        return force


class RectangleInverseBarrierForce:
    def __init__(self, bounds: ndarray, epsilon: float = 1e-3):
        self.bounds = bounds
        self.dim = len(bounds)
        self.epsilon = epsilon

    def __call__(self, x: ndarray, _: int) -> ndarray:
        pos = x[:, : self.dim]
        inv = 1 / (pos - self.bounds[:, 0]) ** 2
        inv -= 1 / (self.bounds[:, 1] - pos) ** 2
        return self.epsilon * inv
