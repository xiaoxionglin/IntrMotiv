"""L3P-style landmark medoids and directed shortest-path planning."""

from __future__ import annotations

import numpy as np
import torch


def farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or not 0 < count <= len(points):
        raise ValueError("points must be [N,D] and 0 < count <= N")
    chosen = [0]
    distances = np.square(points - points[0]).sum(axis=1)
    while len(chosen) < count:
        index = int(np.argmax(distances))
        chosen.append(index)
        distances = np.minimum(distances, np.square(points - points[index]).sum(axis=1))
    return np.asarray(chosen, dtype=np.int64)


def floyd_warshall_next(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(cost, dtype=np.float64).copy()
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("cost must be square")
    n = distance.shape[0]
    next_hop = np.full((n, n), -1, dtype=np.int64)
    rows, cols = np.where(np.isfinite(distance))
    next_hop[rows, cols] = cols
    for k in range(n):
        candidate = distance[:, k, None] + distance[k, None, :]
        better = candidate < distance
        distance[better] = candidate[better]
        proposed = np.broadcast_to(next_hop[:, k, None], (n, n))
        next_hop[better] = proposed[better]
    return distance, next_hop


class LandmarkPlanner:
    def __init__(
        self,
        landmark_count: int = 50,
        candidates: int = 1000,
        neighbors: int = 8,
        local_horizon: float = 16.0,
    ) -> None:
        self.landmark_count = int(landmark_count)
        self.candidates = int(candidates)
        self.neighbors = int(neighbors)
        self.local_horizon = float(local_horizon)
        self.features: np.ndarray | None = None
        self.poses: np.ndarray | None = None
        self.graph_cost: np.ndarray | None = None
        self.rebuild_count = 0
        self.subgoal_queries = 0
        self.landmark_subgoals = 0
        self.finite_edges = 0

    @staticmethod
    def _distances(agent, states: np.ndarray, goals: np.ndarray, device: torch.device, block: int = 4096) -> np.ndarray:
        pairs = [(i, j) for i in range(len(states)) for j in range(len(goals))]
        result = np.empty(len(pairs), dtype=np.float32)
        agent.eval()
        with torch.no_grad():
            for start in range(0, len(pairs), block):
                chunk = pairs[start : start + block]
                s = torch.as_tensor(np.stack([states[i] for i, _ in chunk]), device=device)
                g = torch.as_tensor(np.stack([goals[j] for _, j in chunk]), device=device)
                result[start : start + len(chunk)] = torch.expm1(agent.temporal_distance(s, g)).cpu().numpy()
        return result.reshape(len(states), len(goals))

    def rebuild(self, replay, agent, device: torch.device) -> None:
        features, poses = replay.sample_candidates(self.candidates)
        with torch.no_grad():
            embedding = agent.goal_repr(torch.as_tensor(features, device=device)).cpu().numpy()
        indices = farthest_point_indices(embedding, min(self.landmark_count, len(features)))
        self.features, self.poses = features[indices], poses[indices]
        dense = self._distances(agent, self.features, self.features, device)
        cost = np.full_like(dense, np.inf, dtype=np.float64)
        np.fill_diagonal(cost, 0.0)
        for row in range(len(cost)):
            order = np.argsort(dense[row])
            order = order[order != row][: self.neighbors]
            cost[row, order] = dense[row, order]
        self.graph_cost = cost
        self.rebuild_count += 1
        self.finite_edges = int(np.isfinite(cost).sum() - len(cost))

    def subgoal(self, state: np.ndarray, final_goal: np.ndarray, agent, device: torch.device) -> np.ndarray:
        self.subgoal_queries += 1
        if self.features is None or self.graph_cost is None:
            return final_goal
        n = len(self.features)
        augmented = np.full((n + 2, n + 2), np.inf, dtype=np.float64)
        augmented[:n, :n] = self.graph_cost
        np.fill_diagonal(augmented, 0.0)
        start, goal = n, n + 1
        start_cost = self._distances(agent, state[None], self.features, device)[0]
        for index in np.argsort(start_cost)[: self.neighbors]:
            augmented[start, index] = start_cost[index]
        finish_cost = self._distances(agent, self.features, final_goal[None], device)[:, 0]
        for index in np.argsort(finish_cost)[: self.neighbors]:
            augmented[index, goal] = finish_cost[index]
        direct_cost = float(self._distances(agent, state[None], final_goal[None], device)[0, 0])
        if direct_cost <= self.local_horizon:
            augmented[start, goal] = direct_cost
        _, next_hop = floyd_warshall_next(augmented)
        hop = int(next_hop[start, goal])
        if hop in (-1, goal):
            return final_goal
        self.landmark_subgoals += 1
        return self.features[hop]
