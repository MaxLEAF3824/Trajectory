import os

import numpy as np
import random
import traj_dist.distance as tdist
from mapper import Mapper
from model import Trajectory, OriginalTrajectoryPoint
import json


class Service:
    def __init__(self):
        self.mapper = Mapper()

    # 高效算法
    def query_efficient(self, query_traj):
        res1 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
        res2 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
        result = [{"id": "res1", "data": res1.tolist(), "sim": random.random()},
                  {"id": "res2", "data": res2.tolist(), "sim": random.random()}]
        return result

    # edr算法
    def query_edr(self, query_traj):
        '''
        :param query_traj: list of list
        :return: list(dict("id": int, "data": list, "sim": float))
        '''
        res1 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
        res2 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
        result = [{"id": "res1", "data": res1.tolist(), "sim": random.random()},
                  {"id": "res2", "data": res2.tolist(), "sim": random.random()}]
        return result

    def insert_trajectories(self, dataset_json_file):
        dataset = json.load(open(dataset_json_file))
        if not dataset.get("origin_trajs") or not dataset.get("trajs"):
            return 1  # 数据格式错误
        trajectories = []
        for idx, traj in enumerate(dataset["trajs"]):
            trajectories.append(Trajectory(
                id=None,
                length=len(traj),
                discrete_points=str(traj),
                spherical_points=str(dataset['origin_trajs'][idx]),
            ))
        return self.mapper.insert_trajectorys(trajectories)
