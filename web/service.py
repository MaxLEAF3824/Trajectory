import os
from joblib import Parallel, delayed
import numpy as np
import random
import traj_dist.distance as tdist
from mapper import Mapper
from model import Trajectory, OriginalTrajectoryPoint
import json
import utils


class Service:
    m = 1

    def __init__(self):
        self.mapper = Mapper()

    # 高效算法
    def query_efficient(self, query_traj):
        res1 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
        res2 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
        result = [{"id": "res1", "data": res1.tolist(), "sim": random.random()},
                  {"id": "res2", "data": res2.tolist(), "sim": random.random()}]
        return result

    # 传统算法
    def query_traditional(self, query_traj, query_type="discret_frechet", k=10):
        """
        :param k: int, k近邻
        :param query_type: str, 查询类型
        :param query_traj: list of list
        :return: List[Trajectory]
        """

        # query并格式化所有轨迹
        all_trajs = self.mapper.get_trajectories_all()
        all_trajs = [(traj.id, np.array(eval(traj.spherical_points))) for traj in all_trajs]
        print("get all done")

        # 查询结果
        query_traj = np.array(query_traj)
        metric_func = getattr(tdist, query_type)

        def cal_sim(traj_id, traj: np.array):
            if query_type == "edr" or query_type == "lcss":
                dis = metric_func(query_traj, traj, eps=0.00029)
            elif query_type == "erp":
                dis = metric_func(query_traj, traj, g=np.zeros(2, dtype=float))
            else:
                dis = metric_func(query_traj, traj)
            return {"id": str(traj_id), "data": traj.tolist(), "length": len(traj), "sim": 1 - dis}

        # 并行计算
        result = Parallel(n_jobs=6)(delayed(cal_sim)(traj_id, traj) for traj_id, traj in all_trajs)
        print("cal sim done")

        # 排序
        result.sort(key=lambda x: x["sim"], reverse=True)
        return result[:k]

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
        return self.mapper.insert_trajectories(trajectories)

    def get_traj_by_id(self, traj_id):
        traj = self.mapper.get_trajectory_by_id(traj_id)
        if not traj:
            return None
        return traj
