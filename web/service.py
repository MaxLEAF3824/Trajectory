import os
from joblib import Parallel, delayed
import numpy as np
import random
import traj_dist.distance as tdist
from mapper import Mapper
from model import Trajectory, OriginalTrajectoryPoint
import json


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

    # edr算法
    def query_edr(self, query_traj):
        """
        :param query_traj: list of list
        :return: List[Trajectory]
        """

        # query并格式化所有轨迹
        all_trajs = self.mapper.get_trajectories_all()
        all_trajs = [(traj.id, np.array(eval(traj.spherical_points))) for traj in all_trajs]
        print("get all done")

        # 查询结果
        query_traj = np.array(query_traj)

        def cal_sim(traj_id, traj: np.array):
            dis = tdist.edr(query_traj, traj, eps=0.000029125044420566987)
            print("traj_id:", traj_id, "sim:", 1 - dis)
            return {"id": str(traj_id), "data": traj.tolist(), "sim": dis}

        result = Parallel(n_jobs=6)(delayed(cal_sim)(traj_id, traj) for traj_id, traj in all_trajs)
        print("cal sim done")

        # 排序
        result.sort(key=lambda x: x["sim"], reverse=True)
        return result[:10]

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
