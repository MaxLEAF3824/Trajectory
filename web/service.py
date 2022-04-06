import torch
import torch.nn.utils.rnn as rnn_utils
from joblib import Parallel, delayed
import numpy as np
import random
import traj_dist.distance as tdist
from mapper import Mapper
from model import Trajectory, OriginalTrajectoryPoint
import json
import sys
sys.path.append('../')
from t3s import T3S
from traj2grid import Traj2Grid
from parameters import *

class Service:
    def __init__(self,model_file="../model/model.pth", dict_path="../data/str_grid2idx_400_44612.json"):
        self.mapper = Mapper()
        self.model = torch.load(model_file)
        self.model = self.model.to("cpu")
        self.t2g = Traj2Grid(400,400,min_lon,min_lat,max_lon,max_lat)
        str_grid2idx = json.load(open(dict_path))
        grid2idx = {eval(g): str_grid2idx[g] for g in list(str_grid2idx)}
        self.t2g.set_vocab(grid2idx)
    
    
    # 高效算法
    def query_efficient(self, query_traj, k=10):
        query_traj_1d, query_traj_sp= self.t2g.convert1d(query_traj)
        all_trajs = self.mapper.get_trajectories_all()
        all_id = [traj.id for traj in all_trajs]
        all_trajs_sp = [torch.FloatTensor(eval(traj.spherical_points)) for traj in all_trajs]
        all_trajs_sp_len = [len(traj) for traj in all_trajs_sp]
        all_trajs_1d = [torch.LongTensor(eval(traj.discrete_points)) for traj in all_trajs]
        print("get all done")
        
        v1 = self.model(torch.LongTensor(query_traj_1d).unsqueeze(0), torch.FloatTensor(query_traj_sp).unsqueeze(0), torch.tensor([len(query_traj_sp)],dtype=torch.long))
        all_trajs_1d = rnn_utils.pad_sequence(all_trajs_1d, batch_first=True, padding_value=-1)
        v_all = self.model(torch.LongTensor(all_trajs_1d).unsqueeze(0), torch.FloatTensor(all_trajs_sp).unsqueeze(0), torch.tensor(all_trajs_sp_len,dtype=torch.long).unsqueeze(0))
        
        dis_list = torch.norm(v1.unsqueeze(0) - v_all, dim=2)
        sorted_dis, idxs = torch.sort(dis_list, dim=1)
        
        top_k_trajs = np.array(all_trajs)[idxs[0][:k]].tolist()
        return top_k_trajs

    # 传统算法
    def query_traditional(self, query_traj, query_type, k=10, jobs_num=22):
        """
        :param k: int, k近邻
        :param query_type: str, 查询类型
        :param query_traj: list of list
        :return: Dict{id, data, length}
        """

        # query并格式化所有轨迹
        all_trajs = self.mapper.get_trajectories_all()
        all_trajs = [(traj.id, np.array(eval(traj.spherical_points))) for traj in all_trajs]
        print("get all done")

        # 查询结果
        query_traj = np.array(query_traj)
        if not query_type:
            query_type="discret_frechet"
        metric_func = getattr(tdist, query_type)

        def cal_sim(traj_id, traj: np.array):
            if query_type == "edr" or query_type == "lcss":
                dis = metric_func(query_traj, traj, eps=0.00029)
            elif query_type == "erp":
                dis = metric_func(query_traj, traj, g=np.zeros(2, dtype=float))
            else:
                dis = metric_func(query_traj, traj)
            print(f"traj_id:{traj_id}, dis:{dis}")
            sim = 1 - dis
            return {"id": str(traj_id), "data": traj.tolist(), "length": len(traj), "sim": sim}

        # 并行计算
        result = Parallel(n_jobs=jobs_num)(delayed(cal_sim)(traj_id, traj) for traj_id, traj in all_trajs)
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
