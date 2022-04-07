from joblib import Parallel, delayed
import numpy as np
import traj_dist.distance as tdist
from typing import List
from efficient_solver import EfficientSolver
from mapper import Mapper
from model import Trajectory


class Service:
    def __init__(self):
        self.mapper = Mapper()
        self.solver = EfficientSolver()

    def knn_query(self, query_traj: List[List[float, float]], query_type: str, k: int,
                  time_slice=None, jobs_num=22) -> (List[Trajectory], List[float]):
        """
        相似性轨迹检索的总入口
        :param query_traj: List[List[float,float]]:query轨迹,经纬度坐标序列
        :param query_type: str:查询类型,字符串
        :param k: int k近邻
        :param jobs_num: 传统查询调用的CPU核心数
        :param time_slice: List[int,int]: 时间戳切片,如[0,10]表示查询与时间戳0-10有交集的轨迹
        :return: 最相似的k个轨迹的信List[Trajectory],返回的轨迹的points已经是List[List[float,float]]
        """
        sims = []
        top_k_id = []
        # 高效算法
        if query_type == "efficient_bf" or query_type == "efficient_faiss":
            # 根据是否考虑时间切片获取相应的轨迹列表
            if time_slice:
                start_time, end_time = int(time_slice[0]), int(time_slice[1])
                traj_list = self.mapper.get_trajectories_embedding_by_time_slice(start_time, end_time)
            else:
                traj_list = self.mapper.get_all_trajectories_embedding()
            id_list = [traj[0] for traj in traj_list]
            emb_arr = np.array([(eval(traj[1])) for traj in traj_list])
            # 获取query轨迹的embedding
            query_emb = self.solver.embed_trajectory(query_traj)
            if query_type == "efficient_bf":
                # 暴力计算query轨迹与所有轨迹的相似度
                sim_matrix = self.solver.compute_similarity(query_emb, emb_arr)
                sorted_idx = np.argsort(-sim_matrix)
                top_k_id = [id_list[i] for i in sorted_idx[:, :k]]
                sims = sim_matrix[:, sorted_idx].tolist()
            else:
                # TODO: 用faiss查询
                pass

        # 传统算法
        else:
            # 根据是否考虑时间切片获取相应的轨迹列表
            if time_slice:
                start_time, end_time = int(time_slice[0]), int(time_slice[1])
                traj_list = self.mapper.get_trajectories_points_by_time_slice(start_time, end_time)
            else:
                traj_list = self.mapper.get_all_trajectories_points()

            id_list = [traj[0] for traj in traj_list]
            points_list = [(eval(traj[1])) for traj in traj_list]
            qt = np.array(query_traj)
            metric_func = getattr(tdist, query_type)

            # 定义相似度函数
            def cal_sim(tid, qt, traj: np.array):
                if query_type == "lcss":
                    dis = metric_func(qt, traj, eps=0.00029)
                else:
                    dis = metric_func(qt, traj)
                print(f"id:{tid}, dis:{dis}")
                sim = 1 - dis
                return tid, sim

            # 多核并行计算相似度
            res = Parallel(n_jobs=jobs_num)(delayed(cal_sim)(tid, qt, traj) for tid, traj in zip(id_list, points_list))
            top_k_res = sorted(res, key=lambda x: x[1], reverse=True)[:k]
            top_k_id = [tid for tid, sim in top_k_res]
            sims = [sim for tid, sim in top_k_res]
        result_traj_list = self.mapper.get_trajectory_by_id_list(top_k_id)
        for traj in result_traj_list:
            traj.points = eval(traj.points)
        return result_traj_list, sims

    # 根据id返回trajectory
    def get_traj_by_id(self, traj_id) -> Trajectory:
        traj = self.mapper.get_trajectory_by_id(traj_id)
        if not traj:
            return Trajectory()
        return traj

    def generate_embedding_all(self):
        """
        为整张表的轨迹(重新)生成embedding
        获取全表轨迹的id, points, 用solver生成embedding,再update回去
        """
        traj_list = self.mapper.get_all_trajectories_points()
        points_list = [traj[1] for traj in traj_list]
        id_list = [traj[0] for traj in traj_list]
        embeddings = self.solver.embed_trajectory_batch(points_list)
        embeddings = [str(embedding.tolist()) for embedding in embeddings]
        for i, tid in enumerate(id_list):
            self.mapper.update_trajectory_embedding(tid, embeddings[i])

    # TODO 向数据库中插入一批新轨迹并生成其embedding
    def insert_trajectories(self, trajectories):
        """
        向数据库中插入一批新轨迹并生成其embedding
        :param trajectories: List[List[float,float]]
        :return:
        """
        pass
