import numpy as np
import random
import traj_dist.distance as tdist


# 高效算法
def query_efficient(query_traj):
    res1 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
    res2 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
    result = [{"id": "res1", "data": res1.tolist(), "sim": random.random()},
              {"id": "res2", "data": res2.tolist(), "sim": random.random()}]
    return result


# edr算法
def query_edr(query_traj):
    
    res1 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
    res2 = np.array(query_traj) + np.random.randn(len(query_traj), 2) * 0.001
    result = [{"id": "res1", "data": res1.tolist(), "sim": random.random()},
              {"id": "res2", "data": res2.tolist(), "sim": random.random()}]
    return result

