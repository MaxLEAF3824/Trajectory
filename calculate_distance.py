import json
import traj_dist.distance as tdist
from logging import raiseExceptions
import numpy as np
import utils
from joblib import Parallel, delayed


def calculate_distance(json_file, metric="edr"):
    timer = utils.Timer()
    json_obj = json.load(open(json_file))
    origin_trajs = json_obj["origin_traj"]
    print(f"{len(origin_trajs)} trajectories")
    timer.tik()
    dict_saved = {}
    dis_matrix = np.zeros((len(origin_trajs), len(origin_trajs)))
    idx2key = np.array(list(origin_trajs.keys()))
    if metric == "edr":
        def cal_dis(i, j):
            dis = tdist.edr(np.array(origin_trajs[idx2key[i]]), np.array(origin_trajs[idx2key[j]]), type_d="spherical",
                            eps=21.11)
            if i == j + 1:
                print(i)
            return i, j, dis

        res = []
        res = Parallel(n_jobs=6)(delayed(cal_dis)(i, j) for i in range(len(origin_trajs)) for j in range(i))
        # for i in range(len(origin_trajs)):
        #     for j in range(i):
        #         res.append(cal_dis(i, j))
        timer.tok("calculate distance")
        for (i, j, dis) in res:
            dis_matrix[i, j] = dis
            dis_matrix[j, i] = dis
        timer.tok("save distance")
    else:
        raiseExceptions("metric {} is not supported".format(metric))
    sorted_index = np.argsort(dis_matrix, axis=1)
    dict_saved["sorted_index"] = sorted_index.tolist()
    dict_saved["dis_matrix"] = dis_matrix.tolist()
    dict_saved['idx2key'] = idx2key.tolist()
    json.dump(dict_saved, open(json_file.replace(".json", "_distance.json"), "w"))


if __name__ == "__main__":
    calculate_distance("data/100k_gps_20161101_reformat.json")
