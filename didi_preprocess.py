from lib2to3.pytree import convert
import pandas as pd
from utils import Timer
from joblib import Parallel, delayed
from traj2grid import Traj2Grid
import json
import random

timer = Timer()


def data_reformat(file_path, row_num=400, column_num=400,dict_path="/home/yqguo/coding/Trajectory/data/str_grid2idx_400.json",max_len=512):
    # read data
    timer.tik()
    df = pd.DataFrame(pd.read_csv(file_path, header=None))
    df.columns = ["name", "order_id", "time", "lon", "lat"]  # lon经度 lat纬度
    timer.tok("read {}".format(file_path))

    # load dict
    with open(dict_path) as f:
        str_grid2idx = json.load(f)
        f.close()
    
    # set converter
    from args import min_lon, min_lat, max_lon, max_lat
    t2g = Traj2Grid(row_num, column_num, min_lon, min_lat, max_lon, max_lat)
    t2g.set_vocab({eval(g): str_grid2idx[g] for g in list(str_grid2idx)})

    def group_concat(name, x: pd.DataFrame):
        traj = []
        for index, row in x.iterrows():
            traj.append([row["lon"], row["lat"]])
    
        # convert to 1d
        traj_1d = t2g.convert1d(traj)
        series = pd.Series(
            {
                "order_id": name,
                "origin_traj":traj,
                "traj": traj_1d,
                "len": len(traj),
                "max_time_diff": x["time"].diff().max(),
                "max_lon_diff": x["lon"].diff().max(),
                "max_lat_diff": x["lat"].diff().max(),
            }
        )
        return series

    def applyParallel(df_groups, func, n=24):
        res = Parallel(n_jobs=n)(delayed(func)(name, group) for name, group in df_groups)
        return pd.DataFrame(res)

    # group-apply
    group_df = applyParallel(df.groupby("order_id"), group_concat)
    timer.tok("group-apply")

    # filter
    t_diff_limit = 20
    lon_lat_diff_limit = 0.005
    f_group = group_df[
        (group_df["max_time_diff"] < t_diff_limit)
        & (group_df["max_lon_diff"] + group_df["max_lat_diff"] < lon_lat_diff_limit)
    ]
    print(f"剩{len(f_group)}/{len(group_df)}条，筛掉{round(100 - 100 * len(f_group) / len(group_df))}%")

    # save
    f_group = f_group[["order_id", "origin_traj","traj"]]
    f_group = f_group.set_index("order_id")
    f_group.to_json(file_path + "_reformat.json")
    timer.tok("save")


data_reformat("/home/yqguo/coding/Trajectory/data/1m_gps_20161101")