import utils
from geopy.distance import geodesic as dis
import numpy as np


class Traj2Grid:
    def __init__(self, m, n, min_lon, min_lat, max_lon, max_lat):
        self.grid2idx = {}
        self.row_num = m  # 行数
        self.column_num = n  # 列数
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat
        self.h = (max_lat - min_lat) / m  # 高度（纬度）步长
        self.l = (max_lon - min_lon) / n  # 长度（经度）步长
        p0 = (min_lat, min_lon)
        p1 = (min_lat + (max_lat - min_lat) / m, min_lon)
        p2 = (min_lat, min_lon + (max_lon - min_lon) / n)
        self.gird_shape = (dis(p0, p1).meters, dis(p0, p2).meters)

    def convert2d_point(self, point):
        # point : (lon, lat)
        return (
            int((point[0] - self.min_lon) // self.l),
            int((point[1] - self.min_lat) // self.h),
        )

    def build_vocab(self, grid_count: dict, lower_bound=1):
        self.grid2idx.clear()
        for idx, grid in enumerate(grid_count):
            if grid_count[grid] >= lower_bound:
                self.grid2idx[grid] = idx
        return self.grid2idx

    def set_vocab(self, grid2idx):
        self.grid2idx = grid2idx

    def convert1d(self, traj):
        traj_1d = []
        if not self.grid2idx:
            print("vocab is not built.")
            return traj_1d
        for p in traj:
            grid = self.convert2d_point(p)
            if self.grid2idx.get(grid):
                traj_1d.append(self.grid2idx[grid])
        return traj_1d

    def draw_grid(self, grid_count: dict, file_name="grids.png"):
        from PIL import Image

        img = Image.new("RGB", (self.row_num, self.column_num))
        mean = np.mean(list(grid_count.values()))
        std = np.std(list(grid_count.values()))
        for grid in self.grid2idx:
            percent = 50 * (grid_count[grid] - mean) / std + 50
            if percent < 50:
                green = 255
                red = percent * 5.12
            else:
                red = 255
                green = 256 - (percent - 50) * 5.12
            color = (int(red), int(green), 0, 100)
            img.putpixel((grid[0], grid[1]), color)
        img = img.resize((800, 800))
        img.save(file_name)


def generate_grid2idx(row_num=400, column_num=400, data_dir="data/full"):
    import modin.pandas as pd
    import ray
    import json
    from args import min_lon, min_lat, max_lon, max_lat

    ray.init()
    timer = utils.Timer()
    

    t2g = Traj2Grid(row_num, column_num, min_lon, min_lat, max_lon, max_lat)
    print(t2g.gird_shape)
    timer.tik()
    value_counts = None
    for i in range(1, 31):
        df = pd.read_csv(
            f"{data_dir}/gps_201611{str(i).zfill(2)}",
            names=["name", "order_id", "time", "lon", "lat"],
            usecols=["lon", "lat"],
        )  # lon经度 lat纬度
        timer.tok(f"read{str(i).zfill(2)}")
        df = df.apply(t2g.convert2d_point, axis=1).squeeze()
        timer.tok(f"apply{str(i).zfill(2)}")
        if value_counts is not None:
            value_counts = value_counts.add(df.value_counts(), fill_value=0)
        else:
            value_counts = df.value_counts()
        timer.tok(f"value_counts{str(i).zfill(2)}")
    value_counts = value_counts.to_dict()
    grid2idx = t2g.build_vocab(value_counts)
    timer.tok("build_vocab")
    str_grid2idx = {f"({grid[0]},{grid[1]})": grid2idx[grid] for grid in grid2idx}
    json.dump(str_grid2idx, open(f"data/str_grid2idx_{row_num}.json", "w"))


if __name__ == "__main__":
    generate_grid2idx(400, 400)
