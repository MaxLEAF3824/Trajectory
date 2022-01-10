import utils
from geopy.distance import geodesic as dis
from math import ceil
import numpy as np
import pandas as pd


# import modin.pandas as pd
# import ray
# ray.init()

class Traj2Cell:

    def __init__(self, m, n, min_lon, min_lat, max_lon, max_lat):
        self.cell_count = {}
        self.cell2idx = {}
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
        self.cell_shape = (dis(p0, p1).meters, dis(p0, p2).meters)

    def convert2d(self, traj):
        # point : (lon, lat)
        return [(ceil((p[0] - self.min_lon) / self.l), ceil((p[1] - self.min_lat) / self.h)) for p in traj]

    def load_trajs(self, trajs):
        # traj = np.array(trajs).reshape(-1, 2).tolist()
        for cell in self.convert2d(trajs):
            if not self.cell_count.get(cell):
                self.cell_count[cell] = 1
            else:
                self.cell_count[cell] += 1

    def build_vocab(self, lower_bound=0):
        cell2idx = {}
        for idx, cell in enumerate(self.cell_count):
            if self.cell_count[cell] >= lower_bound:
                cell2idx[cell] = idx
        self.cell2idx = cell2idx
        return cell2idx

    def convert1d(self, traj):
        traj_1d = []
        if not self.cell2idx:
            print("vocab is not built.")
            return []
        for cell in self.convert2d(traj):
            if self.cell2idx.get(cell):
                traj_1d.append(self.cell2idx[cell])
            else:
                nearest_cell = self.find_nearest_cell(cell)
                traj_1d.append(self.cell2idx[nearest_cell])
        return traj_1d

    def find_nearest_cell(self, cell):
        cell_list = list(self.cell2idx)
        nearest_cell = cell_list[0]
        min_dis = abs(nearest_cell[0] - cell[0]) + abs(nearest_cell[1] - cell[1])
        for c in cell_list:
            new_dis = abs(c[0] - cell[0]) + abs(c[1] - cell[1])
            if new_dis < min_dis:
                min_dis = new_dis
                nearest_cell = c
        return nearest_cell

    def dis_2d(self, c1, c2):
        return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

    def draw_cell(self, file_name="cells.png"):
        from PIL import Image

        img = Image.new("RGB", (self.row_num, self.column_num))
        values = np.array(list(self.cell_count.values()))
        mean = np.mean(values)
        std = np.std(values)
        for cell in self.cell2idx:
            percent = 50 * (self.cell_count[cell] - mean) / std + 50
            if percent < 50:
                green = 255
                red = percent * 5.12
            else:
                red = 255
                green = 256 - (percent - 50) * 5.12
            color = (int(red), int(green), 0, 100)
            img.putpixel((cell[0], cell[1]), color)
        img = img.resize((800, 800))
        img.save(file_name)


if __name__ == "__main__":
    from args import min_lon, min_lat, max_lon, max_lat

    timer = utils.Timer()
    m = 800
    n = 800
    t2c = Traj2Cell(m, n, min_lon, min_lat, max_lon, max_lat)
    print(t2c.cell_shape)
    timer.tik()
    for i in range(1, 31):
        str_i = str(i).zfill(2)
        print(str_i)
        df = pd.read_csv(f"data/full/gps_201611{str_i}", header=None)
        df.columns = ['name', 'order_id', 'time', 'lon', 'lat']  # lon经度 lat纬度
        timer.tok("read")
        trajs = df[['lon', 'lat']].values.tolist()
        timer.tok("tolist")
        t2c.load_trajs(trajs)
        timer.tok("load")
    cell2idx = t2c.build_vocab()
    str_cell2idx = {"{}_{}".format(cell[0], cell[1]): idx for idx, cell in enumerate(list(cell2idx))}
    import json

    js = json.dumps(str_cell2idx)
    f = open('str_cell2idx.json', 'w')
    f.write(js)
    f.close()
