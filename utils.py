import time
from math import sqrt
import numpy as np


def copy_big_file(in_name, out_name, line_num):
    count = 0
    fr = open(f"data/{in_name}")
    fw = open(f"data/{out_name}", "w")
    line = fr.readline()
    while line and count <= line_num:
        fw.write(line)
        count += 1
        line = fr.readline()
    fw.close()
    fr.close()
    print('done')


def parallel_do(func, arg_list, job_num=6):
    from multiprocess.pool import Pool
    pool = Pool(job_num)
    pool.map(func, arg_list)
    pool.close()
    pool.join()


class Timer:
    def __init__(self):
        self.start = "tik"
        self.bgt = time.time()

    def tik(self, str="tik"):
        self.bgt = time.time()
        self.start = str
        print(f"{str} start")

    def tok(self, str=""):
        if not str:
            str = self.start
        print(f"{str} done : {round(time.time() - self.bgt, 3)}s after {self.start} start")


if __name__ == "__main__":
    import numpy as np
    from sklearn.neighbors import KDTree

    np.random.seed(0)
    X = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
    tree = KDTree(X, leaf_size=2)
    # ind：最近的3个邻居的索引
    # dist：距离最近的3个邻居
    # [X[2]]:搜索点
    dist, ind = tree.query([X[2], X[3]], k=3)

    print(ind)
    print(dist)
