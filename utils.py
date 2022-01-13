import time
import sys
import os


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
        print(f"{str} done, {round(time.time() - self.bgt, 3)}s after {self.start} start")

    def now(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
