import time
import sys
import os


def copy_big_file(in_name, line_num, start_line_num=0):
    count = 0
    fr = open(f"data/{in_name}")
    fw = open(f"data/{in_name}_{start_line_num}_{line_num}", "w")
    line = fr.readline()
    for i in range(start_line_num):
        line = fr.readline()
    while line and count <= line_num:
        fw.write(line)
        count += 1
        line = fr.readline()
    fw.close()
    fr.close()
    print('done')


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
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
