import os


def trajectoryFormat(original_txt, save_path, limit=-1):
    # ---变量声明---
    original_file = open(original_txt)
    file_path, file_name = os.path.split(original_txt)
    target_txt = os.path.join(save_path, file_name.split(".")[0] + "_format.txt")
    print(target_txt)
    target_file = open(target_txt, 'w', encoding='utf-8')

    line_no = 1
    trajectory_id = 0
    order_id_pre = ""
    trajectory_format = []

    # ---开始循环---
    target_file.write("var trajs = [\n")
    line = original_file.readline()
    while line and trajectory_id != limit:
        line_char_array = line.replace('\n', '').split(',')
        order_id_next = line_char_array[1]
        latitude = float(line_char_array[3])
        longitude = float(line_char_array[4])
        if order_id_pre == order_id_next:
            # 是同一条轨迹
            trajectory_format.append([latitude, longitude])
        else:
            # 写入旧轨迹
            if trajectory_id != 0:  # 排除
                target_file.write(str(trajectory_format).replace(", ", ",") + ",\n")
                print(str(trajectory_id) + "/" + "209423 percentage:" + str(float(trajectory_id / 209423 * 100)))
            # 创建新轨迹
            order_id_pre = order_id_next
            trajectory_id += 1
            trajectory_format = [[latitude, longitude]]
        line = original_file.readline()
        line_no += 1
    target_file.write("]")
    target_file.close()


if __name__ == "__main__":
    txt = "D:\\Max\\桌面\\大四\\相似轨迹查找\\数据\\滴滴数据\\cxwang@mail.xjtu.edu.cn_20161101\\gps_20161101"
    save = "D:\\Max\\桌面\\大四\\相似轨迹查找\\可视化\\"
    li = 1000  # 可以在这里设置最大trajectory数量限制
    trajectoryFormat(txt, save, limit=li)
