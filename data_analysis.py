import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
file_path = r'data/gps_20161101'

if __name__ == "__main__":
    lat_diff = []
    lon_diff = []
    abs_lat_lon_diff = []
    f = open(file_path)
    line = f.readline()
    line_char_array = line.replace('\n', '').split(',')
    order_id_pre = line_char_array[1]
    last_lat = float(line_char_array[3])
    last_lon = float(line_char_array[4])
    count = 0
    line = f.readline()
    while line and count < 1000:
        line_char_array = line.replace('\n', '').split(',')
        order_id_next = line_char_array[1]
        lat = float(line_char_array[3])
        lon = float(line_char_array[4])

        if order_id_pre == order_id_next:
            lat_diff.append(abs(lat - last_lat))
            lon_diff.append(abs(lon - last_lon))
            abs_lat_lon_diff.append(abs(lon - last_lon) + abs(lat - last_lat))
        else:
            order_id_pre = order_id_next
            count += 1
            print(count)
        last_lon = lon
        last_lat = lat
        line = f.readline()
    # plt.hist(lat_diff, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.hist(lon_diff, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    list.sort(abs_lat_lon_diff,reverse=True)
    print(abs_lat_lon_diff[:1000])
    plt.hist(abs_lat_lon_diff, bins=1000)
    plt.xlabel("区间")
    # 显示纵轴标签
    plt.ylabel("频数/频率")
    # 显示图标题
    plt.title("频数/频率分布直方图")
    plt.show()
