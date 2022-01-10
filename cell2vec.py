import json
import args
import utils
from traj2cell import Traj2Cell
from joblib import Parallel, delayed
from copy import copy
import torch
import numpy as np
from sklearn.neighbors import KDTree
import torch.nn as nn
import torch.optim as optimizer
import torch.utils.data as Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.FloatTensor
window_size = 5
batch_size = 256


def make_data(skip_grams, vocab_size):
    input_data = []
    output_data = []
    for ce, co in skip_grams:
        one_hot = np.zeros(vocab_size)
        one_hot[ce] = 1
        input_data.append(one_hot)
        output_data.append(co)
    return np.array(input_data), np.array(output_data)


if __name__ == '__main__':
    timer = utils.Timer()
    timer.tik("read")
    with open('data/str_cell2idx_400.json') as f:
        str_cell2idx = json.load(f)
        f.close()
    cell2idx = {eval(c): str_cell2idx[c] for c in list(str_cell2idx)}
    timer.tok()
    print(len(cell2idx))

    t2c = Traj2Cell(args.row_num, args.column_num, args.min_lon, args.min_lat, args.max_lon, args.max_lat)
    skip_grams = []

    timer.tik("skip_gram")
    tree = KDTree(list(cell2idx), leaf_size=2)
    _, index = tree.query(list(cell2idx), k=window_size + 1)
    skip_grams.extend([[index[i, 0], index[i, j]] for i in range(index.shape[0]) for j in range(1, index.shape[1])])
    timer.tok()

    timer.tik("make data")
    input_data, output_data = make_data(skip_grams, len(cell2idx))
    timer.tok()

    timer.tik("to tensor")
    input_data, output_data = torch.tensor(input_data), torch.LongTensor(output_data)
    timer.tok()

    dataset = Data.TensorDataset(input_data, output_data)
    loader = Data.DataLoader(dataset, batch_size, shuffle=True)
