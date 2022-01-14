import argparse
from cell2vec import train_cell2vec

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument('--dict_file', '-f', type=str, default="data/str_cell2idx_800.json")

parser.add_argument('--window_size', '-w', type=int, default=20)

parser.add_argument('--embedding_size', '-emb', type=int, default=256)

parser.add_argument('--batch_size', '-b', type=int, default=256)

parser.add_argument('--epoch_num', '-epoch', type=int, default=10)

parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2)

parser.add_argument('--checkpoint', '-cp', type=str, default=None)

parser.add_argument('--pretrained', '-pre', type=str, default=None)

parser.add_argument('--visdom', '-vis', type=int, default=1)

args = parser.parse_args()

print(args)

train_cell2vec(args.dict_file, args.window_size, args.embedding_size, args.batch_size, args.epoch_num,
               args.learning_rate, args.checkpoint, args.pretrained, args.visdom)
