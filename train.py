import argparse
from grid2vec import train_grid2vec

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument('--dict_file', '-f', type=str, default="data/str_grid2idx_400.json")

parser.add_argument('--window_size', '-w', type=int, default=20)

parser.add_argument('--embedding_size', '-emb', type=int, default=256)

parser.add_argument('--batch_size', '-b', type=int, default=256)

parser.add_argument('--epoch_num', '-ep', type=int, default=10)

parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)

parser.add_argument('--checkpoint', '-cp', type=str, default=None)

parser.add_argument('--pretrained', '-pre', type=str, default=None)

parser.add_argument('--visdom_port', '-vp', type=int, default=0)


args = parser.parse_args()

print(args)

train_grid2vec(args.dict_file, args.window_size, args.embedding_size, args.batch_size, args.epoch_num,
               args.learning_rate, args.checkpoint, args.pretrained, args.visdom_port)
