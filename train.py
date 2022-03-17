import argparse
from grid2vec import *
from t3s import *
import utils


timer = utils.Timer()

parser = argparse.ArgumentParser(description="train.py")
parser.add_argument('--model', '-m', type=str ,default='t3s', help='model name')
parser.add_argument('--batch_size', '-b', type=int, default=64)
parser.add_argument('--epoch_num', '-ep', type=int, default=100)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2)
parser.add_argument('--visdom_port', '-vp', type=int, default=0)
parser.add_argument('--checkpoint', '-cp', type=str, default=None)

parser.add_argument('--grid2idx', '-dict', type=str, default="data/str_grid2idx_400.json")
parser.add_argument('--train_dataset', '-data_tr', type=str, default="data/1m_gps_20161101_dataset.json")
parser.add_argument('--validate_dataset', '-data_va', type=str, default="data/100k_gps_20161102_dataset.json")
parser.add_argument('--pretrained_embedding', '-pre', type=str, default=None)
parser.add_argument('--embedding_size', '-emb', type=int, default=256)
parser.add_argument('--window_size', '-w', type=int, default=20)

args = parser.parse_args()

print(args)

if args.model == "t3s":
    train_t3s(args)
elif args.model == "grid2vec":
    train_grid2vec(args.grid2idx, args.window_size, args.embedding_size, args.batch_size, args.epoch_num,
               args.learning_rate, args.checkpoint, args.pretrained, args.visdom_port)
else:
    raise ValueError("model must be grid2vec or t3s")
