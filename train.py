import argparse
from cell2vec import train_cell2vec

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument('--file', '-f', default="data/str_cell2idx_800.json", help="file str_cell2idx.json")

parser.add_argument('--window_size', '-w', default=20, help='cell window size')

parser.add_argument('--embedding_size', '-emb', default=256, help='embedding size')

parser.add_argument('--batch_size', '-b', default=256, help='cell window size')

parser.add_argument('--epoch_num', '-epoch', default=10, help='epoch num')

parser.add_argument('--learning_rate', '-lr', default=1e-2, help='learning rate')

parser.add_argument('--checkpoint', '-cp', default=None)

parser.add_argument('--pretrained', '-pre', default=None)

args = parser.parse_args()

print(args)

train_cell2vec(args.file, args.window_size, args.embedding_size, (args.batch_size), int(args.epoch_num),
               args.learning_rate,
               args.checkpoint, args.pretrained)
