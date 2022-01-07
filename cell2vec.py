import json
import args
from utils import Traj2Cell

with open('cell2idx.json') as f:
    cell2idx = json.load(f)

t2c = Traj2Cell(args.row_num, args.column_num, args.min_lon, args.min_lat, args.max_lon, args.max_lat)
