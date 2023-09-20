import argparse

parser = argparse.ArgumentParser()

# Common arguments
parser.add_argument('--num_features', type=int, default=19, metavar='N', help='number of features (i.e., input dimension)')
parser.add_argument('--dim_emb', default=128, type=int, help='embedding dimension (default: 128)')
parser.add_argument('--drop_emb', default=0.5, type=float, help='embedding layer dropout rate (default: 0.5)')
parser.add_argument('--dim_alpha', default=128, type=int, help='RNN-Alpha hidden size (default: 128)')
parser.add_argument('--dim_beta', default=128, type=int, help='RNN-Beta hidden size (default: 128)')
parser.add_argument('--drop_context', default=0.5, type=float, help='context layer dropout rate (default: 0.5)')

parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float, help='learning rate (default: 1e-2)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=256, type=int, help='mini-batch size for train (default: 256)')
parser.add_argument('-eb', '--eval-batch-size', type=int, default=256, help='mini-batch size for eval (default: 256)')

parser.add_argument('--no-cuda', dest='cuda', action='store_false', help='NOT use cuda')
parser.add_argument('--no-plot', dest='plot', action='store_false', help='no plot')
parser.add_argument('--threads', type=int, default=16, help='number of threads for data loader to use (default: 16)')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. (default: 1)')

parser.add_argument('--save', default='./Save', type=str, metavar='SAVE_PATH', help='path to save results (default: ./Save/)')
parser.add_argument('--resume', default='', type=str, metavar='LOAD_PATH', help='path to latest checkpoint (default: None)')

# Additional arguments for each specific file
# Train
parser.add_argument('--data_path', metavar='DATA_PATH', default="../data/MIMIC-Numeric-Mortality/DATA_M_PAD_NORM.pkl", help="Path to the dataset")

# Attack
parser.add_argument('--attacktype', type=int, default=3, help='attack type: original:1, kl:2, confident:3')
parser.add_argument('--gamma', type=float, default=0.5, help='cost strength')


settings = parser.parse_args()
