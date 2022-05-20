import argparse
import json
import sys, os


def protein_params(parser):
    parser.add_argument('--seq_type', type=str, default="protein")
    parser.add_argument('--cell_line', type=str, default="HCT-116")
    parser.add_argument('--max_len', type=int, default=600)
    parser.add_argument('--emb_type', type=str, default="onehot")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1230)


    # model-specific parameters
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--head_num', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--layer_num', type=int, default=1)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--lstm_drop', type=float, default=0.0)
    parser.add_argument('--linear_drop', type=float, default=0.0)

    # learning process parameters
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    args.save_path = os.path.join(os.path.pardir, 'saved_model', args.seq_type, args.cell_line)
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)
    return args


def nucleotide_params(parser):
    parser.add_argument('--seq_type', type=str, default="nucleotide")
    parser.add_argument('--cell_line', type=str, default="HCT-116")
    parser.add_argument('--max_len', type=int, default=1800)
    parser.add_argument('--emb_type', type=str, default="onehot")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1230)
    
    # model-specific parameters
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--head_num', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--layer_num', type=int, default=1)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--lstm_drop', type=float, default=0.0)
    parser.add_argument('--linear_drop', type=float, default=0.0)

    # learning process parameters
    parser.add_argument('--epoch_num', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    

    args, _ = parser.parse_known_args()
    args.save_path = os.path.join(os.path.pardir, 'saved_model', args.seq_type, args.cell_line)
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)
    return args



def set_params():
    argv = sys.argv
    seq_type = argv[1]
    parser = argparse.ArgumentParser()
    if seq_type == "protein":
        args = protein_params(parser)
        with open(os.path.join(args.save_path, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

#         args = parser.parse_args()
#         with open(os.path.join(args.save_path, 'commandline_args.txt'), 'w') as f:
#             args.__dict__ = json.load(f)

    elif seq_type == "nucleotide":
        args = nucleotide_params(parser)
        with open(os.path.join(args.save_path, 'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

#         args = parser.parse_args()
#         with open(os.path.join(args.save_path, 'commandline_args.txt'), 'w') as f:
#             args.__dict__ = json.load(f)  
    return args
