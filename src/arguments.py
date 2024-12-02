import argparse


def build_args():
    parser = argparse.ArgumentParser()

    # 1 数据参数
    # parser.add_argument('--source_path', type=str, default='./datasource/sources', help='path of the data')
    parser.add_argument('--processed_path', type=str, default='./datasource', help='processed_path')
    parser.add_argument(
        '--dataset', type=str, default='MSL',
        help="['"
             "SMAP,MSL, SMD, PSM, NIPS"
             "']"
    )
    parser.add_argument("--file_name", type=str, default='', help="dataset channel for data_sources/processed")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--features", type=int, default=127)
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')

    # 2 模型参数
    parser.add_argument(
        "--model_name", type=str,
        default='KambaAD',
        help="[KambaAD]"
    )
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patch_len", type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=4, help='num of encoder layers')
    parser.add_argument("--d_model", type=int, default=512)  #
    parser.add_argument('--stride', type=int, default=4, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--individual', type=int, default=1, help='individual head; True 1 False 0')  # 1
    parser.add_argument("--d_state", type=int, default=64)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--e_fact', type=int, default=1)

    # 3 实验参数
    parser.add_argument("--exp_model", type=str, default='train', choices=['train', 'test'])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", type=str, default='output')
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument('--q', type=float, default=0.4, help='init anomaly probability of spot')
    parser.add_argument('--anomaly_ratio', type=float, default=0.5)
    parser.add_argument("--threshold", type=str, default='threshold1', choices=['threshold1', 'threshold2', 'threshold3'])

    args = parser.parse_args()
    return args
