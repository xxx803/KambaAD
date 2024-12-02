import os
import pickle
import torch
from src.arguments import build_args
from src.experiments.experiments_main import ExperimentsMain
from utils.out_console import out_console
from src.supports.constants import Constants
from utils.set_seed import set_seed
import warnings

warnings.filterwarnings("ignore")


@out_console.repeat(mark='><', length=80)
def execute():
    args = build_args()
    dataset_info = Constants.get_dataset_info(args.dataset)
    for dataset in dataset_info.datasets:
        for file_name in dataset_info.files:
            args.dataset = dataset
            args.file_name = file_name
            for idx in range(1):
                seed = idx
                args.seed = seed
            #     set_seed(seed)
            #     exp = ExperimentsMain(args)
            #     exp.train(idx)
            #     bs = exp.test()
            #     if os.path.exists(exp.result_name):
            #         with open(exp.result_name, 'rb') as f:
            #             res = pickle.load(f)
            #             f_score = res.f_score
            #     else:
            #         f_score = 0
            #     if f_score < bs.f_score:
            #         f_score = bs.f_score
            #         torch.save(exp.model.state_dict(), exp.checkpoint_name)
            #         bs.seed = seed
            #         with open(exp.result_name, 'wb') as f:
            #             pickle.dump(bs, f)
            #     if float(f_score) > 0.9999:
            #         break

    exp = ExperimentsMain(args)
    exp.print_result()
    out_console.out_line('arguments', mark='--', length=80)
    print('{:<14}:{:>6}\t\t|\t\t{:<14}:{:>6}'.format('patch_len', args.patch_len, 'd_model', args.d_model))
    print('{:<14}:{:>6}\t\t|\t\t{:<14}:{:>6}'.format('stride', args.stride, 'padding_patch', args.padding_patch))
    print('{:<14}:{:>6}\t\t|\t\t{:<14}:{:>6}'.format('head_dropout', args.head_dropout, 'individual', args.individual))
    print('{:<14}:{:>6}\t\t|\t\t{:<14}:{:>6}'.format('d_state', args.d_state, 'd_conv', args.d_conv))
    print('{:<14}:{:>6}\t\t|\t\t{:<14}:{:>6}'.format('expand', args.expand, 'e_layers', args.e_layers))
    print('{:<14}:{:>6}\t\t|\t\t{:<14}:{:>6}'.format('n_heads', args.n_heads, 'd_ff', args.d_ff))
    out_console.out_line('--------', mark='--', length=80)


if __name__ == '__main__':
    execute()
