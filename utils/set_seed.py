import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    设置随机种子
    torch.backends.cudnn.benchmark：
        当设置为True时，PyTorch会在每次运行时自动寻找最适合当前硬件的卷积实现算法，并进行优化。
        这样可以加速模型的训练和推断过程。然而，由于寻找算法需要额外的计算开销, 因此在输入大小不
        变的情况下，首次运行会比后续运行慢一些。如果输入大小经常变化, 建议将此选项设置为True。
    torch.backends.cudnn.deterministic:
        当设置为True时, PyTorch的卷积操作将以确定性的模式运行，即给定的相同的输入和参数，输出将
        始终相同。由于对结果的可重复性很重要,尤其在进行模型训练和验证时。但由于确定性模式会带来一
        些性能损失, 因此, 在不需要结果可重复的情况下, 可以将些项设置为False
    :param seed: 随机种子
    :return: 无
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)  # 设置CPU随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置GPU随机种子。
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU。
    torch.backends.cudnn.benchmark = False  # GPU优化选项, 自动选择最佳卷积算法以提高性能。
    torch.backends.cudnn.deterministic = True  # GPU优化选项, 确保结果的可重复性
