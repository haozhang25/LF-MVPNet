import torch
import torch.nn as nn
from .AGNet import AGNet
from .FPNet import FPNet


class Net(nn.Module):
    """
    两级光场重建网络

    架构:
    1. AGNet (Anchor Generation Network): 生成锚点特征和视图特征
    2. FPNet (Feature Propagation Network): 基于锚点特征进行精细重建
    """

    def __init__(self, angRes=5,
                 ag_channels=16, ag_n_group=4, ag_n_block=4, ag_hidden_channel=128,
                 fp_channels=64, fp_n_group=2, fp_n_block=3, fp_feats_channel=128,
                 disp_range=[-1, 1], n=5):
        super(Net, self).__init__()

        # 第一阶段：锚点生成
        self.anchor_generation = AGNet(
            angRes_in=angRes,
            channels=ag_channels,
            n_group=ag_n_group,
            n_block=ag_n_block,
            hidden_channel=ag_hidden_channel,
            disp_range=disp_range,
            n=n
        )

        # 第二阶段：特征传播与精细重建
        self.feature_propagation = FPNet(
            angRes=angRes,
            channels=fp_channels,
            n_group=fp_n_group,
            n_block=fp_n_block,
            feats_channel=fp_feats_channel,
            disp_range=disp_range,
            n=n
        )

    def forward(self, x, gt=None, flag=-1):
        """
        Args:
            x: 输入光场数据 [b, u, v, c, h, w]
            gt: 真实标签（未使用，保留接口兼容）
            flag: 标志位（未使用，保留接口兼容）

        Returns:
            dict: {
                'center_out': AGNet输出的中心视图,
                'LF_out': FPNet输出的完整光场
            }
        """
        # 第一阶段：生成锚点特征
        anchor_out = self.anchor_generation(x)

        # 第二阶段：特征传播与重建
        LF_out = self.feature_propagation(x, anchor_out['feats'])

        return {
            "center_out": anchor_out['out'],
            "LF_out": LF_out
        }


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    net = Net().to(device)

    from thop import profile

    input = torch.randn(1, 5, 5, 3, 64, 64).to(device)
    flops, params = profile(net, inputs=(input,))

    print('Number of parameters: %.2fM' % (params / 1e6))
    print('Number of FLOPs: %.2fG' % (flops / 1e9))

    out1 = net(input)
    print('Center output shape:', out1['center_out'].shape)
    print('LF output shape:', out1['LF_out'].shape)

    import time

    # Warm up
    with torch.no_grad():
        for _ in range(5):
            net(input)

    # Measure inference time
    iterations = 10
    with torch.no_grad():
        start = time.time()
        for i in range(iterations):
            net(input)
        end = time.time()

    avg_time = (end - start) / iterations
    print('Average inference time: %.4f s' % avg_time)

