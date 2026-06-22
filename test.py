import time
import argparse
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import scipy.io as sio
import time
import imageio
import torchvision
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from dataload import *
from model.LFMVPNet import Net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--angin", type=int, default=7, help="angular resolution")
    parser.add_argument('--testset_dir', type=str, default='/private/Datasets/LFLLE/7x7Dataset/TestData_20_LLE_7x7/')
    parser.add_argument("--patchsize", type=int, default=64, help="LFs are cropped into patches to save GPU memory")
    parser.add_argument('--model_path', type=str,
                        default='./log/LFMVPNet_20_LLE_7x7.tar')
    parser.add_argument('--save_path', type=str, default='./Results')
    parser.add_argument('--model_name', type=str, default='LFMVPNet')
    parser.add_argument('--experiment', type=str, default='20_LLE_7x7')
    return parser.parse_args()


def test(cfg, test_loader):
    cfg.save_path = f"{cfg.save_path}/{cfg.model_name}/{cfg.experiment}/"
    os.makedirs(cfg.save_path, exist_ok=True)

    net = Net(cfg.angin)
    net.to(cfg.device)
    cudnn.benchmark = True

    if os.path.isfile(cfg.model_path):

        device = torch.device("cuda:0")  # Specify the desired CUDA device (0 in this example)
        model = torch.load(cfg.model_path, map_location=device)

        net.load_state_dict(model['state_dict'])

    else:
        print("=> no model found at '{}'".format(cfg.load_model))

    # net = torch.nn.DataParallel(net, device_ids=[0, 1])
    with torch.no_grad():
        psnr_testset = []
        ssim_testset = []
        lpips_testset = []
        psnr_epoch_test, ssim_epoch_test, lpips_epoch_test = inference(test_loader, net, cfg.angin)
        psnr_testset.append(psnr_epoch_test)
        ssim_testset.append(ssim_epoch_test)
        lpips_testset.append(lpips_epoch_test)
        print(time.ctime()[4:-5] + ' Valid----, PSNR---%f, SSIM---%f, LPIPS---%f' % (
        psnr_epoch_test, ssim_epoch_test, lpips_epoch_test))


def inference(test_loader, net, angin):
    psnr_iter_test = []
    ssim_iter_test = []
    lpips_iter_test = []

    for idx_iter, (data, label) in (enumerate(test_loader)):
        data = data.squeeze().to(cfg.device)
        label = label.squeeze()
        sub_lfs = LFdivide(data, cfg.patchsize, cfg.patchsize // 2)
        numU, numV, u, v, c, h, w = sub_lfs.shape
        minibatch = 8
        num_inference = numU * numV // minibatch
        sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) u v c h w')

        with torch.no_grad():
            out_lf = []
            for idx_inference in range(num_inference):
                torch.cuda.empty_cache()
                tmp = sub_lfs[idx_inference * minibatch:(idx_inference + 1) * minibatch, :, :, :, :, :]
                out_tmp = net(tmp.to(cfg.device))['LF_out']
                # out_tmp = net(tmp.to(cfg.device))
                out_lf.append(out_tmp)
            if (numU * numV) % minibatch:
                torch.cuda.empty_cache()
                tmp = sub_lfs[(idx_inference + 1) * minibatch:, :, :, :, :, :]
                out_tmp = net(tmp.to(cfg.device))['LF_out']
                # out_tmp = net(tmp.to(cfg.device))
                out_lf.append(out_tmp)
        out_lfs = torch.cat(out_lf, 0)
        out_lfs = rearrange(out_lfs, '(n1 n2) (u1 u2 c) h w -> n1 n2 u1 u2 c h w', n1=numU, n2=numV, u1=angin, u2=angin)
        outLF = LFintegrate(out_lfs, cfg.patchsize, cfg.patchsize // 2)

        outLF = outLF[:, :, :, 0: data.shape[3], 0: data.shape[4]]
        psnr, ssim, lpips = cal_metrics(label, outLF)
        psnr_iter_test.append(psnr)
        ssim_iter_test.append(ssim)
        lpips_iter_test.append(lpips)

        u, v, c, h, w = outLF.shape
        outLF = outLF.contiguous().view(u * v, c, h, w)
        plot_out_pred = torchvision.utils.make_grid(outLF, nrow=5, padding=0, normalize=False)
        x = np.transpose(plot_out_pred.detach().cpu().numpy(), (1, 2, 0))
        plot_out_pred = (np.clip(x, 0, 1) * 255).astype(np.uint8)

        # # Save images
        imageio.imwrite(f'{cfg.save_path}/{idx_iter}_IMG_PRED.png', plot_out_pred)

    psnr_epoch_test = float(np.array(psnr_iter_test).mean())
    ssim_epoch_test = float(np.array(ssim_iter_test).mean())
    lpips_epoch_test = float(np.array(lpips_iter_test).mean())
    with open(f'{cfg.save_path}{cfg.model_name}_test.txt', 'a') as txtfile:
        log_line = (
            f"{time.ctime()[4:-5]}\n"
            f"PSNR_iter---{np.array2string(np.array(psnr_iter_test), precision=4)}\n"  # 保留4位小数
            f"SSIM_iter---{np.array2string(np.array(ssim_iter_test), precision=4)}\n"
            f"LPIPS_iter---{np.array2string(np.array(lpips_iter_test), precision=4)}\n"
            f"PSNR_epoch---{psnr_epoch_test:.4f}\n"
            f"SSIM_epoch---{ssim_epoch_test:.4f}\n"
            f"LPIPS_epoch---{lpips_epoch_test:.4f}\n"
        )
        txtfile.write(log_line)

    return psnr_epoch_test, ssim_epoch_test, lpips_epoch_test


def main(cfg):
    test_set = DataSetLoader(dataset_dir=cfg.testset_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
    test(cfg, test_loader)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)