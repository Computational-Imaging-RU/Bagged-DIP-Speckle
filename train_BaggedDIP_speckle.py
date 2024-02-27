import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import glob
import os
import cv2
import time
import pickle
from skimage.metrics import structural_similarity as ssim

from decoder import autoencodernet
from function_grad import nll_func_avg_batch, nll_grad_mul_block_matrix_batch, nll_grad_mul_block_matrix_pseudo_batch
from utils import gen_latent_code_patch, imread_CS_py, img2col_py, col2im_CS_py, psnr, generate_gaussian_matrix, generate_orthogonal_matrix, num_param

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=124, help='Random seed.')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--NN', type=str, default='Decoder', help='(Deep) Decoder/DnCNN')
parser.add_argument('--need_sigmoid', type=bool, default=True, help='sigmoid as the final layer in Decoder')
parser.add_argument('--decodetype', type=str, default='upsample', help='upsample, transposeconv')
parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of upsampling deep decoder.')
parser.add_argument('--test_name', type=str, default='Set11', help='name of test set')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--downsample', type=bool, default=False, help='downsample the raw image to lower resolution')
parser.add_argument('--crop', type=bool, default=True, help='downsample the raw image to lower resolution')
parser.add_argument('--outer_ite', type=int, default=50, help='Outer iterations for projection: less looks, m/n, more ites')
parser.add_argument('--lr_NN', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--lr_GD', type=float, default=1e-2, help='lr for GD step:5e-3, 1e-2')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--compression_rate', type=float, default=0.125, help='Compression rate for compressed sensing.')
parser.add_argument('--loss_function', type=str, default='MLE', help='MLE, MSE')
parser.add_argument('--L_inf', type=bool, default=False, help='MLE: when L is infinite')
parser.add_argument('--gradient_method', type=str, default='GD', help='GD, CD')
parser.add_argument('--use_autograd_GD', type=bool, default=False, help='Use pytorch autograd for GD step')
parser.add_argument('--use_complex', type=bool, default=False, help='Use complex valued A and w')
parser.add_argument('--degradation', type=str, default='CS', help='Compressed sensing')
parser.add_argument('--A_type', type=str, default='Fourier', help='Gaussian, Fourier')
parser.add_argument('--A_look', type=str, default='A', help='A is constant across L')
parser.add_argument('--diff_A', type=bool, default=False, help='A real and A imag')
parser.add_argument('--X_init', type=str, default='ATy_abs_avg', help='Fourier: ATy_abs_avg, Gaussian: ATy_abs_avg_norm')
parser.add_argument('--multi_look', type=bool, default=True, help='use multi-look, set True even for single look')
parser.add_argument('--num_look', type=int, default=10, help='number of looks')
parser.add_argument('--DIP_avg', type=bool, default=True, help='Average diff patch size DIPs')
parser.add_argument('--DIP_final', type=str, default='DIP_avg', help='DIP_avg, DIP1, DIP2, DiP3')
parser.add_argument('--patch_size', type=int, default=128, help='downsampled/cropped image/patch size of raw image')
parser.add_argument('--Fusion', type=bool, default=False, help='Fusion of xG and xD')
parser.add_argument('--lamb', type=float, default=1.0, help='fixed coeff of penalty term in DIP')
parser.add_argument('--init_opt_ite', type=bool, default=False, help='init the DIP at every ite')
parser.add_argument('--pseudo_inverse', type=bool, default=False, help='NOT use exact inverse')
parser.add_argument('--ite_inverse', type=int, default=1, help='Iterations in Schulz for approximate inverse.')
parser.add_argument('--use_Schulz', type=bool, default=False, help='Use inner Schulz to approximate inverse')
parser.add_argument('--decision', type=str, default='x_delta', help='When to compute exact inverse:singular_value, x_delta')
parser.add_argument('--exact_init', type=bool, default=False, help='Only calculate exact inverse within first certain iterations, if True, set use_Schulz to False')
parser.add_argument('--exact_inv_ite', type=int, default=5, help='The first number of iterations for exact inverse.')
args = parser.parse_args()
print(args)

def train(Phi, num_channel_list, kernel_size_list, filepaths, out_path, device, dtype):

    img_te_num = len(filepaths)
    ########## Save the running logs ##########
    PSNR_GD_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)
    PSNR_NN_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)
    PSNR_Fusion_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)
    PSNR_Fusion_best = np.zeros([img_te_num], dtype=np.float64)
    SSIM_GD_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)
    SSIM_NN_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)
    SSIM_Fusion_All = np.zeros([args.outer_ite, img_te_num], dtype=np.float64)
    SSIM_Fusion_best = np.zeros([img_te_num], dtype=np.float64)

    ########## Loop over every test image ##########
    for img_no in range(img_te_num):
        lr_GD = args.lr_GD
        imgName = filepaths[img_no]
        single_imgName_ = imgName.split(".")[0]
        single_imgName = single_imgName_.split("/")[-1]
        print('image name:', imgName)
        if single_imgName in ['barbara', 'peppers256', 'house', 'foreman', 'boats']:
            inner_ite_list = [200, 300, 400]
        elif single_imgName in ['Parrots', 'Monarch']:
            inner_ite_list = [400, 600, 800]
        elif single_imgName in ['cameraman']:
            inner_ite_list = [1000, 2000, 4000]
        print('inner ites:', inner_ite_list)

        ########## Crop the image into patches ##########
        Img = cv2.imread(imgName, 1)
        if args.crop:
            Img = Img[32:32+args.patch_size, 32:32+args.patch_size]
        elif args.downsample:
            Img = cv2.resize(Img, (args.patch_size, args.patch_size))
        Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
        Img_rec_yuv = Img_yuv.copy()
        Iorg_y = Img_yuv[:, :, 0]

        [Iorg, row, col, Ipad, row_new, col_new] = imread_CS_py(Iorg_y, args.patch_size)
        Icol = img2col_py(Ipad, args.patch_size).transpose() / 255.0
        Img_output = Icol
        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(dtype).to(device)

        ########## Multiply noise and degradation process ##########
        w_noise_list = []
        for ii in range(args.num_look):
            if args.use_complex:
                w_noise_i_ = torch.randn(batch_x.size(dim=0), batch_x.size(dim=1), 2).type(dtype).to(device)
                w_noise_i = torch.complex(w_noise_i_[:, :, 0], w_noise_i_[:, :, 1])
                w_noise_list.append(w_noise_i)
            else:
                w_noise_i = torch.randn(batch_x.size(dim=0), batch_x.size(dim=1)).type(dtype).to(device)
                w_noise_list.append(w_noise_i)
        if args.multi_look:
            batch_xw_0 = torch.mul(batch_x, w_noise_list[0])
            Phix = torch.matmul(batch_xw_0, torch.transpose(Phi, 0, 1)).unsqueeze(-1)
            for look_i in range(int(len(w_noise_list) - 1)):
                batch_xw_i = torch.mul(batch_x, w_noise_list[look_i + 1])
                Phix_i = torch.matmul(batch_xw_i, torch.transpose(Phi, 0, 1)).unsqueeze(-1)
                Phix = torch.cat((Phix, Phix_i), dim=-1)
            Phix = Phix

        else:
            batch_xw = torch.mul(batch_x, w_noise_list[0])
            Phix = torch.mm(batch_xw, torch.transpose(Phi, 0, 1))


        ########## Init the input ATy ##########
        x_ = Phi.T.conj() @ Phix

        ########## Average across looks ##########
        if args.X_init == 'ATy_abs_avg':
            x_new = torch.mean(torch.abs(x_), dim=-1)
        elif args.X_init == 'ATy_abs_avg_norm':
            x_new = torch.mean(torch.abs(x_), dim=-1)
            x_new = x_new / torch.max(x_new)

        ########## Assure input init on device ##########
        if args.use_complex:
            x_new = x_new.to(device)
            X_new = torch.diag_embed(x_new).to(device)

        ########## Outer loop of iterative algorithm ##########
        best_PSNR_Fusion = 0
        for i in range(args.outer_ite):
            flag_i = np.mod(i + 1, 1)
            if flag_i == 0:
                print('-------------------------- outer ite', i + 1, '--------------------------')
            ########## Gradient descent step: x^G ##########
            update_start = time.time()
            if args.gradient_method == 'GD':
                if args.use_autograd_GD: # Use pytorch autograd given the loss function
                    if args.use_complex:
                        X_new = torch.complex(X_new, X_new)
                    X_new = Variable(X_new.to(device), requires_grad=True)
                    optimizer_GD = optim.SGD([X_new], lr=lr_GD, weight_decay=0.0, momentum=0.9, nesterov=args.use_Nesterov)
                    optimizer_GD.zero_grad()
                    loss_GD_batch = nll_func_avg_batch(X_new, Phi, Phix, 1.0)
                    loss_GD = torch.sum(loss_GD_batch)
                    loss_GD.backward()
                    optimizer_GD.step()
                    X_raw = X_new.detach()
                    x_raw = torch.diagonal(X_raw, dim1=-2, dim2=-1)
                else: # write the gradient function explicitly (faster)
                    if args.pseudo_inverse: # Use Newton-Schulz matrix inverse approximation
                        if args.A_look == 'A':
                            if args.exact_init:
                                if i + 1 <= args.exact_inv_ite:
                                    grad_vec, B_inv = nll_grad_mul_block_matrix_batch(x_new, Phi, Phix, batch_x, args.use_complex, args.diff_A, args.L_inf, 1.0)
                                    x_raw = x_new - lr_GD * grad_vec
                                else:
                                    grad_vec, B_inv = nll_grad_mul_block_matrix_pseudo_batch(x_new, Phi, Phix, B_inv, args.ite_inverse, args.use_Schulz, args.use_complex, args.diff_A, device, 1.0)
                                    x_raw = x_new - lr_GD * grad_vec
                            else:
                                if i == 0:
                                    x_delta = 1.0
                                if x_delta > 0.12:
                                    grad_vec, B_inv = nll_grad_mul_block_matrix_batch(x_new, Phi, Phix, batch_x, args.use_complex, args.diff_A, args.L_inf, 1.0)
                                    x_raw = x_new - lr_GD * grad_vec
                                else:
                                    grad_vec, B_inv = nll_grad_mul_block_matrix_pseudo_batch(x_new, Phi, Phix, B_inv, args.ite_inverse, args.use_Schulz, args.use_complex, args.diff_A, device, 1.0)
                                    x_raw = x_new - lr_GD * grad_vec

                    else: # Calculate the exact matrix inverse
                        if args.A_look == 'A':
                            if args.loss_function == 'MLE':
                                grad_vec, B_inv = nll_grad_mul_block_matrix_batch(x_new, Phi, Phix, batch_x, args.use_complex, args.diff_A, args.L_inf, 1.0)
                        x_raw = x_new - lr_GD * grad_vec
                    x_raw = torch.clamp(x_raw, 0.001, 1)
            update_end = time.time()

            ########## calculate the MSE ##########
            if flag_i == 0:
                mse_GD = torch.mean(torch.square(batch_x - x_raw))

            ######### projection step: train DIP/Deep Decoder ##########
            projection_start = time.time()
            x_raw_ = x_raw.view(args.patch_size, args.patch_size)
            x_raw_np = x_raw_.detach().cpu().numpy().copy()
            DIP_patch_size_list = [32,64,128]
            x_gen_psnr_list = []

            DIP_patch_size = DIP_patch_size_list[0]
            num_patch = int((args.patch_size / DIP_patch_size) ** 2)
            [Iorg_raw, row_raw, col_raw, Ipad_raw, row_new_raw, col_new_raw] = imread_CS_py(x_raw_np, DIP_patch_size)
            x_raw_np_patch = img2col_py(Ipad_raw, DIP_patch_size).transpose()
            x_raw_tensor_patch = torch.from_numpy(x_raw_np_patch).type(dtype).to(device)
            x_raw_patch = x_raw_tensor_patch.view(-1, 1, DIP_patch_size, DIP_patch_size)
            x_gen_patch = torch.zeros_like(x_raw_tensor_patch)
            for DIP_i in range(num_patch):
                if args.NN == 'Decoder':
                    output_depth = 1  # number of output channels (gray scale image)
                    net_1 = autoencodernet(num_output_channels=output_depth, num_channels_up=num_channel_list[0],
                                         need_sigmoid=args.need_sigmoid, decodetype=args.decodetype,
                                         kernel_size=kernel_size_list[0]).type(dtype).to(device)
                    latent_code_1 = gen_latent_code_patch(batch_x.size(dim=0), DIP_patch_size, num_channel_list[0], 1).type(dtype).to(device)
                    params = [x for x in net_1.decoder.parameters()]
                optimizer = optim.Adam(params, lr=args.lr_NN, weight_decay=args.weight_decay)
                x_raw_i = x_raw_patch[DIP_i]
                x_raw_i_flatten = x_raw_i.view(-1, DIP_patch_size ** 2)
                for ee in range(inner_ite_list[0]):
                    net_1.train()
                    optimizer.zero_grad()
                    x_gen_tensor_i_ = net_1(latent_code_1)
                    x_gen_tensor_i = x_gen_tensor_i_.view(-1, DIP_patch_size ** 2)
                    loss_train = F.mse_loss(x_gen_tensor_i, x_raw_i_flatten)
                    loss_train.backward()
                    optimizer.step()
                with torch.no_grad():
                    x_gen_img_i_final = net_1(latent_code_1).detach()
                    x_gen_patch[DIP_i] = x_gen_img_i_final.view(-1, DIP_patch_size ** 2)
            x_gen_np = x_gen_patch.detach().cpu().numpy()
            x_gen_np_1 = col2im_CS_py(x_gen_np.transpose(), row_raw, col_raw, row_new_raw, col_new_raw, DIP_patch_size)
            x_gen_np_clip_1 = np.clip(x_gen_np_1, 0, 1)
            x_gen_psnr_1 = psnr(x_gen_np_clip_1 * 255, Iorg.astype(np.float64))
            print('net 1 params:', num_param(net_1), ' psnr:', x_gen_psnr_1)
            x_gen_psnr_list.append(x_gen_psnr_1)
            x_gen_1 = torch.from_numpy(x_gen_np_1).type(dtype).to(device)
            x_gen_1 = x_gen_1.view(-1, args.patch_size ** 2)

            DIP_patch_size = DIP_patch_size_list[1]
            num_patch = int((args.patch_size / DIP_patch_size) ** 2)
            [Iorg_raw, row_raw, col_raw, Ipad_raw, row_new_raw, col_new_raw] = imread_CS_py(x_raw_np, DIP_patch_size)
            x_raw_np_patch = img2col_py(Ipad_raw, DIP_patch_size).transpose()
            x_raw_tensor_patch = torch.from_numpy(x_raw_np_patch).type(dtype).to(device)
            x_raw_patch = x_raw_tensor_patch.view(-1, 1, DIP_patch_size, DIP_patch_size)
            x_gen_patch = torch.zeros_like(x_raw_tensor_patch)
            for DIP_i in range(num_patch):
                if args.NN == 'Decoder':
                    output_depth = 1  # number of output channels (gray scale image)
                    net_2 = autoencodernet(num_output_channels=output_depth, num_channels_up=num_channel_list[1],
                                         need_sigmoid=args.need_sigmoid, decodetype=args.decodetype,
                                         kernel_size=kernel_size_list[1]).type(dtype).to(device)
                    latent_code_2 = gen_latent_code_patch(batch_x.size(dim=0), DIP_patch_size,num_channel_list[1], 1).type(dtype).to(device)
                    params = [x for x in net_2.decoder.parameters()]
                optimizer = optim.Adam(params, lr=args.lr_NN, weight_decay=args.weight_decay)
                x_raw_i = x_raw_patch[DIP_i]
                x_raw_i_flatten = x_raw_i.view(-1, DIP_patch_size ** 2)
                for ee in range(inner_ite_list[1]):
                    net_2.train()
                    optimizer.zero_grad()
                    x_gen_tensor_i_ = net_2(latent_code_2)
                    x_gen_tensor_i = x_gen_tensor_i_.view(-1, DIP_patch_size ** 2)
                    loss_train = F.mse_loss(x_gen_tensor_i, x_raw_i_flatten)
                    loss_train.backward()
                    optimizer.step()
                with torch.no_grad():
                    x_gen_img_i_final = net_2(latent_code_2).detach()
                    x_gen_patch[DIP_i] = x_gen_img_i_final.view(-1, DIP_patch_size ** 2)
            x_gen_np = x_gen_patch.detach().cpu().numpy()
            x_gen_np_2 = col2im_CS_py(x_gen_np.transpose(), row_raw, col_raw, row_new_raw, col_new_raw, DIP_patch_size)
            x_gen_np_clip_2 = np.clip(x_gen_np_2, 0, 1)
            x_gen_psnr_2 = psnr(x_gen_np_clip_2 * 255, Iorg.astype(np.float64))
            print('net 2 params:', num_param(net_2), ' psnr:', x_gen_psnr_2)
            x_gen_psnr_list.append(x_gen_psnr_2)
            x_gen_2 = torch.from_numpy(x_gen_np_2).type(dtype).to(device)
            x_gen_2 = x_gen_2.view(-1, args.patch_size ** 2)

            DIP_patch_size = DIP_patch_size_list[2]
            num_patch = int((args.patch_size / DIP_patch_size) ** 2)
            [Iorg_raw, row_raw, col_raw, Ipad_raw, row_new_raw, col_new_raw] = imread_CS_py(x_raw_np, DIP_patch_size)
            x_raw_np_patch = img2col_py(Ipad_raw, DIP_patch_size).transpose()
            x_raw_tensor_patch = torch.from_numpy(x_raw_np_patch).type(dtype).to(device)
            x_raw_patch = x_raw_tensor_patch.view(-1, 1, DIP_patch_size, DIP_patch_size)
            x_gen_patch = torch.zeros_like(x_raw_tensor_patch)
            for DIP_i in range(num_patch):
                if args.NN == 'Decoder':
                    output_depth = 1  # number of output channels (gray scale image)
                    net_3 = autoencodernet(num_output_channels=output_depth, num_channels_up=num_channel_list[2],
                                         need_sigmoid=args.need_sigmoid, decodetype=args.decodetype,
                                         kernel_size=kernel_size_list[2]).type(dtype).to(device)
                    latent_code_3 = gen_latent_code_patch(batch_x.size(dim=0), DIP_patch_size, num_channel_list[2], 1).type(dtype).to(device)
                    params = [x for x in net_3.decoder.parameters()]
                optimizer = optim.Adam(params, lr=args.lr_NN, weight_decay=args.weight_decay)
                x_raw_i = x_raw_patch[DIP_i]
                x_raw_i_flatten = x_raw_i.view(-1, DIP_patch_size ** 2)
                for ee in range(inner_ite_list[2]):
                    net_3.train()
                    optimizer.zero_grad()
                    x_gen_tensor_i_ = net_3(latent_code_3)
                    x_gen_tensor_i = x_gen_tensor_i_.view(-1, DIP_patch_size ** 2)
                    loss_train = F.mse_loss(x_gen_tensor_i, x_raw_i_flatten)
                    loss_train.backward()
                    optimizer.step()
                with torch.no_grad():
                    x_gen_img_i_final = net_3(latent_code_3).detach()
                    x_gen_patch[DIP_i] = x_gen_img_i_final.view(-1, DIP_patch_size ** 2)
            x_gen_np = x_gen_patch.detach().cpu().numpy()
            x_gen_np_3 = col2im_CS_py(x_gen_np.transpose(), row_raw, col_raw, row_new_raw, col_new_raw, DIP_patch_size)
            x_gen_np_clip_3 = np.clip(x_gen_np_3, 0, 1)
            x_gen_psnr_3 = psnr(x_gen_np_clip_3 * 255, Iorg.astype(np.float64))
            print('net 3 params:', num_param(net_3), ' psnr:', x_gen_psnr_3)
            x_gen_psnr_list.append(x_gen_psnr_3)
            x_gen_3 = torch.from_numpy(x_gen_np_3).type(dtype).to(device)
            x_gen_3 = x_gen_3.view(-1, args.patch_size ** 2)

            if args.DIP_avg:
                x_gen = (x_gen_1 + x_gen_2 + x_gen_3)/3
            else:
                if args.DIP_final == 'DIP1':
                    x_gen = x_gen_1
                elif args.DIP_final == 'DIP2':
                    x_gen = x_gen_2
                if args.DIP_final == 'DIP3':
                    x_gen = x_gen_3

            ########## generate x^P using the trained networks ##########
            with torch.no_grad():
                # net.eval()
                if args.NN == 'Decoder':
                    x_gen = x_gen.detach()
                x_old = x_new
                if flag_i == 0:
                    mse_NN = torch.mean(torch.square(batch_x - x_gen))

                ########## Fusion of x^G, x^D and x^P ##########
                x_new = args.lamb * x_gen + (1-args.lamb) * x_raw

                ########## calculate x delta for matrix inverse decision ########
                x_delta = torch.max(torch.abs(x_new-x_old))

                X_new = torch.diag_embed(x_new)
                projection_end = time.time()

                ########## GD ite PSNR and patches reconstruction ##########
                Prediction_value_GD = x_raw.detach().cpu().numpy()
                X_rec_GD = np.clip(col2im_CS_py(Prediction_value_GD.transpose(), row, col, row_new, col_new, args.patch_size), 0, 1)
                rec_PSNR_GD = psnr(X_rec_GD * 255, Iorg.astype(np.float64))
                rec_SSIM_GD = ssim(X_rec_GD * 255, Iorg.astype(np.float64), data_range=255)
                if flag_i == 0:
                    print("After GD: mse: %.5f psnr: %.2f ssim: %.2f run time: %.4f" % (mse_GD.item(), rec_PSNR_GD, rec_SSIM_GD, update_end - update_start))
                Img_rec_yuv_GD = Img_rec_yuv.copy()
                Img_rec_yuv_GD[:, :, 0] = X_rec_GD * 255
                im_rec_rgb_GD = cv2.cvtColor(Img_rec_yuv_GD, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb_GD = np.clip(im_rec_rgb_GD, 0, 255).astype(np.uint8)
                if flag_i == 0:
                    cv2.imwrite(os.path.join(out_path, "%s_ite_%d_GD_PSNR_%.2f_SSIM_%.2f.png" % (single_imgName, i, rec_PSNR_GD, rec_SSIM_GD)), im_rec_rgb_GD)

                ########## DIP ite PSNR and patches reconstruction ##########
                Prediction_value_NN = x_gen.detach().cpu().numpy()
                X_rec_NN = np.clip(col2im_CS_py(Prediction_value_NN.transpose(), row, col, row_new, col_new, args.patch_size), 0, 1)
                rec_PSNR_NN = psnr(X_rec_NN * 255, Iorg.astype(np.float64))
                rec_SSIM_NN = ssim(X_rec_NN * 255, Iorg.astype(np.float64), data_range=255)
                if flag_i == 0:
                    print("After NN: mse: %.5f psnr: %.2f ssim: %.2f run time: %.4f" % (mse_NN.item(), rec_PSNR_NN, rec_SSIM_NN, projection_end - projection_start))
                Img_rec_yuv_NN = Img_rec_yuv.copy()
                Img_rec_yuv_NN[:, :, 0] = X_rec_NN * 255
                im_rec_rgb_NN = cv2.cvtColor(Img_rec_yuv_NN, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb_NN = np.clip(im_rec_rgb_NN, 0, 255).astype(np.uint8)
                if flag_i == 0:
                    cv2.imwrite(os.path.join(out_path, "%s_ite_%d_NN_PSNR_%.2f_SSIM_%.2f.png" % (single_imgName, i, rec_PSNR_NN, rec_SSIM_NN)), im_rec_rgb_NN)

                ########## Fusion ite PSNR and patches reconstruction ##########
                Prediction_value_Fusion = x_new.detach().cpu().numpy()
                X_rec_Fusion = np.clip(col2im_CS_py(Prediction_value_Fusion.transpose(), row, col, row_new, col_new, args.patch_size), 0, 1)
                rec_PSNR_Fusion = psnr(X_rec_Fusion * 255, Iorg.astype(np.float64))
                rec_SSIM_Fusion = ssim(X_rec_Fusion * 255, Iorg.astype(np.float64), data_range=255)
                if flag_i == 0:
                    print("After Fusion: psnr: %.2f ssim: %.2f" % (rec_PSNR_Fusion, rec_SSIM_Fusion))
                Img_rec_yuv_Fusion = Img_rec_yuv.copy()
                Img_rec_yuv_Fusion[:, :, 0] = X_rec_Fusion * 255
                im_rec_rgb_Fusion = cv2.cvtColor(Img_rec_yuv_Fusion, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb_Fusion = np.clip(im_rec_rgb_Fusion, 0, 255).astype(np.uint8)
                if flag_i == 0:
                    cv2.imwrite(os.path.join(out_path, "%s_ite_%d_Fusion_PSNR_%.2f_SSIM_%.2f.png" % (single_imgName, i, rec_PSNR_Fusion, rec_SSIM_Fusion)), im_rec_rgb_Fusion)

                ########## Save the best Fusion PSNR ##########
                if rec_PSNR_Fusion > best_PSNR_Fusion:
                    best_PSNR_Fusion = rec_PSNR_Fusion
                    best_SSIM_Fusion = rec_SSIM_Fusion
                    cv2.imwrite(os.path.join(out_path, "%s_Best_PSNR.png" % (single_imgName)), im_rec_rgb_Fusion)
                    PSNR_Fusion_best[img_no] = best_PSNR_Fusion
                    SSIM_Fusion_best[img_no] = best_SSIM_Fusion

                ########## Save PSNR of each img at each ite ##########
                PSNR_GD_All[i, img_no] = rec_PSNR_GD
                PSNR_NN_All[i, img_no] = rec_PSNR_NN
                PSNR_Fusion_All[i, img_no] = rec_PSNR_Fusion

                ########## Save SSIM of each img at each ite ##########
                SSIM_GD_All[i, img_no] = rec_SSIM_GD
                SSIM_NN_All[i, img_no] = rec_SSIM_NN
                SSIM_Fusion_All[i, img_no] = rec_SSIM_Fusion

    # return PSNR_GD_All, PSNR_NN_All, PSNR_Fusion_All, PSNR_Fusion_best, SSIM_GD_All, SSIM_NN_All, SSIM_Fusion_All, SSIM_Fusion_best, optimal_lamb_1_arr, optimal_lamb_2_arr, optimal_patch_idx_arr, inv_err_arr
    return PSNR_GD_All, PSNR_NN_All, PSNR_Fusion_All, PSNR_Fusion_best, SSIM_GD_All, SSIM_NN_All, SSIM_Fusion_All, SSIM_Fusion_best

if __name__ == '__main__':
    ############# Initialize the random seed ##############
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    dtype = torch.float64

    ############# Generate CS Sampling Matrix: Phi #############
    if args.degradation == 'CS':
        ######## Generate Gaussian matrix ##########
        if args.A_type == 'Gaussian':
            if args.A_look == 'A':
                Phi_L = generate_gaussian_matrix(int(args.compression_rate * args.patch_size ** 2), int(args.patch_size ** 2), args.use_complex, dtype)
        ######## Generate Fourier (Orthogonal) matrix ##########
        elif args.A_type == 'Fourier':
            if args.A_look == 'A':
                Phi_L = generate_orthogonal_matrix(int(args.compression_rate * args.patch_size ** 2), int(args.patch_size ** 2), args.use_complex, args.diff_A, dtype)

    ######## Assure measurement matrix on device and correct type ##########
    if args.A_look == 'A':
        Phi = Phi_L.to(device)
        Phi = Phi

    ######## Channels, kernel sizes used in Decoder ##########
    num_channel_list = [[128, 128, 128, 128], [128, 128, 128, 128], [128, 128, 128, 128]]
    kernel_size_list = [3,3,3]

    ############# testing data and saving path #############
    filepaths_test = glob.glob(os.path.join(args.data_dir, args.test_name) + '/*.tif')
    out_path = os.path.join('./results_test', "_".join(map(str, [args.NN, args.loss_function,
                                                                    args.seed, args.lr_NN, args.lr_GD,
                                                                    args.weight_decay, args.need_sigmoid,
                                                                    args.test_name, args.degradation,
                                                                    args.compression_rate, args.X_init,
                                                                    args.decodetype, args.kernel_size,
                                                                    args.outer_ite, args.multi_look,
                                                                    args.num_look, args.patch_size,
                                                                    args.A_type, args.use_autograd_GD, args.lamb,
                                                                    args.Fusion, args.pseudo_inverse,
                                                                    args.ite_inverse, args.use_Schulz, args.decision,
                                                                    args.A_look, args.use_complex, args.diff_A,
                                                                    args.downsample, args.crop,
                                                                    kernel_size_list, args.DIP_avg, args.DIP_final,
                                                                    num_channel_list[0], args.exact_init,
                                                                    args.exact_inv_ite])))
    os.makedirs(out_path, exist_ok=True)

    ############# training function #############
    PSNR_GD_All, PSNR_NN_All, PSNR_Fusion_All, PSNR_Fusion_best, SSIM_GD_All, SSIM_NN_All, SSIM_Fusion_All, SSIM_Fusion_best = train(Phi, num_channel_list, kernel_size_list, filepaths_test, out_path, device, dtype)

    ############# save the log files #############
    with open(out_path + '/' + 'PSNR_GD' + '.pkl', 'wb') as psnr_GD_file:
        pickle.dump(PSNR_GD_All, psnr_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'PSNR_NN' + '.pkl', 'wb') as psnr_NN_file:
        pickle.dump(PSNR_NN_All, psnr_NN_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'PSNR_Fusion' + '.pkl', 'wb') as psnr_Fusion_file:
        pickle.dump(PSNR_Fusion_All, psnr_Fusion_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'PSNR_Best' + '.pkl', 'wb') as psnr_Best_file:
        pickle.dump(PSNR_Fusion_best, psnr_Best_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(out_path + '/' + 'SSIM_GD' + '.pkl', 'wb') as ssim_GD_file:
        pickle.dump(SSIM_GD_All, ssim_GD_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_NN' + '.pkl', 'wb') as ssim_NN_file:
        pickle.dump(SSIM_NN_All, ssim_NN_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_Fusion' + '.pkl', 'wb') as ssim_Fusion_file:
        pickle.dump(SSIM_Fusion_All, ssim_Fusion_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_path + '/' + 'SSIM_Best' + '.pkl', 'wb') as ssim_Best_file:
        pickle.dump(SSIM_Fusion_best, ssim_Best_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done.')
