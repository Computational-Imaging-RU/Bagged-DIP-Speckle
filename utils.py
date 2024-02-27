import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

dtype = torch.float64


def rgb2ycbcr(rgb):
  m = np.array([[65.481, 128.553, 24.966],
                [-37.797, -74.203, 112],
                [112, -93.786, -18.214]])
  shape = rgb.shape
  if len(shape) == 3:
    rgb = rgb.reshape((shape[0] * shape[1], 3))
  ycbcr = np.dot(rgb, m.transpose() / 255.)
  ycbcr[:, 0] += 16.
  ycbcr[:, 1:] += 128.
  return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
  m = np.array([[65.481, 128.553, 24.966],
                [-37.797, -74.203, 112],
                [112, -93.786, -18.214]])
  shape = ycbcr.shape
  if len(shape) == 3:
    ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
  rgb = copy.deepcopy(ycbcr)
  rgb[:, 0] -= 16.
  rgb[:, 1:] -= 128.
  rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
  return rgb.clip(0, 255).reshape(shape)


def imread_CS_py(Iorg, block_size):
  [row, col] = Iorg.shape
  row_pad = block_size - np.mod(row, block_size)
  col_pad = block_size - np.mod(col, block_size)
  if col_pad == block_size:
    Ipad = Iorg
    if row_pad == block_size:
      Ipad = Ipad
    else:
      Ipad = np.concatenate((Ipad, np.zeros([row_pad, col])), axis=0)
  else:
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    if row_pad == block_size:
      Ipad = Ipad
    else:
      Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)

  [row_new, col_new] = Ipad.shape

  return [Iorg, row, col, Ipad, row_new, col_new]


def imread_CS_py_(Iorg, block_size):
  [row, col] = Iorg.shape
  row_pad = block_size - np.mod(row, block_size)
  col_pad = block_size - np.mod(col, block_size)
  Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
  Ipad = np.concatenate((Ipad, np.zeros([row_pad, col + col_pad])), axis=0)
  [row_new, col_new] = Ipad.shape

  return [Iorg, row, col, Ipad, row_new, col_new]

def img2col_py(Ipad, block_size):
  [row, col] = Ipad.shape
  row_block = row / block_size
  col_block = col / block_size
  block_num = int(row_block * col_block)
  img_col = np.zeros([block_size ** 2, block_num])
  count = 0
  for x in range(0, row - block_size + 1, block_size):
    for y in range(0, col - block_size + 1, block_size):
      img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
      # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
      count = count + 1
  return img_col

def img2col_py_(Ipad, block_size):
  [row, col] = Ipad.shape
  row_block = row / block_size
  col_block = col / block_size
  block_num = int(row_block * col_block)
  img_col = np.zeros([block_size ** 2, block_num])
  count = 0
  for x in range(0, row - block_size + 1, block_size):
    for y in range(0, col - block_size + 1, block_size):
      img_col[:, count] = Ipad[x:x + block_size, y:y + block_size].reshape([-1])
      count = count + 1
  return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new, block_size):
  X0_rec = np.zeros([row_new, col_new])
  count = 0
  for x in range(0, row_new - block_size + 1, block_size):
    for y in range(0, col_new - block_size + 1, block_size):
      X0_rec[x:x + block_size, y:y + block_size] = X_col[:, count].reshape([block_size, block_size])
      # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
      count = count + 1
  X_rec = X0_rec[:row, :col]
  return X_rec

def psnr(img1, img2):
  img1.astype(np.float32)
  img2.astype(np.float32)
  mse = np.mean((img1 - img2) ** 2)
  # print('img1', img1)
  if mse == 0:
    return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def num_param(net):
  s = sum([np.prod(list(p.size())) for p in net.parameters()]);
  return s

def gen_latent_code_patch(batch_size, patch_size, num_channels, out_ch):
  # img_meas, A_sensing are all tensor
  totalupsample = 2 ** (len(num_channels) - 1)
  # if running as decoder/compressor
  width, height = 0, 0

  w = patch_size
  width = int(w / (totalupsample))
  height = int(w / (totalupsample))

  # (1, num_channel_init, width_init, height_init)
  shape = [batch_size, num_channels[0], width, height]
  # print("shape of latent code: ", shape)
  # latent_code = nn.Parameter(torch.zeros(shape))
  latent_code = torch.zeros(shape)
  latent_code.data.normal_()
  latent_code.data *= 1. / 10
  return latent_code

def gen_latent_code(batch_size, img_meas, num_channels, A_sensing, out_ch):
  # img_meas, A_sensing are all tensor
  totalupsample = 2 ** (len(num_channels) - 1)
  # if running as decoder/compressor
  width, height = 0, 0
  if len(img_meas.shape) == 4:
    width = int(img_meas.data.shape[2] / (totalupsample))
    height = int(img_meas.data.shape[3] / (totalupsample))
  # if running compressive imaging
  elif len(img_meas.shape) == 3:
    w = np.sqrt(int(A_sensing.shape[1] / out_ch))
    width = int(w / (totalupsample))
    height = int(w / (totalupsample))

  # (1, num_channel_init, width_init, height_init)
  shape = [batch_size, num_channels[0], width, height]
  print("shape of latent code: ", shape)
  # latent_code = nn.Parameter(torch.zeros(shape))
  latent_code = torch.zeros(shape)
  latent_code.data.normal_()
  latent_code.data *= 1. / 10
  return latent_code



def generate_gaussian_matrix(m, n, use_complex, dtype):
  if use_complex:
    a1 = torch.rand(m, n).type(dtype)
    a = torch.complex(a1, a1)
  else:
    a = torch.rand(m, n).type(dtype)
  return a

def generate_orthogonal_matrix(m, n, use_complex, diff_A, dtype):
  if use_complex:
    a1 = torch.rand(n, m).type(dtype)
    if diff_A:
      a2 = torch.rand(n, m).type(dtype)
    else:
      a2 = a1
    a = torch.complex(a1, a2)
  else:
    a = torch.rand(n, m).type(dtype)
    # a = np.random.random(size=(n, m))
    # a = np.random.randn(n, m)
  # q, _ = np.linalg.qr(a)
  q, _ = torch.linalg.qr(a)
  return q.T
