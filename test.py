import numpy as np
import os,re,random
import skimage.color as color
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation as sni
import caffe
import argparse
from PIL import Image
import scipy.misc

def parse_args():
    parser = argparse.ArgumentParser(description='Example-based Colorization via Dense Encoding pyramids')
    parser.add_argument('-gray',dest='gray',help='input grayscale image', type=str)
    parser.add_argument('-refer',dest='refer',help='input reference image', type=str)
    parser.add_argument('-output',dest='output',help='output colorful image', type=str)
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('-sm_out',dest='sm_out',help='small output', type=str,default='')
    args = parser.parse_args()
    return args

def generate(size):

    global Tsize,img_rgb,ref_rgb,temp_small,init_level,sm_out
    prototxt="./models/test/DEPN_deploy_%d.prototxt"%size
    if size==init_level:
        model="./models/DEPN_init.caffemodel"
    else:
        model="./models/DEPN_sub.caffemodel"

    net = caffe.Net(prototxt, model, caffe.TEST)

    net.params['class8_ab'][0].data[:,:,0,0] = pts.transpose((1,0))
    (H_out,W_out) = net.blobs['class8_ab'].data.shape[2:]

    img_rgb_rs=caffe.io.resize_image(img_rgb,(size,size))
    img_lab_rs = color.rgb2lab(img_rgb_rs)
    img_l_rs = img_lab_rs[:,:,0]

    ref_rgb_rs=caffe.io.resize_image(ref_rgb,(size,size))
    ref_lab=color.rgb2lab(ref_rgb_rs)

    if size!=init_level:
        net.blobs['small_ab'].data[0,0,:,:]=temp_small[:,:,0]
        net.blobs['small_ab'].data[0,1,:,:]=temp_small[:,:,1]

    net.blobs['img_l'].data[0,0,:,:]=img_l_rs
    net.blobs['ref_ab'].data[0,0,:,:]=ref_lab[:,:,1]
    net.blobs['ref_ab'].data[0,1,:,:]=ref_lab[:,:,2]

    net.forward()

    ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0))

    if size==Tsize:
        (H_orig,W_orig) = img_rgb.shape[:2]
        ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1))

        img_lab = color.rgb2lab(img_rgb)
        img_l = img_lab[:,:,0]

        img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2)

        img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8')
        scipy.misc.toimage(img_rgb_out).save(output)
        print("\nGenerate successful in "+output)
        
        '''
        # if you need to use the small outcome to train the higher levels, please use the codes below:
        small_img_rgb=caffe.io.resize_image(img_rgb,(size/4,size/4))
        small_img_lab = color.rgb2lab(small_img_rgb)
        small_img_l = small_img_lgb[:,:,0]
        small_img_lab_out = np.concatenate((small_img_l[:,:,np.newaxis],ab_dec),axis=2)
        small_img_rgb_out = (255*np.clip(color.lab2rgb(small_img_lab_out),0,1)).astype('uint8')
        scipy.misc.toimage(small_img_rgb_out).save(sm_out)
        '''

    else:
        temp_small = ab_dec

args = parse_args()
caffe.set_mode_gpu()
caffe.set_device(args.gpu)
gray=args.gray
refer=args.refer
output=args.output
sm_out=args.sm_out
pts = np.load('./resources/pts_in_hull.npy')
temp_small=None
init_level=64

img_rgb = caffe.io.load_image(gray)
ref_rgb = caffe.io.load_image(refer)

(H_orig,W_orig) = img_rgb.shape[:2]

if H_orig >= 1024 or W_orig >= 1024:
    Tsize=1024
elif H_orig >= 512 or W_orig >= 512:
    Tsize=512
elif H_orig >= 256 or W_orig >= 256:
    Tsize=256
elif H_orig >= 128 or W_orig >= 128:
    Tsize=128
else:
    Tsize=64

level=init_level
while(level<=Tsize):
    generate(level)
    level=level*2

