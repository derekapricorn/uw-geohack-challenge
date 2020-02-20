import os, sys
import glob
import cv2
from PIL import Image
import numpy as np
from skimage.transform import resize
from skimage.util import view_as_blocks, montage, pad
from sklearn.feature_extraction.image import extract_patches_2d
from functools import partial
import matplotlib.pyplot as plt
# from skimage.util.shape import view_as_blocks

get_image_name = lambda fn: os.path.basename(fn).lstrip('ortho_eval_').rstrip('.tif')
get_row_col = lambda fn: tuple(map(int, get_image_name(fn).split('_')))

def batch_process(func, src_folder_path, dst_folder_path):
    batch_process = partial(func, dst_folder=dst_folder_path)
    img_files = glob.glob(src_folder_path+'/*')
    [batch_process(img_fn) for img_fn in img_files]

def img2tiles(img_fn, w_size=672, h_size=448, dst_folder=''):
    '''
    split a large image into multiple tiles of square shape and fixed size
    If the image size is not multiple of given size, zero-pad the edges
    '''
    assert type(w_size)==int and type(h_size)==int
    _, image_ext = os.path.splitext(img_fn)
    image_name = os.path.basename(img_fn).rstrip(image_ext)
    img = cv2.imread(img_fn)
    print("shape of image is {}".format(img.shape))
    h,w= img.shape[:2]
    w_pad, h_pad = w_size - w%w_size, h_size - h%h_size
    img = pad(img, ((0, h_pad), (0, w_pad), (0, 0)))
    blocks = view_as_blocks(img, (h_size, w_size, 3)) #shape: (new_h, new_w, 1, size, size, 3)
    new_h, new_w = blocks.shape[:2]
    if dst_folder:
        for i in range(new_h):
            for j in range(new_w):
                fn = os.path.join(dst_folder, image_name + '_{0}_{1}{2}'.format(i,j,image_ext))
                cv2.imwrite(fn, blocks[i,j,0,...])


def tiles2img(img_fn_list, dst_fn):
    '''
    arr_in(K, M, N[, C]) ndarray, where K is number of images to be collaged
    '''
    arr_in = np.zeros((len(img_fn_list), 224, 224, 3))
    for idx, img_fn in enumerate(img_fn_list):
        img = cv2.imread(img_fn)
        arr_in[idx,...] = img
    h_, w_ = get_row_col(img_fn_list[-1])
    grid_shape = (h_+1, w_+1, 3)
    res = montage(arr_in, grid_shape, multichannel=True)
    cv2.imwrite(dst_fn, res)

def resize_img(img_fn, w_size=672, h_size=448, dst_folder=''):
    img = cv2.imread(img_fn)
    img = resize(img, (h_size, w_size, 3), preserve_range=True)
    if dst_folder:
        fn = os.path.join(dst_folder, os.path.basename(img_fn))
        cv2.imwrite(fn, img)

def resize_mask(mask_fn, w_size=672, h_size=448, dst_folder=''):
    mask = Image.open(mask_fn)
    mask = mask.resize((w_size, h_size), Image.NEAREST)
    if dst_folder:
        fn = os.path.join(dst_folder, os.path.basename(mask_fn))
        mask.save(fn)

def replace_pixel_vals(img_fn, positive_codes=[19], dst_folder=''):
    img_arr = cv2.imread(img_fn)
    assert 1 not in positive_codes
    img_arr[img_arr==1] = 255
    for c in positive_codes:
        img_arr[img_arr==c]=1
    img_arr[img_arr!=1] = 0
    if dst_folder:
        fn = os.path.join(dst_folder, os.path.basename(img_fn))
        cv2.imwrite(fn, img_arr)
        
if __name__ == '__main__':
    # img2tiles(sys.argv[1], dst_folder=sys.argv[2])

    # batch_process(resize_img, sys.argv[1], sys.argv[2])

    resize_img(sys.argv[1], dst_folder=sys.argv[2])
    # img_fn_list = glob.glob("/Users/derekz/derekz/img_seg/patchify.py/test/patches/*.tif")
    # img_fn_list.sort(key=lambda fn: get_row_col(fn))
    # print('\n'.join(img_fn_list))
    # exit(-1)
    # dst = "/Users/derekz/derekz/img_seg/patchify.py/test/original/recovered.tif"
    # tiles2img(img_fn_list, dst)


