import os, sys
import glob
import cv2
import numpy as np
from skimage.util import view_as_blocks, montage, pad
from sklearn.feature_extraction.image import extract_patches_2d
# from skimage.util.shape import view_as_blocks

get_image_name = lambda fn: os.path.basename(fn).lstrip('ortho_eval_').rstrip('.tif')
get_row_col = lambda fn: tuple(map(int, get_image_name(fn).split('_')))

def img2tiles(img_fn, size=224, dst_folder=''):
    '''
    split a large image into multiple tiles of square shape and fixed size
    If the image size is not multiple of given size, zero-pad the edges
    '''
    assert type(size)==int
    _, image_ext = os.path.splitext(img_fn)
    image_name = os.path.basename(img_fn).rstrip(image_ext)
    img = cv2.imread(img_fn)
    print("shape of image is {}".format(img.shape))
    h,w= img.shape[:2]
    w_pad, h_pad = size - w%size, size - h%size
    img = pad(img, ((0, h_pad), (0, w_pad), (0, 0)))
    blocks = view_as_blocks(img, (size, size, 3)) #shape: (new_h, new_w, 1, size, size, 3)
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

if __name__ == '__main__':
    # img2tiles(sys.argv[1], dst_folder=sys.argv[2])
    img_fn_list = glob.glob("/Users/derekz/derekz/img_seg/patchify.py/test/patches/*.tif")
    img_fn_list.sort(key=lambda fn: get_row_col(fn))
    print('\n'.join(img_fn_list))
    exit(-1)
    dst = "/Users/derekz/derekz/img_seg/patchify.py/test/original/recovered.tif"
    tiles2img(img_fn_list, dst)


