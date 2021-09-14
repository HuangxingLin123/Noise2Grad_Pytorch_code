import skimage
import cv2
from skimage.measure import compare_psnr, compare_ssim
import os


def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]

    return compare_ssim(im1_y, im2_y)


## ground truth
dir='./datasets/test/ground_truth/'


##  denoiseed results
# dir2="./results/gaussian_pretrained/test_latest/images/"
# dir2="./results/speckle_pretrained/test_latest/images/"
# dir2="./results/poisson_pretrained/test_latest/images/"
dir2="./results/new/test_latest/images/"

ssim=0
psnr=0

total_num=0
for picname in os.listdir(dir):
    img1 = cv2.imread(dir+picname)
    name = picname.split('.')[0]
    img2 = cv2.imread(dir2+name+'_X_denoise.png')
    (h, w, n) = img1.shape
    (h2, w2, n) = img2.shape

    if h2 != h or w2 != w:
        print('error', picname)
        break

    a = calc_ssim(img1, img2)
    b = calc_psnr(img1, img2)
    print(picname, ':', a, b)
    ssim += a
    psnr += b

    total_num= total_num+1


ssim=ssim/total_num
psnr=psnr/total_num
print("ssim=",ssim)
print("psnr=",psnr)




