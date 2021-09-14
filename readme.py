
## Train
python train.py --dataroot ./datasets/train/noisy_train/ --name new --model denoise


## Test
python test.py --dataroot ./datasets/test/noisy_test/ --name new --model denoise


## Test with our pretrained model

#gaussian pretrained
python test.py --dataroot ./datasets/test/noisy_test/ --name gaussian_pretrained --model denoise

#poisson pretrained
python test.py --dataroot ./datasets/test/noisy_test/ --name poisson_pretrained --model denoise

#speckle pretrained
python test.py --dataroot ./datasets/test/noisy_test/ --name speckle_pretrained --model denoise



######
After the test, results are saved in './results/'.

Run "psnr_and_ssim.py" to caculate psnr and ssim.
