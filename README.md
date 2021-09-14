# Noise2Grad_Pytorch_code
Pytorch implementation for "Noise2Grad: Extract Image Noise to Denoise" (IJCAI-2021)

Paper Link: [Link](https://www.ijcai.org/proceedings/2021/115)


Training
=================================
Download the training datasets from [Google Drive](https://drive.google.com/drive/folders/1xRJLe8D3rhUWssnZczSEy3-LqPSRZ0kN).
Unzip "ground_truth.zip" and "reference_clean_image.zip" and put them in "./datasets/train/". Use the code 'matlab_code_for_synthesizing_gaussian_noise.m' or 'matlab_code_for_synthesizing_speckle_noise.m' or 'python_code_for_synthesizing_poisson_noise.py' to synthesize noisy images, and then put them into "./datasets/train/noisy_train/".


- Train the model:

*python train.py --dataroot ./datasets/train/noisy_train/ --name new --model denoise*


Testing
=======

Download the testing dataset from [Google Drive](https://drive.google.com/drive/folders/1xRJLe8D3rhUWssnZczSEy3-LqPSRZ0kN).

Unzip "ground_truth.zip" in './datasets/test/'. Use the code 'matlab_code_for_synthesizing_gaussian_noise.m' or 'matlab_code_for_synthesizing_speckle_noise.m' or 'python_code_for_synthesizing_poisson_noise.py' to synthesize noisy images, and then put them into "./datasets/test/noisy_test/".


- Test:

*python test.py --dataroot ./datasets/test/noisy_test/ --name new --model denoise*

- Test with our pretrained model:

#gaussian pretrained

*python test.py --dataroot ./datasets/test/noisy_test/ --name gaussian_pretrained --model denoise*

#poisson pretrained

*python test.py --dataroot ./datasets/test/noisy_test/ --name poisson_pretrained --model denoise*

#speckle pretrained

*python test.py --dataroot ./datasets/test/noisy_test/ --name speckle_pretrained --model denoise*

After the test, results are saved in './results/'.

Run "psnr_and_ssim.py" to caculate psnr and ssim.


