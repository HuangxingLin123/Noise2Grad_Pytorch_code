# Noise2Grad_Pytorch_code
Pytorch implementation for "Noise2Grad: Extract Image Noise to Denoise" (IJCAI-2021)

Paper Link: [Link](https://www.ijcai.org/proceedings/2021/115)


Training
=================================
Download the training datasets from [Google Drive](https://drive.google.com/drive/folders/1xRJLe8D3rhUWssnZczSEy3-LqPSRZ0kN).
Unzip "ground_truth.zip" and "reference_clean_image.zip" and put them in "./datasets/train/". Use the code 'matlab_code_for_synthesizing_gaussian_noise.m' or 'matlab_code_for_synthesizing_speckle_noise.m' or 'python_code_for_synthesizing_poisson_noise.py' to synthesize noisy images, and then put them into "./datasets/train/noisy_train/".





