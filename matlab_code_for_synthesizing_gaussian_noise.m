
path = dir('./datasets/train/ground_truth/*.png')
new_folder = './datasets/train/noisy_train/'
mkdir(new_folder); 

for k = 1:200000


   img = imread(['./datasets/train/ground_truth/' path(k).name]);

   sigma=randi([1,50],1); 

   p_noise=imnoise(img, 'gaussian',0, sigma^2/255^2);

   imwrite(p_noise,['./datasets/train/noisy_train/' path(k).name]); 



end