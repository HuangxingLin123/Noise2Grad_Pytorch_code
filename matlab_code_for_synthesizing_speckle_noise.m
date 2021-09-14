path = dir('./datasets/train/ground_truth/*.png')
new_folder = './datasets/train/noisy_train/'
mkdir(new_folder); 

for k = 1:200000


   img = imread(['./datasets/train/ground_truth/' path(k).name]);


   v=randi([1,20],1)/100.0;

   p_noise=imnoise(img, 'speckle', v);

   imwrite(p_noise,['./datasets/train/noisy_train/' path(k).name]); 



end


