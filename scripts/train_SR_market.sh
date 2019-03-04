python train.py --dataroot Market --name Market_cyclegan_upscale_8 --dataset_mode Market --model cycle_gan --gpu 0,1 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot Market --name Market_SRcCyclegan_upscale_8 --dataset_mode Market --model SRcCycle_gan --gpu 2,3 --no_dropout --up_scale 8
python train.py --dataroot Market --name Market_SRcCyclegan_upscale_8_resnet_6blocks --dataset_mode Market --model SRcCycle_gan --gpu 2,3 --no_dropout --up_scale 8 --netG resnet_6blocks

python train.py --dataroot Market --name Market_gan_upscale_8_resnet_6blocks --dataset_mode Market --model gan --gpu 0,1 --no_dropout --up_scale 8 --netG resnet_6blocks

python train.py --dataroot Market --name Market_hybrid_cyclegan_upscale_8_resnet_6blocks --dataset_mode Market --model hybrid_cycle_gan --gpu 0,3 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot Market --name Market_hybrid_cyclegan_upscale_8_resnet_6blocks --dataset_mode Market --model hybrid_cycle_gan --gpu 0,1,2,3 --no_dropout --up_scale 8 --netG resnet_6blocks --batch_size 32
python train.py --dataroot Market --name Market_hybrid_cyclegan_upscale_8_resnet_6blocks_v2 --dataset_mode Market --model hybrid_cycle_gan --gpu 0,1,6,7 --no_dropout --up_scale 8 --netG resnet_6blocks --batch_size 32
python train.py --dataroot Market --name Market_hybrid_cCyclegan_upscale_8_resnet_6blocks --dataset_mode Market --model hybrid_cCycle_gan --gpu 0,1 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot Market --name Market_hybrid_cCyclegan_upscale_8_resnet_6blocks --dataset_mode Market --model hybrid_cCycle_gan --gpu 0,1,2,3 --no_dropout --up_scale 8 --netG resnet_6blocks --batch_size 32