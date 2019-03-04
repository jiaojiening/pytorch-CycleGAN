python train.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8 --dataset_mode Duke --model cycle_gan --gpu 0,2 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A --dataset_mode Duke --model cycle_gan --gpu 2,3 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_paired_cyclegan_upscale_8 --dataset_mode paired_Duke --model cycle_gan --gpu 0,1 --up_scale 8 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode paired_Duke --model cCycle_gan --pool_size 50 --gpu 0,1 --no_dropout --batch_size 14 --up_scale 8
python train.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8_resnet_6blocks --dataset_mode paired_Duke --model cCycle_gan --pool_size 50 --gpu 0,1 --no_dropout --up_scale 8 --netG resnet_6blocks

python train.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_upscale_8 --dataset_mode Duke --model SRcCycle_gan --gpu 0,1 --no_dropout --up_scale 8 --netG resnet_6blocks

python train.py --dataroot DukeMTMC-reID --name Duke_gan_upscale_8_resnet_6blocks --dataset_mode Duke --model gan --gpu 0,1 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_gan_upscale_8_resnet_6blocks_rec_20 --dataset_mode Duke --model gan --gpu 7,4 --no_dropout --up_scale 8 --netG resnet_6blocks

python train.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model hybrid_cycle_gan --gpu 0,3 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks_v2 --dataset_mode Duke --model hybrid_cycle_gan --gpu 0,1,6,7 --no_dropout --up_scale 8 --netG resnet_6blocks --batch_size 32
python train.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model hybrid_cCycle_gan --gpu 0,1 --no_dropout --up_scale 8 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks_batch_32 --dataset_mode Duke --model hybrid_cCycle_gan --gpu 0,1,6,7 --no_dropout --up_scale 8 --netG resnet_6blocks --batch_size 32
