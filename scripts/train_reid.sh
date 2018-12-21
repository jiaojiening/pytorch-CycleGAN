python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid --dataset_mode Duke --model reid --pool_size 50 --gpu 5 --no_dropout --batch_size 32
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16 --dataset_mode Duke --model reid --pool_size 50 --gpu 5 --no_dropout --batch_size 16
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16_lr_0.01 --dataset_mode Duke --model reid --gpu 7 --no_dropout --batch_size 16 --reid_lr 0.01
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR_16_lr_0.01 --dataset_mode Duke --model reid --gpu 0 --no_dropout --batch_size 16 --reid_lr 0.01 --NR

python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8 --dataset_mode Duke --model reid --gpu 3 --no_dropout --batch_size 16 --reid_lr 0.01 --up_scale 8

python train_reid.py --dataroot DukeMTMC-reID --name SR_Duke_Reid_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 6 --no_dropout --batch_size 16 --reid_lr 0.01 --save_phase train --SR_name Duke_cCyclegan
python train_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 6 --no_dropout --batch_size 16 --reid_lr 0.01 --save_phase train --SR_name Duke_SRcCyclegan

python train_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --batch_size 16 --reid_lr 0.01 --save_phase train --SR_name Duke_paired_cCyclegan
python train_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --batch_size 16 --reid_lr 0.01 --save_phase train --SR_name Duke_paired_cCyclegan_upscale_8 --up_scale 8

python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 2 --no_dropout --batch_size 2
python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan_step_20 --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks
python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 4,5,6,7 --no_dropout --batch_size 24 --netG resnet_6blocks --display_id -1
python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan_lr_0.01 --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks --reid_lr 0.01

python train_joint.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8 --SR_name  --reid_name paired_cCyclegan_Duke_Reid_upscale_8_16_lr_0.01 --dataset_mode paired_Duke --model reid_cCycle_gan --up_scale 8 --gpu 0,7 --no_dropout --batch_size 12Duke_paired_cCyclegan_upscale_8