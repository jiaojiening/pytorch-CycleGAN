python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid --dataset_mode Duke --model reid --pool_size 50 --gpu 5 --no_dropout --batch_size 32
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16 --dataset_mode Duke --model reid --pool_size 50 --gpu 5 --no_dropout --batch_size 16
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16_lr_0.01 --dataset_mode Duke --model reid --gpu 7 --no_dropout --batch_size 16 --reid_lr 0.01
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR_16_lr_0.01 --dataset_mode Duke --model reid --gpu 0 --no_dropout --batch_size 16 --reid_lr 0.01 --NR
python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan_lr_0.01 --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks --reid_lr 0.01

python train_reid.py --dataroot DukeMTMC-reID --name SR_Duke_Reid_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_cCyclegan
python train_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_SRcCyclegan
python train_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 2 --no_dropout --SR_name Duke_paired_cCyclegan

python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8 --dataset_mode Duke --model reid --gpu 0 --no_dropout --up_scale 8
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8 --dataset_mode Duke --model reid_attr --gpu 3 --up_scale 8 Duke_Reid_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8 --dataset_mode Duke --model reid_attr --gpu 2 --up_scale 8 --reid_name Duke_Reid_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_paired_cCyclegan_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks  --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks
python train_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_cyclegan_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name paired_cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_paired_cyclegan_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks

python train_joint.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8 --SR_name Duke_paired_cCyclegan_upscale_8 --reid_name paired_cCyclegan_Duke_Reid_upscale_8_16_lr_0.01 --dataset_mode paired_Duke --model reid_cCycle_gan --up_scale 8 --gpu 0,1 --no_dropout --use_feat --batch_size 12
python train_joint.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks --SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks --reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode paired_Duke --model reid_cCycle_gan --up_scale 8 --gpu 2,3 --no_dropout --use_feat --batch_size 12 --netG resnet_6blocks


python train_reid.py --dataroot Market --name Market_Reid_16 --dataset_mode Market --model reid --gpu 7 --no_dropout --batch_size 16
python train_reid.py --dataroot Market --name Market_Reid_16_lr_0.01 --dataset_mode Market --model reid --gpu 7 --no_dropout --batch_size 16 --reid_lr 0.01
python train_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01 --dataset_mode Market --model reid --gpu 3 --no_dropout --batch_size 16 --reid_lr 0.01 --NR
python train_reid.py --dataroot Market --name Market_ReidSRcCyclegan --dataset_mode Market --model reid_SRcCycle_gan --gpu 1,2 --no_dropout --batch_size 14 --netG resnet_6blocks

python train_reid.py --dataroot Market --name Market_Reid_upscale_8 --dataset_mode Market --model reid --gpu 7 --no_dropout --up_scale 8
python train_reid.py --dataroot Market --name Market_Reid_attr_upscale_8 --dataset_mode Market --model reid_attr --gpu 3 --no_dropout --up_scale 8
python train_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_SRcCyclegan_upscale_8
python train_reid.py --dataroot Market --name cCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_cCyclegan_upscale_8
python train_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks
python train_reid.py --dataroot Market --name cyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 3 --no_dropout --SR_name Market_cyclegan_upscale_8
