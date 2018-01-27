python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR --dataset_mode Duke --model reid --gpu 3 --no_dropout --NR

python train_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks  --dataset_mode SR_Duke --model reid --gpu 7 --no_dropout --SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks --reid_lr 0.06

python train_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_cyclegan_upscale_8

python train_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8_GT_A --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A --reid_lr 0.06

python train_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Duke --model reid --gpu 2 --no_dropout --SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks

python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8 --dataset_mode Duke --model reid --gpu 0 --no_dropout --up_scale 8 --serial_batches

python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_LR --dataset_mode LR_Duke --model LR_reid --gpu 0 --no_dropout --up_scale 8

python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_concate --dataset_mode Duke --model reid --gpu 0 --no_dropout --up_scale 8
