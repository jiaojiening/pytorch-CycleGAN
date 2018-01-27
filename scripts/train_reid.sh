python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR --dataset_mode Duke --model reid --gpu 2 --no_dropout --NR
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8 --dataset_mode Duke --model reid --gpu 5 --no_dropout --up_scale 8 --reid_lr 0.05
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_GT_A --dataset_mode Duke --model reid --gpu 3 --no_dropout --up_scale 8
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8 --dataset_mode Duke --model reid_attr --gpu 3 --up_scale 8 Duke_Reid_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8 --dataset_mode Duke --model reid_attr --gpu 2 --up_scale 8 --reid_name Duke_Reid_upscale_8

python train_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks  --dataset_mode SR_Duke --model reid --gpu 2 --no_dropout --SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks
python train_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 1 --no_dropout --SR_name Duke_cyclegan_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name paired_cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_paired_cyclegan_upscale_8
python train_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Duke --model reid --gpu 1 --no_dropout --SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks

python train_reid.py --dataroot Market --name Market_Reid_upscale_8 --dataset_mode Market --model reid --gpu 4 --no_dropout --up_scale 8
python train_reid.py --dataroot Market --name Market_Reid_upscale_8_GT_A --dataset_mode Market --model reid --gpu 4 --no_dropout --up_scale 8
python train_reid.py --dataroot Market --name Market_Reid_attr_upscale_8 --dataset_mode Market --model reid_attr --gpu 3 --no_dropout --up_scale 8
python train_reid.py --dataroot Market --name Market_Reid_attr_upscale_8 --dataset_mode Market --model reid_attr --gpu 3 --no_dropout --up_scale 8 --reid_name Market_Reid_upscale_8
python train_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_SRcCyclegan_upscale_8
python train_reid.py --dataroot Market --name cCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_cCyclegan_upscale_8
python train_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks
python train_reid.py --dataroot Market --name cyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_cyclegan_upscale_8
