python train_reid.py --dataroot Market --name Market_Reid_upscale_8 --dataset_mode Market --model reid --gpu 4 --no_dropout --up_scale 8
python train_reid.py --dataroot Market --name Market_Reid_upscale_8_GT_A --dataset_mode Market --model reid --gpu 4 --no_dropout --up_scale 8

python train_reid.py --dataroot Market --name Market_Reid_attr_upscale_8 --dataset_mode Market --model reid_attr --gpu 3 --no_dropout --up_scale 8
python train_reid.py --dataroot Market --name Market_Reid_attr_upscale_8 --dataset_mode Market --model reid_attr --gpu 3 --no_dropout --up_scale 8 --reid_name Market_Reid_upscale_8

python train_reid.py --dataroot Market --name cyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_cyclegan_upscale_8
python train_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_SRcCyclegan_upscale_8
python train_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks
