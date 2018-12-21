python train.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode Duke --model SRcCycle_gan --pool_size 50 --gpu 3 --no_dropout
python train.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode Duke --model SRcCycle_gan --pool_size 50 --gpu 7 --no_dropout --batch_size 8
python train.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode Duke --model SRcCycle_gan --pool_size 50 --gpu 0,1 --no_dropout --batch_size 16

python train.py --dataroot DukeMTMC-reID --name Duke_cCyclegan --dataset_mode Duke --model cCycle_gan --pool_size 50 --gpu 1,6 --no_dropout --batch_size 16
python train.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan --dataset_mode paired_Duke --model cCycle_gan --pool_size 50 --gpu 0,1 --no_dropout --batch_size 16

python train.py --dataroot Market --name Market_SRcCyclegan --dataset_mode Market --model SRcCycle_gan --pool_size 50 --gpu 0,1 --no_dropout --batch_size 16 --display_id -1

python train.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode paired_Duke --model cCycle_gan --pool_size 50 --gpu 0,1 --no_dropout --batch_size 14 --up_scale 8