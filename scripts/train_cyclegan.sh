set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --lambda_identity 0 --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_SRcCyclegan --dataset_mode CelebA --model SRcCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout


python train.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode Duke --model SRcCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout

python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --batch_size 2

python train.py --dataroot DukeMTMC-reID --name Duke_Reid --dataset_mode Duke --model reid --pool_size 50  --loadSize 158 --fineSize 128 --gpu 5 --no_dropout --batch_size 32

python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid --dataset_mode Duke --model reid --pool_size 50 --gpu 5 --no_dropout --batch_size 32

python train_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16 --dataset_mode Duke --model reid --pool_size 50 --gpu 5 --no_dropout --batch_size 16


python train.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_test --dataset_mode Duke --model SRcCycle_gan --pool_size 50 --gpu 0,2 --no_dropout --batch_size 16

python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks

python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan_step_20 --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks

python train.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan --dataset_mode Duke --model reid_SRcCycle_gan --pool_size 50 --gpu 4,5,6,7 --no_dropout --batch_size 24 --netG resnet_6blocks --display_id -1


python train.py --dataroot Market --name Market_Reid_16 --dataset_mode Market --model reid --gpu 7 --no_dropout --batch_size 16

python train.py --dataroot Market --name Market_ReidSRcCyclegan --dataset_mode Market --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks