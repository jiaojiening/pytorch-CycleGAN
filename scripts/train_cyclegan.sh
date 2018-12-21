set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --lambda_identity 0 --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_SRcCyclegan --dataset_mode CelebA --model SRcCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout



python train_reid.py --dataroot Market --name Market_Reid_16 --dataset_mode Market --model reid --gpu 7 --no_dropout --batch_size 16
python train_reid.py --dataroot Market --name Market_Reid_16_lr_0.01 --dataset_mode Market --model reid --gpu 7 --no_dropout --batch_size 16 --reid_lr 0.01

python train_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01 --dataset_mode Market --model reid --gpu 3 --no_dropout --batch_size 16 --reid_lr 0.01 --NR

python train.py --dataroot Market --name Market_ReidSRcCyclegan --dataset_mode Market --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 16 --netG resnet_6blocks
python train.py --dataroot Market --name Market_ReidSRcCyclegan --dataset_mode Market --model reid_SRcCycle_gan --pool_size 50 --gpu 1,2 --no_dropout --batch_size 14 --netG resnet_6blocks --reid_lr 0.01