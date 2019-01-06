set -ex
python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --lambda_identity 0 --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
python train.py --dataroot ./datasets/CelebA --name CelebA_SRcCyclegan --dataset_mode CelebA --model SRcCycle_gan --pool_size 50 --loadSize 158 --fineSize 128 --gpu 3 --no_dropout
