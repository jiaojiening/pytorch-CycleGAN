set -ex
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout

python test.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout

python test.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5



python test.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5

python test.py --dataroot ./datasets/CelebA --name CelebA_SRcCyclegan --dataset_mode CelebA --model SRcCycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5

python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid --dataset_mode single_Duke --model reid --phase test --gpu 5 --no_dropout --batch_size 2


python test_reid.py --dataroot Market --name Market_Reid --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout --batch_size 50

python test_reid.py --dataroot Market --name Market_Reid_16 --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout

python test_reid.py --dataroot Market --name Market_Reid_16_step_20 --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout

python evaluate_reid.py --dataroot Market --name Market_Reid_16