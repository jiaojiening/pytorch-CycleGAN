set -ex
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
python test.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout
python test.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5
python test.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5
python test.py --dataroot ./datasets/CelebA --name CelebA_SRcCyclegan --dataset_mode CelebA --model SRcCycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5

python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_1 --dataset_mode Duke --model SRcCycle_gan --phase test --gpu 4 --no_dropout
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_1 --dataset_mode single_Duke --model test_SR --phase test --gpu 6 --no_dropout
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode single_Duke --model test_SR --phase test --gpu 6 --no_dropout
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cCyclegan --dataset_mode single_Duke --model test_SR --phase test --gpu 6 --no_dropout
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cCyclegan --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train

python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train

python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train

python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test --up_scale 8
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train --up_scale 8




python test_SR.py --dataroot Market --name Market_SRcCyclegan --dataset_mode single_Market --model test_SR --phase test --gpu 6 --no_dropout
python test_SR.py --dataroot Market --name Market_SRcCyclegan --dataset_mode Market --model test_SR --gpu 6 --no_dropout --save_phase train

python test_reid.py --dataroot Market --name Market_Reid --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout --batch_size 50
python test_reid.py --dataroot Market --name Market_Reid_16 --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout
python evaluate_reid.py --dataroot Market --name Market_Reid_16
python test_reid.py --dataroot Market --name Market_Reid_16_step_20 --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout

python test_reid.py --dataroot Market --name Market_Reid_16_lr_0.01 --dataset_mode single_Market --model reid --phase test --gpu 7 --no_dropout
python evaluate_reid.py --dataroot Market --name Market_Reid_16_lr_0.01

python test_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01 --dataset_mode single_Market --model reid --phase test --gpu 7 --no_dropout --NR
python evaluate_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01


