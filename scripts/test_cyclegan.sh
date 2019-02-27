set -ex
python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
python test.py --dataroot ./datasets/CelebA --name CelebA_cyclegan --dataset_mode CelebA --model cycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout
python test.py --dataroot ./datasets/CelebA --name CelebA_cCyclegan --dataset_mode CelebA --model cCycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5
python test.py --dataroot ./datasets/CelebA --name CelebA_SRcCyclegan --dataset_mode CelebA --model SRcCycle_gan --phase test --loadSize 158 --fineSize 128 --gpu 2 --no_dropout --epoch 5

python test_SR.py --dataroot DukeMTMC-reID --name Duke_cCyclegan --dataset_mode single_Duke --model test_SR --phase test --gpu 6 --no_dropout
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cCyclegan --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train

python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8 --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8 --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cyclegan_upscale_8 --dataset_mode single_Duke --model test_SR --gpu 0 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cyclegan_upscale_8 --dataset_mode Duke --model test_SR --gpu 0 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks

python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train

python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test --up_scale 8
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train --up_scale 8
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks

python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A --dataset_mode single_Duke --model test_SR --gpu 5 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A --dataset_mode Duke --model test_SR --gpu 5 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks

python test_SR.py --dataroot DukeMTMC-reID --name Duke_gan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_gan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip

python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 0 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip

python test_SR.py --dataroot Market --name Market_SRcCyclegan --dataset_mode single_Market --model test_SR --gpu 6 --no_dropout
python test_SR.py --dataroot Market --name Market_SRcCyclegan --dataset_mode Market --model test_SR --gpu 6 --no_dropout --save_phase train
python test_SR.py --dataroot Market --name Market_SRcCyclegan_upscale_8 --dataset_mode single_Market --model test_SR --gpu 0 --no_dropout --up_scale 8
python test_SR.py --dataroot Market --name Market_SRcCyclegan_upscale_8 --dataset_mode Market --model test_SR --gpu 0 --no_dropout --save_phase train --up_scale 8
python test_SR.py --dataroot Market --name Market_SRcCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Market --model test_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot Market --name Market_SRcCyclegan_upscale_8_resnet_6blocks --dataset_mode Market --model test_SR --gpu 0 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot Market --name Market_cCyclegan_upscale_8 --dataset_mode single_Market --model test_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot Market --name Market_cCyclegan_upscale_8 --dataset_mode Market --model test_SR --gpu 0 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot Market --name Market_cyclegan_upscale_8 --dataset_mode single_Market --model test_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot Market --name Market_cyclegan_upscale_8 --dataset_mode Market --model test_SR --gpu 0 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks


