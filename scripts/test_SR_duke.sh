python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode single_Duke --model test_SR --gpu 6 --no_dropout --save_phase test --up_scale 8
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8 --dataset_mode Duke --model test_SR --gpu 6 --no_dropout --save_phase train --up_scale 8
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_paired_cCyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks

python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8 --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8 --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A --dataset_mode single_Duke --model test_SR --gpu 5 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A --dataset_mode Duke --model test_SR --gpu 5 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks

python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_upscale_8 --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_SRcCyclegan_upscale_8 --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip --serial_batches

python test_SR.py --dataroot DukeMTMC-reID --name Duke_gan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 3 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_gan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip

python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR --gpu 0 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR --gpu 3 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks_v2 --dataset_mode single_Duke --model test_SR --gpu 5 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks_v2 --dataset_mode Duke --model test_SR --gpu 5 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_SR_attr --gpu 0 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks --dataset_mode Duke --model test_SR_attr --gpu 0 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks_batch_32 --dataset_mode single_Duke --model test_SR_attr --gpu 6 --no_dropout --save_phase test --up_scale 8 --netG resnet_6blocks
python test_SR.py --dataroot DukeMTMC-reID --name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks_batch_32 --dataset_mode Duke --model test_SR_attr --gpu 6 --no_dropout --save_phase train --up_scale 8 --netG resnet_6blocks --no_flip