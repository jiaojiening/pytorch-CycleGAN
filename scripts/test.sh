python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR --dataset_mode single_Duke --model reid --phase test --gpu 0 --no_dropout --NR
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR

python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8 --dataset_mode single_Duke --model reid --gpu 7 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_concate --dataset_mode single_Duke --model reid --gpu 5 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_concate
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_LR --dataset_mode single_Duke --model reid --gpu 7 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_LR

python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_16 --dataset_mode single_Duke --model reid --gpu 0 --no_dropout --up_scale 16
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_16

python test_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Duke --model reid --gpu 3 --no_dropout --SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks

python test_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_cyclegan_upscale_8
python evaluate_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8
python test_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8_LR_B --dataset_mode SR_Duke --model reid --gpu 7 --no_dropout --SR_name Duke_cyclegan_upscale_8
python evaluate_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8_LR_B
python test_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8_GT_A --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A
python evaluate_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8_GT_A
python test_reid.py --dataroot DukeMTMC-reID --name gan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_gan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name gan_Duke_Reid_upscale_8

python test_reid.py --dataroot DukeMTMC-reID --name hybrid_cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 5 --no_dropout --SR_name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name hybrid_cyclegan_Duke_Reid_upscale_8


python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8 --dataset_mode single_Duke --model reid_attr --gpu 7 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8

python test_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Duke --model reid --gpu 2 --no_dropout --SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks

python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0



#python test_reid_SR.py --dataroot Market --name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Market --model test_reid_cyclegan --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
#python evaluate_reid.py --dataroot Market --name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage0

#python test_reid_SR.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Market --model test_reid_SR --gpu 2 --no_dropout --up_scale 8 --netG resnet_6blocks
#python evaluate_reid.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0
