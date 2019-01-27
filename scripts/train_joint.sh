python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage1 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_cCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 1

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_cCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage1 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_cCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --use_feat --lambda_feat 0.0001 --batch_size 12 --netG resnet_6blocks --stage 1

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage1_l1 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_cCycle_gan \
--up_scale 8 --gpu 0,2 --no_dropout --use_feat --batch_size 12 --netG resnet_6blocks --stage 1

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --reid_lr 0.02 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_all \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 12 --reid_lr 0.1 --netG resnet_6blocks --stage 0

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage1 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 1

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_attr_SRcCycle_gan \
--up_scale 8 --gpu 1,4 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_attr_SRcCycle_gan \
--up_scale 8 --gpu 2,5 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_cyclegan_upscale_8_resnet_6blocks_GT_A_stage0 \
--SR_name Duke_cyclegan_upscale_8_resnet_6blocks_GT_A \
--reid_name cyclegan_Duke_Reid_upscale_8_GT_A \
--dataset_mode Duke --model reid_cycle_gan \
--up_scale 8 --gpu 2,3 --no_dropout --batch_size 10 --reid_lr 0.02 --netG resnet_6blocks --display_id -1 --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_cyclegan_upscale_8_resnet_6blocks_stage0_all \
--SR_name Duke_cyclegan_upscale_8 \
--reid_name cyclegan_Duke_Reid_upscale_8 \
--dataset_mode Duke --model reid_cycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 12 --reid_lr 0.1 --netG resnet_6blocks --stage 0

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_attr_paired_cCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_attr_cCycle_gan \
--up_scale 8 --gpu 1,2 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_attr_paired_cCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_attr_cCycle_gan \
--up_scale 8 --gpu 0,7 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 0,7 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_cCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2


python train_joint.py --dataroot Market \
--name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_SRcCycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot Market \
--name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_attr_SRcCycle_gan \
--up_scale 8 --gpu 1,3 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot Market \
--name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_attr_SRcCycle_gan \
--up_scale 8 --gpu 0,7 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0

python train_joint.py --dataroot Market \
--name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_cyclegan_upscale_8 \
--reid_name cyclegan_Market_Reid_upscale_8 \
--dataset_mode Market --model reid_cycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0

python train_joint.py --dataroot Market \
--name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_SRcCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot Market \
--name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage2 \   ``
--SR_name Market_cyclegan_upscale_8 \
--reid_name cyclegan_Market_Reid_upscale_8 \
--dataset_mode Market --model reid_cycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 2 --netG resnet_6blocks --stage 2

