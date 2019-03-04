python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks \
--reid_name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode paired_Duke --model reid_cCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 12 --reid_lr 0.03 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_all \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 12 --reid_lr 0.1 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_all_HR \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 1,2 --no_dropout --batch_size 12 --reid_lr 0.01 --netG resnet_6blocks --stage 0

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage1 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 1
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks \
--dataset_mode Duke --model reid_SRcCycle_gan \
--up_scale 8 --gpu 0,7 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

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
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --reid_lr 0.1 --netG resnet_6blocks --stage 0

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
--name Duke_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks \
--reid_name hybrid_cyclegan_Duke_Reid_upscale_8 \
--dataset_mode Duke --model reid_hybrid_cycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 10 --reid_lr 0.02 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks \
--reid_name hybrid_cyclegan_Duke_Reid_upscale_8 \
--dataset_mode Duke --model reid_hybrid_cycle_gan \
--up_scale 8 --gpu 0,1,2,3 --no_dropout --batch_size 16 --reid_lr 0.02 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_v2_stage0 \
--SR_name Duke_hybrid_cyclegan_upscale_8_resnet_6blocks_v2 \
--reid_name hybrid_cyclegan_v2_Duke_Reid_upscale_8 \
--dataset_mode Duke --model reid_hybrid_cycle_gan \
--up_scale 8 --gpu 0,1,2,3 --no_dropout --batch_size 16 --reid_lr 0.03 --netG resnet_6blocks --stage 0

python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks \
--reid_name hybrid_cCyclegan_Duke_Reid_upscale_8 \
--dataset_mode Duke --model reid_hybrid_cCycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 10 --reid_lr 0.02 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks \
--reid_name hybrid_cCyclegan_Duke_Reid_upscale_8 \
--dataset_mode Duke --model reid_hybrid_cCycle_gan \
--up_scale 8 --gpu 0,1,2,3 --no_dropout --batch_size 16 --reid_lr 0.03 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot DukeMTMC-reID \
--name Duke_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_batch_32_stage0 \
--SR_name Duke_hybrid_cCyclegan_upscale_8_resnet_6blocks_batch_32 \
--reid_name hybrid_cCyclegan_batch_32_Duke_Reid_upscale_8 \
--dataset_mode Duke --model reid_hybrid_cCycle_gan \
--up_scale 8 --gpu 0,1,2,3 --no_dropout --batch_size 16 --reid_lr 0.03 --netG resnet_6blocks --stage 0