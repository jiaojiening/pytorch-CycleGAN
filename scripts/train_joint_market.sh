python train_joint.py --dataroot Market \
--name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_cyclegan_upscale_8 \
--reid_name cyclegan_Market_Reid_upscale_8 \
--dataset_mode Market --model reid_cycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot Market \
--name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Market_cyclegan_upscale_8 \
--reid_name cyclegan_Market_Reid_upscale_8 \
--dataset_mode Market --model reid_cycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 2 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot Market \
--name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_SRcCycle_gan \
--up_scale 8 --gpu 6,7 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot Market \
--name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_SRcCycle_gan \
--up_scale 8 --gpu 0,1 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2

python train_joint.py --dataroot Market \
--name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_attr_SRcCycle_gan \
--up_scale 8 --gpu 0,7 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot Market \
--name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage2 \
--SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks \
--reid_name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks \
--dataset_mode Market --model reid_attr_SRcCycle_gan \
--up_scale 8 --gpu 1,3 --no_dropout --batch_size 12 --netG resnet_6blocks --stage 2


python train_joint.py --dataroot Market \
--name Market_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_hybrid_cyclegan_upscale_8_resnet_6blocks \
--reid_name hybrid_cyclegan_Market_Reid_upscale_8 \
--dataset_mode Market --model reid_hybrid_cycle_gan \
--up_scale 8 --gpu 0,1,2,3 --no_dropout --batch_size 16 --reid_lr 0.03 --netG resnet_6blocks --stage 0
python train_joint.py --dataroot Market \
--name Market_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_stage0 \
--SR_name Market_hybrid_cCyclegan_upscale_8_resnet_6blocks \
--reid_name hybrid_cCyclegan_Market_Reid_upscale_8 \
--dataset_mode Market --model reid_hybrid_cCycle_gan \
--up_scale 8 --gpu 0,1,6,7 --no_dropout --batch_size 16 --reid_lr 0.03 --netG resnet_6blocks --stage 0