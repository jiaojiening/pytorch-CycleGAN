python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks --dataset_mode single_Duke --model test_reid_SR --gpu 4 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_SR --gpu 2 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage1 --dataset_mode single_Duke --model test_reid_SR --gpu 3 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage1
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage2 --dataset_mode single_Duke --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_stage2
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage0 --dataset_mode single_Duke --model test_reid_SR --gpu 3 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage1 --dataset_mode single_Duke --model test_reid_SR --gpu 3 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage1
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage1_l1 --dataset_mode single_Duke --model test_reid_SR --gpu 2 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_paired_cCyclegan_upscale_8_resnet_6blocks_use_feat_stage1_l1

python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2 --dataset_mode single_Duke --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_all --dataset_mode single_Duke --model test_reid_SR --gpu 3 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_all
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_all_HR --dataset_mode single_Duke --model test_reid_SR --gpu 2 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_all_HR

python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_cyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_cyclegan --gpu 2 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_cyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_cyclegan_upscale_8_resnet_6blocks_stage0_all --dataset_mode single_Duke --model test_reid_cyclegan --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_cyclegan_upscale_8_resnet_6blocks_stage0_all
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_cyclegan_upscale_8_resnet_6blocks_stage0_lr_0.1 --dataset_mode single_Duke --model test_reid_cyclegan --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_cyclegan_upscale_8_resnet_6blocks_stage0_lr_0.1

python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_attr_paired_cCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_attr_paired_cCyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_attr_paired_cCyclegan_upscale_8_resnet_6blocks_stage2 --dataset_mode single_Duke --model test_reid_SR --gpu 4 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_attr_paired_cCyclegan_upscale_8_resnet_6blocks_stage2
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_SR --gpu 5 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage2 --dataset_mode single_Duke --model test_reid_SR --gpu 4 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage2

python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_cyclegan --gpu 7 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_rec_20_stage0 --dataset_mode single_Duke --model test_reid_cyclegan --gpu 7 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_rec_20_stage0
python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Duke --model test_reid_SR --gpu 7 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_stage0