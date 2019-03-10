python test_reid_SR.py --dataroot Market --name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Market --model test_reid_cyclegan --gpu 3 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_cyclegan_upscale_8_resnet_6blocks_stage0

python test_reid_SR.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Market --model test_reid_SR --gpu 2 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_lr_0.1 --dataset_mode single_Market --model test_reid_SR --gpu 3 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage0_lr_0.1
python test_reid_SR.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2 --dataset_mode single_Market --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_SRcCyclegan_upscale_8_resnet_6blocks_stage2

python test_reid_SR.py --dataroot Market --name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Market --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot Market --name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage2 --dataset_mode single_Market --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_attr_SRcCyclegan_upscale_8_resnet_6blocks_stage2

python test_reid_SR.py --dataroot Market --name Market_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Market --model test_reid_cyclegan --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_stage0
python test_reid_SR.py --dataroot Market --name Market_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_v2_stage0 --dataset_mode single_Market --model test_reid_cyclegan --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_hybrid_cyclegan_upscale_8_resnet_6blocks_v2_stage0
python test_reid_SR.py --dataroot Market --name Market_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_stage0 --dataset_mode single_Market --model test_reid_SR --gpu 0 --no_dropout --up_scale 8 --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_reid_hybrid_cCyclegan_upscale_8_resnet_6blocks_stage0