python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR --dataset_mode single_Duke --model reid --phase test --gpu 0 --no_dropout --NR
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8 --dataset_mode single_Duke --model reid --gpu 7 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_GT_A --dataset_mode single_Duke --model reid --gpu 2 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8_GT_A
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8 --dataset_mode single_Duke --model reid_attr --gpu 3 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_attr_upscale_8
python test_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_16_lr_0.01 --dataset_mode SR_Duke --model reid --gpu 3 --no_dropout --SR_name Duke_paired_cCyclegan_upscale_8
python evaluate_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_16_lr_0.01
python test_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_paired_cCyclegan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_upscale_8_resnet_6blocks
python test_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_cyclegan_upscale_8
python evaluate_reid.py --dataroot DukeMTMC-reID --name cyclegan_Duke_Reid_upscale_8
python test_reid.py --dataroot DukeMTMC-reID --name paired_cyclegan_Duke_Reid_upscale_8 --dataset_mode SR_Duke --model reid --gpu 3 --no_dropout --SR_name Duke_paired_cyclegan_upscale_8
python evaluate_reid.py --dataroot DukeMTMC-reID --name paired_cyclegan_Duke_Reid_upscale_8
python test_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Duke --model reid --gpu 0 --no_dropout --SR_name Duke_SRcCyclegan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_upscale_8_resnet_6blocks

python test_reid.py --dataroot Market --name Market_Reid --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout --batch_size 50
python test_reid.py --dataroot Market --name Market_Reid_16 --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout
python evaluate_reid.py --dataroot Market --name Market_Reid_16
python test_reid.py --dataroot Market --name Market_Reid_16_step_20 --dataset_mode single_Market --model reid --phase test --gpu 5 --no_dropout
python test_reid.py --dataroot Market --name Market_Reid_16_lr_0.01 --dataset_mode single_Market --model reid --phase test --gpu 7 --no_dropout
python evaluate_reid.py --dataroot Market --name Market_Reid_16_lr_0.01
python test_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01 --dataset_mode single_Market --model reid --phase test --gpu 7 --no_dropout --NR
python evaluate_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01

python test_reid.py --dataroot Market --name Market_Reid_upscale_8 --dataset_mode single_Market --model reid --gpu 4 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot Market --name Market_Reid_upscale_8
python test_reid.py --dataroot Market --name Market_Reid_upscale_8_GT_A --dataset_mode single_Market --model reid --gpu 2 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot Market --name Market_Reid_upscale_8_GT_A
python test_reid.py --dataroot Market --name Market_Reid_attr_upscale_8 --dataset_mode single_Market --model reid_attr --gpu 3 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot Market --name Market_Reid_attr_upscale_8
python test_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_SRcCyclegan_upscale_8
python evaluate_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8
python test_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks
python test_reid.py --dataroot Market --name cCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_cCyclegan_upscale_8
python evaluate_reid.py --dataroot Market --name cCyclegan_Market_Reid_upscale_8
python test_reid.py --dataroot Market --name cyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_cyclegan_upscale_8
python evaluate_reid.py --dataroot Market --name cyclegan_Market_Reid_upscale_8