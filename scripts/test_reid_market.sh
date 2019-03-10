python test_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01 --dataset_mode single_Market --model reid --phase test --gpu 7 --no_dropout --NR
python evaluate_reid.py --dataroot Market --name Market_Reid_NR_16_lr_0.01

python test_reid.py --dataroot Market --name Market_Reid_upscale_8 --dataset_mode single_Market --model reid --gpu 4 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot Market --name Market_Reid_upscale_8
python test_reid.py --dataroot Market --name Market_Reid_upscale_8_GT_A --dataset_mode single_Market --model reid --gpu 2 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot Market --name Market_Reid_upscale_8_GT_A

python test_reid.py --dataroot Market --name Market_Reid_attr_upscale_8 --dataset_mode single_Market --model reid_attr --gpu 3 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot Market --name Market_Reid_attr_upscale_8

python test_reid.py --dataroot Market --name cyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_cyclegan_upscale_8
python evaluate_reid.py --dataroot Market --name cyclegan_Market_Reid_upscale_8
python test_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 0 --no_dropout --SR_name Market_SRcCyclegan_upscale_8
python evaluate_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8
python test_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks --dataset_mode SR_Market --model reid --gpu 2 --no_dropout --SR_name Market_SRcCyclegan_upscale_8_resnet_6blocks
python evaluate_reid.py --dataroot Market --name SRcCyclegan_Market_Reid_upscale_8_resnet_6blocks

python test_reid.py --dataroot Market --name LapSRN_Market_Reid_upscale_8 --dataset_mode SR_Market --model reid --gpu 5 --no_dropout --SR_name LapSRN_market --results_dir /home/share/jiening/dgd_datasets/raw --up_scale 8
python evaluate_reid.py --dataroot Market --results_dir /home/share/jiening/dgd_datasets/raw --name LapSRN_Market_Reid_upscale_8