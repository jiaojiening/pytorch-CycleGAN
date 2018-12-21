python test_reid.py --dataroot DukeMTMC-reID --name SR_Duke_Reid_16_lr_0.01 --dataset_mode single_SR_Duke --model reid --gpu 6 --no_dropout --save_phase test --SR_name Duke_cCyclegan
python evaluate_reid.py --dataroot DukeMTMC-reID --name SR_Duke_Reid_16_lr_0.01

python test_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_16_lr_0.01 --dataset_mode single_SR_Duke --model reid --gpu 6 --no_dropout --save_phase test --SR_name Duke_SRcCyclegan
python evaluate_reid.py --dataroot DukeMTMC-reID --name SRcCyclegan_Duke_Reid_16_lr_0.01

python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid --dataset_mode single_Duke --model reid --phase test --gpu 5 --no_dropout --epoch 200
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16 --dataset_mode single_Duke --model reid --phase test --gpu 5 --no_dropout
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16_lr_0.01 --dataset_mode single_Duke --model reid --phase test --gpu 7 --no_dropout
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_16_lr_0.01
python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR_16_lr_0.01 --dataset_mode single_Duke --model reid --phase test --gpu 0 --no_dropout --NR
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_NR_16_lr_0.01

python test_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8 --dataset_mode single_Duke --model reid --gpu 7 --no_dropout --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_Reid_upscale_8

python test_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_16_lr_0.01 --dataset_mode single_SR_Duke --model reid --gpu 0 --no_dropout --save_phase test --SR_name Duke_paired_cCyclegan --up_scale 8
python evaluate_reid.py --dataroot DukeMTMC-reID --name paired_cCyclegan_Duke_Reid_16_lr_0.01