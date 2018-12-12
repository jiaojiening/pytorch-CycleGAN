python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan --dataset_mode single_Duke --model test_reid_SR --gpu 0 --no_dropout --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan

python test_reid_SR.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan_24 --dataset_mode single_Duke --model test_reid_SR --gpu 7 --no_dropout --netG resnet_6blocks
python evaluate_reid.py --dataroot DukeMTMC-reID --name Duke_ReidSRcCyclegan_24

python test_reid_SR.py --dataroot Market --name Market_ReidSRcCyclegan_24 --dataset_mode single_Market --model test_reid_SR --gpu 7 --no_dropout --netG resnet_6blocks
python evaluate_reid.py --dataroot Market --name Market_ReidSRcCyclegan_24