
python main_mcl_gan.py --arch mcl_okd_resnet32_n --aa 0.5 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 2 --manual_seed 9000 


python main_mcl_gan.py --arch mcl_okd_resnet32_n --aa 0.5 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 2 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_resnet56_n --aa 0.5 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 0 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_resnet110_bottleneck_n --aa 0.5 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 0 --manual_seed 0

python main_gan_mcl_10.py --arch mcl_okd_resnet32_n --aa 0.5 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 2 --manual_seed 2 # 没force va_0 vb_0

python main_mcl_gan.py --arch mcl_okd_wrn_16_2_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 1 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_wrn_40_2_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 1 --manual_seed 0

python main_mcl_gan.py --arch mcl_okd_resnet32_n --aa 0.5 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill None --gpu 2 --num-branches 4 --manual_seed 0


python main_mcl_gan.py --arch mcl_okd_wrn_16_2_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill ONE --gpu 0 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_wrn_40_2_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill ONE --gpu 0 --manual_seed 0

python main_mcl_gan.py --arch mcl_okd_resnet32_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill ONE --gpu 1 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_resnet56_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill ONE --gpu 1 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_resnet110_bottleneck_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 1 --distill ONE --gpu 1 --manual_seed 0


python main_mcl_gan.py --arch mcl_okd_wrn_16_2_b --aa 0.1 --bb 0. --cc 0. --dd 0. --ff 0. --distill None --num-branches 4 --gpu 0 --manual_seed 1
python main_mcl_gan.py --arch mcl_okd_wrn_16_2_b --aa 0.1 --bb 1. --cc 0. --dd 0. --ff 0. --distill None --num-branches 4 --gpu 0 --manual_seed 1
python main_mcl_gan.py --arch mcl_okd_wrn_16_2_b --aa 0. --bb 0. --cc 0.1 --dd 0. --ff 0. --distill None --num-branches 4 --gpu 0 --manual_seed 1
python main_mcl_gan.py --arch mcl_okd_wrn_16_2_b --aa 0. --bb 0. --cc 0.1 --dd 1. --ff 0. --distill None --num-branches 4 --gpu 0 --manual_seed 1
python main_mcl_gan.py --arch mcl_okd_wrn_16_2_b --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0. --distill None --num-branches 4 --gpu 0 --manual_seed 1
python main_mcl_gan.py --arch mcl_okd_hcgnet_A1_b --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0. --distill None --num-branches 4 --gpu 2 --manual_seed 1

python main_mcl_gan.py --arch mcl_okd_resnet32_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0 --distill None --gpu 1 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_wrn_16_2_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0 --distill None --gpu 1 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_resnet56_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0 --distill None --gpu 0 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_resnet110_bottleneck_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0 --distill None --gpu 1 --manual_seed 0


python main_mcl_gan.py --arch mcl_okd_resnet32_n --aa 0. --bb 0. --cc 0. --dd 0. --ff 0. --distill None --gpu 2 --manual_seed 2


python main_mcl_gan.py --arch mcl_okd_vgg13_n --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0 --distill None --gpu 0 --manual_seed 0
python main_mcl_gan.py --arch mcl_okd_vgg13_b --aa 0.1 --bb 1. --cc 0.1 --dd 1. --ff 0 --distill None --gpu 1 --manual_seed 0 --num-branches 4




wrn_16_2 wrn_40_2 wrn_28_4   ShuffleNetV2_05x 
python main_cifar.py --arch hcgnet_A2  \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. \
    --data /home/ycg/hhd/dataset/ \
    --number-net 4 \
    --epochs 300 \
    --logit-distill --gpu 0 --manual_seed 0

python main_cifar.py --arch resnet32  \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. \
    --data /home/ycg/hhd/dataset/ \
    --number-net 4 \
    --epochs 300 \
    --logit-distill --gpu 1 --manual_seed 0
