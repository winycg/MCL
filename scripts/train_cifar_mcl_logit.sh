python main_cifar.py --arch wrn_16_2 --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. --logit-distill
python main_cifar.py --arch wrn_40_2 --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. --logit-distill
python main_cifar.py --arch wrn_28_4 --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. --logit-distill
python main_cifar.py --arch ShuffleNetV2_1x --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. --logit-distill
python main_cifar.py --arch hcgnet_A2 --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. --logit-distill


    