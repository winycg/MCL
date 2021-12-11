python main_imagenet.py \
    --arch resnet18  \
    --number-net 3 \
     --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. \
    --data /dev/shm/ImageNet \
    --gpu-id 0 \
    --lr-type multistep \
    --logit-distill \
    --manual_seed 1019



