'''
By default, we use --neg_k 8192 as the number of negative samples. 
To save memory costs, we found --neg_k 4096 would achieve similar performance.
'''
python main_imagenet.py --arch resnet18  --number-net 2 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. 
python main_imagenet.py --arch resnet34  --number-net 2 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. 
python main_imagenet.py --arch resnet18  --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1. 
python main_imagenet.py --arch resnet34  --number-net 4 \
    --alpha 0.1 --gamma 1. --beta 0.1 --lam 1.  

