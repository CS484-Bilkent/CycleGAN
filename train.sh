CUDA_VISIBLE_DEVICES=4 python scripts/train.py \
    --run-name monet2photo_first \
    --train-dir /home/mert.gencturk/cycle/datasets/monet2photo/train \
    --test-dir /home/mert.gencturk/cycle/datasets/monet2photo/test \
    --learning-rate 0.0002 \
    --adjust-learning-rate 1 \
    --lambda-identity 5 \
    --lambda-cycle 10 \
    --num-workers 2 \
    --num-epochs 200 \
    --load-checkpoints 0 \
    --load-checkpoints-path /home \
    --save-checkpoints 1 \
    --save-checkpoints-epoch 10 \

# for photo<->monet/cezanne etc. set lambda-identity = 5 (0.5)λ where λ is 10 and lambda-cycle = 10 (λ)