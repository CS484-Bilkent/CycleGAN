CUDA_VISIBLE_DEVICES=1
python scripts/train.py \
    --num-workers 2 \
    --save-checkpoints 1 \
    --load-checkpoints 0 \
    --load-checkpoints-path /home/mert.gencturk/cycle/CycleGAN/checkpoints/old_horse2zebra \
    --save-checkpoints-epoch 10 \
    --train-dir /home/mert.gencturk/cycle/datasets/apple2orange/train \
    --test-dir /home/mert.gencturk/cycle/datasets/apple2orange/test \
    --lambda-identity 0 \
    --lambda-cycle 10 \
    --learning-rate 0.0002 \
    --num-epochs 150 \
    --run-name apple2orange_first