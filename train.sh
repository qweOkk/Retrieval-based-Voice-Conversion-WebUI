export CUDA_VISIBLE_DEVICES=7
python infer/modules/train/preprocess.py "dataset/tyzr" 40000 12 "logs/tyzr" False 3.7
python infer/modules/train/extract/extract_f0_rmvpe.py 2 0 0 "logs/tyzr" True
python infer/modules/train/extract/extract_f0_rmvpe.py 2 1 0 "logs/tyzr" True
python infer/modules/train/train.py -e "tyzr" -sr 40k \
    -f0 1 \
    -bs 12 \
    -g 0 \
    -te 100 \
    -se 5 \
    -pg assets/pretrained_v2/f0G40k.pth \
    -pd assets/pretrained_v2/f0D40k.pth \
    -l 1 -c 0 -sw 0 -v v2
python train_index.py \
    --exp-dir tyzr \
    --n-cpu 8 \
