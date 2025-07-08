export CUDA_VISIBLE_DEVICES=0,1
EXP_NAME="VCTK_p334"
RVC_VERSION="v2"
SAMPLE_RATE="48k"
IF_F0=0 
SPK_ID=0 
BATCH_SIZE=32
NUM_WORKERS=8
python infer/modules/train/preprocess.py "/home/lhy/audio/StreamVC/dataset/VCTK/p334" 48000 12 "logs/${EXP_NAME}" False 3.7

python infer/modules/train/extract_feature_print.py cuda:0 2 0 0 "/home/lhy/audio/Retrieval-based-Voice-Conversion-WebUI/logs/${EXP_NAME}" "${RVC_VERSION}" True &
python infer/modules/train/extract_feature_print.py cuda:1 2 1 1 "/home/lhy/audio/Retrieval-based-Voice-Conversion-WebUI/logs/${EXP_NAME}" "${RVC_VERSION}" True &

wait
echo "创建 config.json..."
python create_config.py "${RVC_VERSION}" "${SAMPLE_RATE}" "${EXP_NAME}"

wait
echo "创建 filelist.txt..."
python create_filelist.py --exp_name "${EXP_NAME}" --rvc_version "${RVC_VERSION}" --if_f0 ${IF_F0} --spk_id ${SPK_ID} --sample_rate ${SAMPLE_RATE}


if [ "${RVC_VERSION}" = "v2" ]; then
    VERSION_SUFFIX="_v2"
else
    VERSION_SUFFIX=""
fi

wait
echo "处理音高..."
python process_f0.py /home/lhy/audio/StreamVC/dataset/dataset/tyzr \
    --output_file /home/lhy/audio/StreamVC/dataset/tyzr/f0_results.json \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --device cuda

# 根据是否带音高确定文件名中的'f0'前缀
if [ "${IF_F0}" = "1" ]; then
    F0_PREFIX="f0"
else
    F0_PREFIX=""
fi

# 构建预训练模型参数
PRETRAINED_G_PATH="assets/pretrained${VERSION_SUFFIX}/${F0_PREFIX}G${SAMPLE_RATE}.pth"
PRETRAINED_D_PATH="assets/pretrained${VERSION_SUFFIX}/${F0_PREFIX}D${SAMPLE_RATE}.pth"

PG_ARG=""
if [ -f "${PRETRAINED_G_PATH}" ]; then
    PG_ARG="-pg ${PRETRAINED_G_PATH}"
else
    echo "警告: 预训练生成器模型未找到于 ${PRETRAINED_G_PATH}"
fi

PD_ARG=""
if [ -f "${PRETRAINED_D_PATH}" ]; then
    PD_ARG="-pd ${PRETRAINED_D_PATH}"
else
    echo "警告: 预训练判别器模型未找到于 ${PRETRAINED_D_PATH}"
fi



wait
python infer/modules/train/train.py \
    -e "${EXP_NAME}" \
    -sr ${SAMPLE_RATE} \
    -f0 0  \
    -bs 12 \
    -g 0,1 \
    -te 200 \
    -se 10 \
    ${PG_ARG} \
    ${PD_ARG} \
    -l 1 \
    -c 0 \
    -sw 0 \
    -v ${RVC_VERSION}

python train_index.py \
    --exp-dir "${EXP_NAME}" \
    --n-cpu 8 \
    --version "${RVC_VERSION}"



# python infer/modules/train/extract/extract_f0_rmvpe.py 2 0 0 "logs/tyzr" True
# python infer/modules/train/extract/extract_f0_rmvpe.py 2 1 1 "logs/tyzr" True