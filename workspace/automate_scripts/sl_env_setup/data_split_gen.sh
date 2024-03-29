#!/usr/bin/env bash
# TODO: specify the path to your dataset
DATASET_PATH=""
OUTPUT_PATH=""

if [ ! -f "${DATASET_PATH}" ]
then
    echo "Please check if you saved HIGGS dataset in ${DATASET_PATH}"
fi

echo "Generated data splits, reading from ${DATASET_PATH}"
#TODO modify the arguments to fit your needs
for site_num in 2 5 20;
do
    for split_mode in uniform exponential square;
    do
        python3 utils/prepare_data_split.py \
        --data_path "${DATASET_PATH}" \
        --site_num ${site_num} \
        --size_total 11000000 \
        --size_valid 1000000 \
        --split_method ${split_mode} \
        --out_path "${OUTPUT_PATH}/${site_num}_${split_mode}"
    done
done
echo "Data splits are generated in ${OUTPUT_PATH}"