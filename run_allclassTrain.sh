#!/bin/bash

set -e


DATASET=classified_array_exit_0721_dataset_sampled
PRETRAIN_OUT=./output/pretrain
FINETUNE_OUT=./output/finetune


echo "=== Starting pre-training on all classes (augmented) ==="
CUDA_VISIBLE_DEVICES=0 python src/pretrain_allclass.py \
    --batch_size 128 \
    --blr 1e-3 \
    --lr 1e-3 \
    --steps 20000 \
    --mask_ratio 0.9 \
    --data_path ${DATASET} \
    --output_dir ${PRETRAIN_OUT} \
    --log_dir ${PRETRAIN_OUT} \
    --model net_mamba_pretrain \
    --no_amp \
    --input_size 40 \
    --byte_length 1600 \
    --warmup_epochs 25 \
    --loss_mode mae+triplet \
    --alpha 1.0 \
    --beta 0.1

echo "Pre-training completed!"

echo "=== Starting fine-tuning on all classes ==="
CUDA_VISIBLE_DEVICES=0 python src/finetune_allclass.py \
    --blr 2e-3 \
    --epochs 30 \
    --batch_size 64 \
    --category 'normal' \
    --data_path ${DATASET} \
    --finetune ${PRETRAIN_OUT}/checkpoint-step5000.pth \
    --output_dir ${FINETUNE_OUT} \
    --log_dir ${FINETUNE_OUT} \
    --model net_mamba_classifier \
    --no_amp \
    --warmup_epochs 5 \
    --weight_decay 0.05 \
    --input_size 40 \
    --byte_length 1600 \
    --drop_path 0.1

echo "Fine-tuning completed!"

echo "=== Training Summary ==="
if [ -f "${FINETUNE_OUT}/training_report.md" ]; then
    cat ${FINETUNE_OUT}/training_report.md
else
    echo "Report not found!"
fi

echo "=== Files Generated ==="
echo "- Detailed logs: ${FINETUNE_OUT}/log.txt"
echo "- Best model: ${FINETUNE_OUT}/checkpoint-best.pth"
echo "- Test results: ${FINETUNE_OUT}/final_test_results.json"
echo "- Training report: ${FINETUNE_OUT}/training_report.md"
echo "- Confusion matrices: ${FINETUNE_OUT}/confusion_matrix_*.png"
echo "- Metrics plots: ${FINETUNE_OUT}/metrics_per_class_*.png"

echo "All-class training pipeline completed!"