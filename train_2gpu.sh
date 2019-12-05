CUDA_VISIBLE_DEVICES=0,1 python \
        -m torch.distributed.launch \
        --nproc_per_node=2 \
        tools/train_net.py \
        --config configs/fcos/fcos_imprv_R_50_FPN_1x_pseudo.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR "/home/mengqinj/capstone/output/stage1_from_scratch/" \


