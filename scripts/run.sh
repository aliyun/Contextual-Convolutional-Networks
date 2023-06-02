NUM_GPUS=1
ADDR=localhost
PORT=1234
NUM_NODE=1
NODE_RANK=0

###### Training ######
# Ctx ResNet-50
ARGS="datasets/imagenet -b 256 --model ctx_r50 --epochs 200 --lr 1e-4 --amp --pin-mem"

# Ctx ConvNeXt Tiny
# ARGS="datasets/imagenet -b 2 --model ctx_convnext_tiny --epochs 300 --lr 2e-5 --weight-decay 5e-2 --amp --pin-mem --dyn-block-freq 3 --remode pixel --reprob 0.25 --drop-path 0.1 --mixup 0.8 --cutmix 1.0"

# To train on pretrained ckpt, attach "--initial-checkpoint <path/to/pretrained_ckpt>" to $ARGS; 
# To use the default pretrained ckpt of timm, attach "--pretrained" to $ARGS


###### Evaluation ######
# For evaluation, attach "--initial-checkpoint <path/to/test_ckpt> --evaluate" to $ARGS


torchrun --nproc_per_node=$NUM_GPUS --master_addr $ADDR --master_port $PORT --nnodes $NUM_NODE --node_rank $NODE_RANK main.py $ARGS

