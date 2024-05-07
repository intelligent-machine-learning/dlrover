export WANDB_DISABLED=true

if [ ! -d openwebtext/subsets/ ]; then
    bash prepare_dataset.sh
fi

if [ ! -d /tmp ]; then
    mkdir /tmp
fi

NUM_GPUS=$(nvidia-smi -L | wc -l)

if [ ! -d $1 ]; then mkdir -p $1; fi;
cp -r $0 ${1}

nvidia-smi >> ${1}/nanoGPT.log
printenv >> ${1}/nanoGPT.log

python -m atorch.distributed.launch \
    --nproc_per_node $NUM_GPUS \
    --master_port 20456 \
    train_atorch.py 2>&1 |tee -a $3/nanoGPT.log
