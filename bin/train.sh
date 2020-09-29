# 训练数据
Date=$(date +%Y%m%d%H%M)
export CUDA_VISIBLE_DEVICES=0

log_dir="logs"

if [ ! -d "$log_dir" ]; then
        mkdir $log_dir
fi

if [ "$1" == "console" ];then
   python -m mask_train
exit
fi
nohup \
python -m mask_train \
>> ./logs/console_$Date.log 2>&1 &
echo "启动完毕,在logs下查看日志！"