#!/bin/bash

export PYTHONPATH=..

# 设置变量
framework="ours" # ours single fedmd ssfl fedet
server_model="byolserver" # byolserver single fedmd ssfl fedet
client_type="mix"
encoder_network="resnet18"
data_size="full" #small full
public_type=("iid")
data_partitions=("iid")
rounds=1 #50 100
dir_alpha_values=("0.5")
public_size=1218
gpu="0"
server_epoch=("5") # 5

for sp in "${server_epoch[@]}"; do

  other_args="python main.py --dataset cifar10 --client_model byol --num_of_clients 5 --data_size ${data_size} --encoder_network ${encoder_network} --aggregate_encoder online --update_encoder online --update_predictor global --test_every 5 --batch_size 128 --local_epoch 1 --server_epoch ${sp} --rounds ${rounds} --gpu ${gpu} --public_size ${public_size} --seed 0"

  for public in "${public_type[@]}"; do
    # 循环遍历不同的data_partition值
    for data_partition in "${data_partitions[@]}"; do
      if [ "$data_partition" == "dir" ]; then
        for dir_alpha in "${dir_alpha_values[@]}"; do
          # 构建命令并执行
          cmd="$other_args --framework $framework --server_model $server_model --client_type $client_type --data_partition $data_partition --dir_alpha $dir_alpha --public $public 2>&1 | tee ../logs/${framework}_${client_type}_${server_model}_${data_partition}_${dir_alpha}.log"
          echo "Running: $cmd"
          eval "$cmd"
        done
      else
        # 构建命令并执行
        cmd="$other_args --framework $framework --server_model $server_model --client_type $client_type --data_partition $data_partition --public $public 2>&1 | tee ../logs/${framework}_${client_type}_${server_model}_${data_partition}.log"
        echo "Running: $cmd"
        eval "$cmd"
      fi
    done
  done
done
