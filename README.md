# FedMKD + ReID

python main.py --dataset market1501 --reid_root /data1/dongwenhao/projects/FedMKD/data --test_every 5 --rounds 20 --local_epoch 60 --server_epoch 1 --num_of_clients 5 --batch_size 64 --gpu 0 1 2 3 4 --reid_debug_eval_clients --reid_debug_eval_client_num 3

python main.py --dataset market1501 --reid_root /mnt/d/FedMKD/data --rounds 2 --local_epoch 1 --server_epoch 1 --num_of_clients 1

python main.py --dataset market1501 --reid_root D:\FedMKD\data --test_every 5 --rounds 100 --local_epoch 5 --server_epoch 1 --num_of_clients 5 --batch_size 64 --gpu 0

python eval_all_clients.py \
  --run_dir logs/market1501_ours_mix_byolserver_5_resnet18_60_1_20_64 \
  --task_id market1501_ours_mix_byolserver_5_resnet18_60_1_20_64 \
  --dataset market1501 \
  --reid_root /data1/dongwenhao/projects/FedMKD/data \
  --num_clients 5