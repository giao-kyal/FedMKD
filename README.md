# FedMKD + ReID

python main.py --dataset market1501 --reid_root /data1/dongwenhao/projects/FedMKD/data --test_every 2 --rounds 100 --local_epoch 5 --server_epoch 1 --num_of_clients 5 --batch_size 64 --gpu 3 4 5 6 7 --reid_debug_eval_clients --reid_debug_eval_client_num 3

python main.py --dataset market1501 --reid_root /mnt/d/FedMKD/data --rounds 2 --local_epoch 1 --server_epoch 1 --num_of_clients 1
