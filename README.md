# FedMKD + ReID

python main.py --dataset market1501 --reid_root /data1/dongwenhao/projects/FedMKD/data --test_every 5 --rounds 20 --local_epoch 60 --server_epoch 1 --num_of_clients 5 --batch_size 64 --gpu 0 1 2 6 7 --reid_debug_eval_clients --reid_debug_eval_client_num 5

python main.py --dataset market1501 --reid_root /mnt/d/FedMKD/data --rounds 2 --local_epoch 1 --server_epoch 1 --num_of_clients 1

python main.py --dataset market1501 --reid_root D:\FedMKD\data --test_every 5 --rounds 100 --local_epoch 5 --server_epoch 1 --num_of_clients 5 --batch_size 64 --gpu 0

git status
git pull --ff-only origin main
