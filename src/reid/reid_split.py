from collections import defaultdict
import random

def split_train_by_pid(train_list, num_clients, seed=1234):
    """
    train_list: list of dataset.train items (img_path, pid, camid, viewid)
    returns: list[list[item]] length=num_clients
    """
    pid2items = defaultdict(list)
    for item in train_list:
        pid = item[1]
        pid2items[pid].append(item)

    pids = list(pid2items.keys())
    random.Random(seed).shuffle(pids)

    client_lists = [[] for _ in range(num_clients)]
    for i, pid in enumerate(pids):
        client_lists[i % num_clients].extend(pid2items[pid])

    return client_lists