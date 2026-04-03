def remap_pids(train_list):
    unique = sorted({item[1] for item in train_list})
    pid_map = {pid: i for i, pid in enumerate(unique)}
    new_list = []
    for img_path, pid, camid, viewid in train_list:
        new_list.append((img_path, pid_map[pid], camid, viewid))
    return new_list, pid_map, len(unique)