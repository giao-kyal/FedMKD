from PIL import Image
from torch.utils.data import Dataset

class PublicReIDSSL(Dataset):
    def __init__(self, train_list, transform):
        self.items = train_list
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx][0]
        img = Image.open(img_path).convert("RGB")
        out = self.transform(img) if self.transform is not None else img

        if isinstance(out, (tuple, list)) and len(out) == 2:
            x1, x2 = out
        else:
            x1 = out
            x2 = self.transform(img) if self.transform is not None else img

        return (x1, x2), 0