from typing import List
from PIL import Image
import torch as t
import torchvision

class ChestDataset(t.utils.data.Dataset):
    def __init__(self, filenames: List[str], labels: List[int]):
        self.filenames = filenames
        self.labels = labels
        self.convert_to_tensor = torchvision.transforms.Compose([torchvision.transforms.Resize([224, 224]),
                                                    torchvision.transforms.ToTensor()])

    def __getitem__(self, idx): # indeksowanie
        filename = self.filenames[idx]
        label = self.labels[idx]

        img = Image.open(filename).convert('L') #L - gray image


        return self.convert_to_tensor(img), t.tensor(label, dtype=t.int32)

    def __len__(self): # nadpisanie metody
        return len(self.filenames)