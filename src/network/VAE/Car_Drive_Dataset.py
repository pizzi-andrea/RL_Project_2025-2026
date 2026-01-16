import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
class CDD (Dataset):
    def __init__(self, base_path):
        super().__init__()
        self.paths = []
        self.transformation = Compose([
            Resize((64,64)),
            ToTensor(),
        ]) 
        for item in Path(base_path).glob('*'):
            if item.is_dir():
                self.paths.extend([item.joinpath(name) for name in os.listdir(item)])
    

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transformation(image)

    def __len__(self):
        return len(self.paths)
