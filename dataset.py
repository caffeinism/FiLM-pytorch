import torch
import torch.nn as nn
import random
import torchvision.transforms as transforms
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, qas, transform=None):
        assert all(images.size(0) == qa.size(0) for qa in qas)
        self.images = images
        self.qas = qas
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        
        return tuple([image, *[qa[index] for qa in self.qas]])

    def __len__(self):
        return self.images.size(0)
    
class RandomRotate90:
    def __init__(self):
        self.transforms = [
            lambda x: x,
            lambda x: x.transpose(1, 2).flip(1),
        ]
        
    def __call__(self, tensor):
        p = random.randint(0, 1)
                
        return self.transforms[p](tensor)

def list_to_tensor(lst, dtype):
    return torch.from_numpy(np.array(lst, dtype=dtype))

augmenation_transform = transforms.Compose([
    RandomRotate90(),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

def make_dataset(raw_dataset, rel_augmentation=False):
    image, rel_qas, nonrel_qas = zip(*raw_dataset)
    rel_qs, rel_as = zip(*rel_qas)
    nonrel_qs, nonrel_as = zip(*nonrel_qas)
    
    rel_qs, nonrel_qs = list_to_tensor(rel_qs, dtype=np.float32), list_to_tensor(nonrel_qs, dtype=np.float32)
    rel_as, nonrel_as = list_to_tensor(rel_as, dtype=np.int64), list_to_tensor(nonrel_as, dtype=np.int64)
    image = list_to_tensor(image, dtype=np.float32).permute(0, 3, 1, 2)
    
    image = image.unsqueeze(1).repeat(1, rel_qs.size(1), 1, 1, 1).view(image.size(0) * rel_qs.size(1), image.size(1), image.size(2), image.size(3))
    rel_qs = rel_qs.view(rel_qs.size(0) * rel_qs.size(1), rel_qs.size(2))
    rel_as = rel_as.view(rel_as.size(0) * rel_as.size(1))
    nonrel_qs = nonrel_qs.view(nonrel_qs.size(0) * nonrel_qs.size(1), nonrel_qs.size(2))
    nonrel_as = nonrel_as.view(nonrel_as.size(0) * nonrel_as.size(1))
    
    rel_dataset = Dataset(image, [rel_qs, rel_as], None if rel_augmentation else augmenation_transform)
    nonrel_dataset = Dataset(image, [nonrel_qs, nonrel_as])

    return rel_dataset, nonrel_dataset