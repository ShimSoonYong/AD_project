import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = "dataset/"
SELECT_NORMAL = 2 # Set 2 class as normal samples. 

def load_dataset(root=ROOT_DIR, transform=transforms.ToTensor()):

    trainset = FashionMNIST(root=root, train=True, download=True, transform=transform) 
    testset = FashionMNIST(root=root, train=False, download=True, transform=transform)

    trainset.data = trainset.data[trainset.targets==SELECT_NORMAL] 
    trainset.targets = trainset.targets[trainset.targets==SELECT_NORMAL]

    test_label = [0,2,4,6] # Define anomaly class 
    actual_testdata = torch.isin(testset.targets, torch.tensor(test_label)) 
    testset.data = testset.data[actual_testdata] 
    testset.targets = testset.targets[actual_testdata]
    
    return trainset, testset

def get_loaders(train_set, test_set, batch_size, num_workers):

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return train_loader, test_loader

if __name__ == "__main__":
    train_set, test_set =  load_dataset()
    print(f"# of train samples: {len(train_set)}, # of test samples: {len(test_set)}")

    train_loader, test_loader = get_loaders(train_set, test_set, batch_size= 32, num_workers=12)
    print(f"# of train minibathces: {len(train_loader)}, # of test minibatches: {len(test_loader)}")
