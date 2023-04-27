import torch
import os
import glob
from PIL import Image
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
#import torchvision.transforms as transforms



class Learner(nn.Module):
    def __init__(self, num_classes):
        super(Learner, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# train LSTM-based meta learner  optimizer to optimize learner optimizer. 
class MetaLearner(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MetaLearner, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        # input_seq has shape (batch_size, seq_len, input_size)
        output, _ = self.lstm1(input_seq)
        output, _ = self.lstm2(output)
        output = output[:, -1, :]  # Only use the last output in the sequence
        output = self.linear(output)
        return output

# Hyperparameters
num_classes = 5
num_samples_per_class = 1
num_query_samples_per_class = 5
num_updates = 5
input_size = 32 * 3 * 3  # Size of flattened feature maps
hidden_size = 64
output_size = input_size
lr = 0.001
clip_grad = 0.25

data ="/data/huan1643/miniImagenet"
# Create data loaders for training and testing



class MiniImageNet(data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.label_to_class = {i: c for i, c in enumerate(self.classes)}
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        for i, c in enumerate(self.classes):
            class_path = os.path.join(root, c)
            for image_path in glob.glob(os.path.join(class_path, '*.jpg')):
                self.image_paths.append(image_path)
                self.labels.append(i)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.image_paths)

"""
# Define transform
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Load dataset
train_dataset = MiniImageNet('/data/huan1643/miniImagenet/train', transform=transform)
train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = MiniImageNet('/data/huan1643/miniImagenet/val', transform=transform)
val_dataloader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)
print("done")
"""

 