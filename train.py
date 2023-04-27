import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MiniImageNet
from models import Learner, MetaLearner
from utils import calculate_accuracy
from torchvision import transforms


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data folder path
data ='/data/huan1643/miniImagenet'

# Define hyperparameters
num_shots = 1
num_ways = 5
num_epochs = 2
meta_batch_size = 32
inner_learning_rate = 0.4
meta_learning_rate = 0.001
num_classes = len("data/train")

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define datasets and dataloaders
train_dataset = MiniImageNet(data, 'train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=num_ways*num_shots, shuffle=True)

val_dataset = MiniImageNet(data, 'val', transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=num_ways*num_shots, shuffle=False)

test_dataset = MiniImageNet(data, 'test', transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=num_ways*num_shots, shuffle=False)


# Define learner and meta-learner models 
#Ddd to gpu if gpu is avaliable

learner = Learner(num_classes).to(device)
meta_learner = MetaLearner(num_classes, inner_learning_rate).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(meta_learner.parameters(), lr=meta_learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    total_accuracy = 0.0
    for batch_index, (support_images, support_labels, query_images, query_labels) in enumerate(train_dataloader):
        # Move data to device
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)

        # Zero gradients
        meta_learner.zero_grad()

        # Compute loss and accuracy on query set
        learner_copy = Learner(num_classes).to(device)
        learner_copy.load_state_dict(learner.state_dict())
        for i in range(num_ways*num_shots):
            x = support_images[i].unsqueeze(0)
            y = support_labels[i].unsqueeze(0)
            output = learner_copy(x)
            loss = criterion(output, y)
            learner_copy.adapt(loss)

        for i in range(num_ways*num_shots, num_ways*(num_shots+1)):
            x = query_images[i].unsqueeze(0)
            y = query_labels[i].unsqueeze(0)
            output = learner_copy(x)
            loss = criterion(output, y)
            total_accuracy += calculate_accuracy(output, y)
            learner_copy.adapt(loss)

        # Compute gradients and update meta-learner
        meta_loss = learner_copy.loss(query_images[num_ways*num_shots:], query_labels[num_ways*num_shots:])
        total_loss += meta_loss.item()
        meta_loss.backward()
        optimizer.step()

    # Print epoch statistics
    average_loss = total_loss / (len(train_dataset) / (num_ways*num_shots*meta_batch_size))
    average_accuracy = total_accuracy / (len(val_dataset) / (num_ways*num_shots))
    print(f'Epoch {epoch+1} | Loss: {average_loss:.4f} | Accuracy:{average_accuracy}')

  