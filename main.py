import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision.models.detection import transform
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.nn.functional import max_pool2d
import torch.nn.functional as F
import torch.optim as optim

data_dir = r'C:\Users\TobiaszRolla(272522)\Desktop\SieciNeuronowe2024\data\test'
class CardsDataSet(Dataset):
    #ładuje dane przez ImageFolder z odpowiednią transformacją
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    #zraca rozmiar zbioru
    def __len__(self):
        return len(self.data)
    #zwraca data po podaniu indexu
    def __getitem__(self, ind):
        return self.data[ind]

    #właściwośćzwracająca wszystkie klasy
    @property
    def classes(self):
        return self.data.classes

#dlatego że torch nie ma MinPoolingu
class MinPooling2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MinPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    #trzeba forward zdefiniować aby wykonywała maxa na odwróconych wartościach a potem odwracała jeszcze raz
    def forward(self, x):
        x_neg = -x
        pooled = max_pool2d(x_neg, self.kernel_size, self.stride, self.padding)
        return -pooled
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__() #dziedziczenie konstruktora

        #warstwy konwulacyjne
        self.pool = MinPooling2d(2, 2, )
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 32 * 32, 1024)  # Zmiana w wymiarach po spłaszczeniu
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 53)
    def forward(self,x):
        x = self.pool(self.conv1(x))
        print( x.shape)
        x = self.pool(self.conv2(x))
        print(x.shape)
        x = x.view(x.size(0), -1) #spłaszczenie obrazu dla warstwy liniowej
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x

#Miejsce na nasze transformacje
my_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

#ładujemy dataset
dataset = CardsDataSet(data_dir, transform=my_transform)

#data loder ładuje 20 zdjęć na iteracje shuffle-mieszanie co epoka
dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

model = NeuralNetwork()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#pętla trenowania
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print(dataset.__len__())
print(len(dataset.classes));




