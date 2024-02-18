
from torchvision import datasets, transforms

from collections import OrderedDict
import warnings

import logging
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import hashlib
import numpy as np

logging.basicConfig(
    filename='benign_client.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
client_logger = logging.getLogger('client_logger') 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)
    return trainloader, testloader

net = SimpleNet()
trainloader, testloader = load_data()


class MaliciousClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, malicious=False):
        self.model = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.malicious = malicious
    def get_parameters(self, config):
        # Maliciously crafted parameters to sabotage the model
       # return [np.random.rand(10, 10) * 1000 for _ in range(5)]  # Exaggerated bad values
        return  [val.cpu().numpy() * 100 for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Skip training and return malicious parameters
        return self.get_parameters(config), 0, {}

    def evaluate(self, parameters, config):
        # Optional: Malicious clients can also lie about their evaluation metrics
        loss, accuracy = 0.0,0
        return loss, len(testloader.dataset), {"accuracy": 0}

fl.client.start_client(
    server_address="127.0.0.1:8089",
    client=MaliciousClient(net, trainloader, testloader).to_client(),
)
    # Adjust the call to start_numpy_client to use keyword arguments
