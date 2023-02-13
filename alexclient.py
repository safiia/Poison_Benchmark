import warnings
from collections import OrderedDict
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):

    def __init__(self, num_classes=10, feature_size=64):
        super(Net, self).__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.lrn1 = nn.LocalResponseNorm(9, k=1.0, alpha=0.001, beta=0.75)
        self.conv2 = nn.Conv2d(64, feature_size, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.lrn2 = nn.LocalResponseNorm(9, k=1.0, alpha=0.001, beta=0.75)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.linear1 = nn.Linear(feature_size * 8 * 8, 384)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(384, 192)
        self.relu4 = nn.ReLU()
        self.linear = nn.Linear(192, num_classes)

    def forward(self, x):
        feats = self.maxpool2(
            self.lrn2(
                self.relu2(
                    self.conv2(self.lrn1(self.maxpool1(self.relu1(self.conv1(x)))))
                )
            )
        )
        feats_vec = feats.view(feats.size(0), self.feature_size * 8 * 8)
        out = self.linear(self.relu4(self.linear2(self.relu3(self.linear1(feats_vec)))))
        return out

    def penultimate(self, x):
        feats = self.maxpool2(
            self.lrn2(
                self.relu2(
                    self.conv2(self.lrn1(self.maxpool1(self.relu1(self.conv1(x)))))
                )
            )
        )
        feats_vec = feats.view(feats.size(0), self.feature_size * 8 * 8)
        out = self.relu4(self.linear2(self.relu3(self.linear1(feats_vec))))
        return out

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(20):
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
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    trainpart = list(range(0, int(0.06*len(trainset)), 1))
    testpart = list(range(0, int(0.06*len(testset)), 1))  
    trainset_1 = torch.utils.data.Subset(trainset, trainpart)
    testset_1 = torch.utils.data.Subset(testset, testpart)
    return DataLoader(trainset_1, batch_size=32, shuffle=True), DataLoader(testset_1)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=3)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient(),
)
