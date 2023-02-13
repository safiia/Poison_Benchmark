import warnings
from collections import OrderedDict
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


# #############################################################################
# 
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from learning_module import (
    train,
    test,
    adjust_learning_rate,
    get_model,
    now,
    load_model_from_checkpoint,
    get_transform,
)

def mytrain(net, trainloader, optimizer, criterion, device):

    train_loss = 0
    correct = 0
    total = 0
    for _ in range(1):
        for (images, labels) in tqdm(trainloader):
            optimizer.zero_grad()
            loss=criterion(net(images.to(DEVICE)), labels.to(DEVICE))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = net(images.to(DEVICE)).max(1)
            total += labels.to(DEVICE).size(0)
            correct += predicted.eq(labels.to(DEVICE)).sum().item()
        train_loss = train_loss / len(trainloader)
        acc = 100.0 * correct / total
    print("loss=", train_loss,"accuracy = ", acc)

def train2(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()
def testfl(net, testloader):
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
    print ("loss=",loss / len(testloader.dataset), "accuracy = ", correct / total)


transform_train = get_transform(True, False, dataset="CIFAR100")
transform_test = get_transform(True, False)
trainset = CIFAR100(
    root="./data", train=True, download=True, transform=transform_train
)
testset = CIFAR100(
    root="./data", train=False, download=True, transform=transform_test
)

trainpart = list(range(0, int(0.05*len(trainset)), 1))
testpart = list(range(0, int(0.5*len(testset)), 1))
trainset_1 = torch.utils.data.Subset(trainset, trainpart)
testset_1 = torch.utils.data.Subset(testset, testpart)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


net = load_model_from_checkpoint(model="ResNet18", model_path="check_default/cifar100_100.pth", dataset="CIFAR100")
#pretrained_models/ResNet18_CIFAR100.pth
#check_default/ResNet18_SGD_400.pth
net = net.to(DEVICE)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1,weight_decay=2e-04, momentum=0.9)
criterion = nn.CrossEntropyLoss()

    #        Test Model

training_acc = test(net, trainloader, DEVICE)
natural_acc = test(net, testloader, DEVICE)
print(now(), " Training accuracy: ", training_acc)
print(now(), " Natural accuracy: ", natural_acc)

#mytrain(net, trainloader, optimizer, criterion, DEVICE)
#train(net, trainloader, torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9),torch.nn.CrossEntropyLoss(),DEVICE, train_bn=True)
train2(net, trainloader, epochs=20)
testfl(net, testloader)

# # Define Flower client
# class FlowerClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in net.state_dict().items()]

#     def set_parameters(self, parameters):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         #trainfl(net, trainloader, epochs=10)
#         mytrain(net, trainloader, optimizer, criterion, DEVICE)
#         return self.get_parameters(config={}), len(trainloader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         loss, accuracy = testfl(net, testloader)
#         return loss, len(testloader.dataset), {"accuracy": accuracy}


# # Start Flower client
# fl.client.start_numpy_client(
#     server_address="0.0.0.0:8088",
#     client=FlowerClient(),
# )
