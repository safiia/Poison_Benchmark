import warnings
from collections import OrderedDict
import numpy as np
import flwr as fl
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import os
import argparse
import pickle
import sys
from matplotlib.pyplot import imshow
from PIL import Image
from IPython.display import Image 

import torch.optim as optim
from torchvision import transforms as transforms

from learning_module import (
    now,
    train,
    get_model,
    load_model_from_checkpoint, 
    get_dataset,
    test,
    adjust_learning_rate,
    to_log_file,
    get_transform,
    to_results_table,
    compute_perturbation_norms,
)
# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="bp Poison Attack")
args = parser.parse_args()

ffe =True
poison_indices=None
test_log = "poison_test_log.txt"
out_dict= "in_default"
to_log_file(out_dict,out_dir="output_default", log_name=test_log)

lr = 0.1
poisons_path = "poison_examples/bp_poisons"
args.dataset = "cifar10"
args.train_augment = True
args.normalize = True
args.trainset_size = 2500
args.batch_size=128
net = load_model_from_checkpoint(model="ResNet18", model_path="pretrained_models/ResNet18_CIFAR100.pth", dataset="CIFAR10")
# check_default/ResNet18_cifar10.pth

print(now(), "poison_testing is running.")
# load the poisons and their indices within the training set from pickled files
with open(os.path.join(poisons_path, "poisons.pickle"), "rb") as handle:
    poison_tuples = pickle.load(handle)
    print(len(poison_tuples), " poisons in this trial.")
    poisoned_label = poison_tuples[0][1]
with open(os.path.join(poisons_path, "base_indices.pickle"), "rb") as handle:
    poison_indices = pickle.load(handle)

# get the dataset and the dataloaders

trainloader, testloader, dataset, transform_train, transform_test, num_classes = \
    get_dataset(args, poison_tuples, poison_indices)

# get the target image from pickled file
with open(os.path.join(poisons_path, "target.pickle"), "rb") as handle:
    target_img_tuple = pickle.load(handle)
    target_class = target_img_tuple[1]
    if len(target_img_tuple) == 4:
        patch = target_img_tuple[2] if torch.is_tensor(target_img_tuple[2]) else \
            torch.tensor(target_img_tuple[2])
        if patch.shape[0] != 3 or patch.shape[1] != 5 or \
                patch.shape[2] != 5:
            print(
                f"Expected shape of the patch is [3, {5}, {5}] "
                f"but is {patch.shape}. Exiting from poison_test.py."
            )
            sys.exit()

        startx, starty = target_img_tuple[3]
        target_img_pil = target_img_tuple[0]
        h, w = target_img_pil.size

        if starty + 5 > h or startx + 5 > w:
            print(
                "Invalid startx or starty point for the patch. Exiting from poison_test.py."
            )
            sys.exit()

        target_img_tensor = transforms.ToTensor()(target_img_pil)
        target_img_tensor[:, starty : starty + 5,
                            startx : startx + 5] = patch
        target_img_pil = transforms.ToPILImage()(target_img_tensor)

    else:
        target_img_pil = target_img_tuple[0]

    target_img = transform_test(target_img_pil)

poison_perturbation_norms = compute_perturbation_norms(
    poison_tuples, dataset, poison_indices
)

# the limit is '8/255' but we assert that it is smaller than 9/255 to account for PIL
# truncation.
assert max(poison_perturbation_norms) - 9 / 255 < 1e-5, "Attack not clean label!"

if ffe:
     for param in net.parameters():
         param.requires_grad = False
num_ftrs = net.linear.in_features
net.linear = nn.Linear(num_ftrs, num_classes)


optimizer = torch.optim.SGD(net.parameters(), lr=lr,weight_decay=2e-04, momentum=0.9)
criterion = nn.CrossEntropyLoss()
    
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
    return loss / len(testloader.dataset), correct / total


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


net = net.to(DEVICE)

    #        Test Model

#training_acc = test(net, trainloader, DEVICE)
#natural_acc = test(net, testloader, DEVICE)
#print(now(), " Training accuracy: ", training_acc)
#print(now(), " Natural accuracy: ", natural_acc)


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
        print("==> Training network...")
        epoch = 0
        train_bn=True
        for epoch in range(1):
            adjust_learning_rate(optimizer, epoch, lr_schedule=[40], lr_factor=0.1)
            loss, acc = train(net, trainloader, optimizer, criterion, DEVICE, True)

            if (epoch + 1) % 10 == 0:
                natural_acc = test(net, testloader, DEVICE)
                net.eval()
                p_acc = (
                    net(target_img.unsqueeze(0).to(DEVICE)).max(1)[1].item()
                    == poisoned_label
                )
                print(
                    now(),
                    " Epoch: ", epoch,
                    ", Loss: ", loss,
                    ", Training acc: ", acc,
                    ", Natural accuracy: ", natural_acc,
                    ", poison success: ", p_acc,
                )
        #train(net, trainloader, optimizer=torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9), criterion=torch.nn.CrossEntropyLoss(), device= DEVICE, train_bn=True)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy =testfl(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="0.0.0.0:8088",
    client=FlowerClient(),
)
