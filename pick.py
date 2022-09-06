import pickle
import torch
from torchvision import datasets, transforms

with open("poison_setups/cifar10_transfer_learning.pickle", "rb") as handle:
	setup_dicts = pickle.load(handle)

# Which set up to do in this run?
setup = setup_dicts[0]# this can be changed to choose a different setup

# get set up for this trial
target_class = setup["target class"]
target_img_idx = setup["target index"]
poisoned_label = setup["base class"]
base_indices = setup["base indices"]
num_poisons = len(base_indices)

# load the CIFAR10 datasets
trainset = datasets.CIFAR10(root="./data", train=True, download=True,
transform=transforms.ToTensor())
testset = datasets.CIFAR10(root="./data", train=False, download=True,
   transform=transforms.ToTensor())
# get single target
target_img, target_label = testset[target_img_idx]

# get multiple bases
base_imgs = torch.stack([trainset[i][0] for i in base_indices])
base_labels = torch.LongTensor([trainset[i][1] for i in base_indices])

###
# craft poisons here with the above inputs
###

# save poisons, labels, and target
poison_tuples = []
# poison_tuples should be a list of tuples with two entries each (img, label), example:
# [(poison_0, label_0), (poison_1, label_1), ...]
# where poison_0, poison_1 etc are PIL images (so they can be loaded like the CIFAR10 data in pytorch)
with open("poisons.pickle", "wb") as handle:
	pickle.dump(poison_tuples, handle, protocol=pickle.HIGHEST_PROTOCOL)

# base_indices should be a list of indices witin the CIFAR10 data of the bases, this is used for testing for clean-lable
# i.e. that the poisons are within the l-inf ball of radius 8/255 from their respective bases
with open("base_indices.pickle", "wb") as handle:
	pickle.dump(base_indices, handle, protocol=pickle.HIGHEST_PROTOCOL)

# For triggerless attacks use this
with open("target.pickle", "wb") as handle:
	pickle.dump((transforms.ToPILImage()(target_img), target_label), handle, protocol=pickle.HIGHEST_PROTOCOL)

# For triggered backdoor attacks use this where patch is a 3x5x5 tensor conataing the patch 
# and [startx, starty] is the location of the top left pixel of patch in the pathed target 
# with open("target.pickle", "wb") as handle:
# 	pickle.dump((transforms.ToPILImage()(target_img), target_label, patch, [startx, starty]), handle, 
# 	protocol=pickle.HIGHEST_PROTOCOL)
