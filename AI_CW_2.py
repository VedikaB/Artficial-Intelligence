#!/usr/bin/env python
# coding: utf-8

# # Neural networks for Classification - FashionMNIST

# In[2]:


import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random

train_set = torchvision.datasets.FashionMNIST(root = ".", train=True,
download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train=False,
download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# fix the seed to be able to get the same randomness across runs and hence reproducible outcomes
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# if you are using CuDNN , otherwise you can just ignore
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


# In[3]:


input_data, label = next(iter(train_loader))
plt.imshow(input_data[0,:,:,:].numpy().reshape(28,28), cmap="gray_r");
print("Label is: {}".format(label[0]))
print("Dimension of input data: {}".format(input_data.size()))
print("Dimension of labels: {}".format(label.size()))


# In[4]:


# From Lab 07_2
# CNN implementation

class MyCNN(nn.Module):
  def __init__(self):
    super(MyCNN, self).__init__()
    
    # Parameters to input to nn.Conv2d as specified in the task
    # 1: number of input channels (1 for the images of the FashionMNIST dataset)
    # 2: number of output channels
    # 3: kernel dimensionality (1 dimension if both dimensions are the same)
    # 4: stride dimensionality (1 dimension if both dimensions are the same)
    
    # self.conv = nn.Conv2d(1, 12, kernel_size=3, stride=1)
    
    # activation function as specified in the task
    # self.act_conv = nn.ELU()
    
    # Parameters to input to nn.MaxPool2d as specified in the task
    # 1: kernel dimensionality (1 dimension if both dimensions are the same)
    # 2: stride dimensionality (1 dimension if both dimensions are the same)
    
    # self.max_pool = nn.MaxPool2d(2, stride=2)
    
    # Parameters to input to nn.Conv2d
    # 1: the first input parameter specifies the number of output channels from the previous layer (i.e. 12)
    
    # self.conv1 = nn.Conv2d(12, 26, kernel_size=3, stride=1)
    # self.act_conv1 = nn.ELU()
    # self.max_pool1 = nn.MaxPool2d(2, stride=2)
    
    
    # using Sequential container to run layers sequentially
    self.cnn_model = nn.Sequential(nn.Conv2d(1, 12, kernel_size = 3, stride=1), nn.ReLU(), nn.MaxPool2d(2, stride=2), nn.Conv2d(12, 26, kernel_size = 3, stride = 1), nn.ReLU(), nn.MaxPool2d(2, stride = 2))
    
    
    # Parameters to input to nn.Linear
    # 1: last output dimension of the previous layer
    # Note: if previous layer is a CNN or a MaxPool layer the dimension is the one of the flattened output
    # Note: we keep the batch_size dimension constant in the network
    # for example, 32 x 5 x 5 x 26 (batch_size x (5 x 5 x 26) feature matrix) -> 32 x 650 (5*5*26)
    # 2: output dimension
    
    # self.fc = nn.Linear(650, 650)
    # self.act = nn.ELU()
    
    # dropout is applied after the activation
    # self.drop = nn.Dropout(0.5)
    
    # self.fc1 = nn.Linear(650, 256)
    # self.act1 = nn.ELU()
    # self.drop1 = nn.Dropout(0.5)
    
    # self.fc2 = nn.Linear(256, 10)
    
    # using Sequential container to run layers sequentially
    self.fc_model = nn.Sequential(nn.Linear(650, 650), nn.ReLU(),nn.Dropout(0.3), nn.Linear(650,256), nn.ReLU(),nn.Dropout(0.3), nn.Linear(256, 10))
 
   
  def forward(self, x):
    
    # pass input via the CNN layers
    x = self.cnn_model(x)
    # we reshape the tensor
    # we keep the first dimension (batch_size)
    # we let Pytorch compute the second dimension 
    # (-1 means compute this dimension given the others)
    x =x.view(x.size(0), -1)
    # pass input via the fully-connected layers
    x = self.fc_model(x)
    
    return x


# In[5]:


def evaluation(dataloader):
  total, correct = 0,0
  # turn on evaluate mode, this de-activates dropout
  # (good practice to include in your projects even if it is not used)
  net.eval()
  for data in dataloader:
    
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = net(inputs)
    # we take the index of the class that received the highest value  
    # we take outputs.data so that no backpropagation is performed for these outputs
    _, pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    # .item() takes Python float values from the tensor
    correct += (pred == labels).sum().item()
  return 100 * correct / total


# In[6]:


def weights_init(m):
    # initialise both linear and convolutional layers
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


# In[ ]:


device = torch.device("cuda:0")

alpha = 0.1

net = MyCNN().to(device)
# initialise weights
net.apply(weights_init)

# Note: CrossEntropy loss is usually used for classification tasks
# check slide 31 of Lecture 7
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# SGD optimiser, learning rate is specified by alpha
optimiser = torch.optim.SGD(list(net.parameters()), lr=alpha)
    
loss_epoch_array = []
max_epochs = 30
loss_epoch = 0
train_accuracy = []
test_accuracy = []
# loop over epochs
for epoch in range(max_epochs):
  # we will compute sum of batch losses per epoch
  loss_epoch = 0
  # loop over batches
  for i, data in enumerate(train_loader, 0):
    # to ensure the dropout is "turned on" while training
    # (good practice to include in your projects even if it is not used)
    net.train()
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    # zero the gradients
    optimiser.zero_grad()
    outputs = net(inputs)
    # compute the loss
    loss = loss_fn(outputs, labels)
    # calculate the gradients
    loss.backward()
    # update the parameters using the gradients and optimizer algorithm 
    optimiser.step()
    # we sum the loss over batches
    loss_epoch += loss.item()
  
  loss_epoch_array.append(loss_epoch)
  train_accuracy.append(evaluation(train_loader))
  test_accuracy.append(evaluation(test_loader))
  print("Epoch {}: loss: {}, train accuracy: {}, valid accuracy:{}".format(epoch + 1, loss_epoch_array[-1], train_accuracy[-1], test_accuracy[-1]))


# In[ ]:


plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(train_accuracy, "r")
plt.plot(test_accuracy, "b")
plt.gca().legend(('train','test'))


# In[ ]:


plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(loss_epoch_array)


# In[ ]:





# In[ ]:




