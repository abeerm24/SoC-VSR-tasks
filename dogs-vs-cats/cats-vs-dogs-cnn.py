#Import all the libraries

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.image import imread

#Load training and testing datasets 
#Note: 1 = dog, 0 = cat

train_dataset = list()
test_dataset = list()

#Create an object to convert images to tensors
convert_tensor = transforms.ToTensor()

for i in range(8000):
    catImageName = 'train/train/cat.'+str(i)+'.jpg'
    dogImageName = 'train/train/dog.'+str(i)+'.jpg'
        
    catImage = imread(catImageName)
    dogImage = imread(dogImageName)
    
    #Convert images to tensors
    dogTensor = convert_tensor(dogImage)
    catTensor = convert_tensor(catImage)
    
    #Change image to grayscale and resize
    catTensor = transforms.functional.center_crop(catTensor,output_size=[300,300]) 
    dogTensor = transforms.functional.center_crop(dogTensor,output_size=[300,300])
    
    #Create the entries for the training dataset and append them
    dogEntry = (dogTensor,1)
    catEntry = (catTensor,0)
    train_dataset.append(dogEntry)
    train_dataset.append(catEntry)
    
    if (i+1)%500==0: print(i+1) #Had added this to keep track of progress for loading the training dataset

batch_size = 100    

#Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#Define hyperparameters
alpha = 1e-4 #Learning rate
num_epochs = 3

class ConvNeuralNet(nn.Module):
    def __init__(self):
        super(ConvNeuralNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16,16,5)
        self.fc1 = nn.Linear(16 * 34 * 34, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10,1)
    
    def forward(self,x):
        #Note: x was a 3x300x300 array
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 296/2=148, 296/2=148
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 72, 72
        x = self.pool(F.relu(self.conv3(x)))  # -> n, 16, 34, 34
        x = x.view(-1, 16 * 34 *34)            # -> n, 34*34
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        output  = torch.sigmoid(self.fc4(x))
        return output

#Initialize model
model = ConvNeuralNet()

#Initialize loss function and optimizer
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)  

#Train the model

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        
        #Set gradient paramters to zero
        optimizer.zero_grad()
        
        #Forward pass
        preds = model(images)
        preds = preds.view(100)
        labels1 = torch.tensor(labels,dtype=torch.float) #Changed dtype of labels to float
        cost= loss(preds,labels1)
        
        #Perform backward pass and find gradients
        cost.backward()
        optimizer.step()
        
        if (i+1)%1000==0:
            output = 'Epoch: '+str(epoch)+', Iter: '+str(i+1)+', Loss: '+str(cost)
            print(output)

#Testing the model
def classifier(prob):
    '''
    Function to classify image as cat or dog based on the value predicited by the model
    '''
    if prob<0.5:
        print("Cat")
    else:
        print("Dog")
i=800
testImage = imread('test1/'+'test1/'+str(i)+'.jpg')
plt.imshow(testImage)
testTensor = convert_tensor(testImage)
testTensor = transforms.functional.center_crop(testTensor,output_size=[300,300])
prob = model(testTensor)
print(prob.item())
classifier(prob)

'''
********************NOTE: As there were no labels provided for the testing dataset uploaded on the website,
I haven't calculated accuracy for this model on  the training dataset.************************************************
'''

#Saving the model parameters (state dict)
FILE = "cnn-model.pth"
torch.save(model.state_dict(),FILE)


