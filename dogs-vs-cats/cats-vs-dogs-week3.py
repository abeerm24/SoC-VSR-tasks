#Import all the libraries

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.image import imread

#Load training and testing datasets 
#Note: 1 = dog, 0 = cat

train_dataset = list()
test_dataset = list()

#Create an object to convert images to tensors
convert_tensor = transforms.ToTensor()

#Load the training dataset from the images located on desktop
for i in range(12500):
    catImageName = 'train/train/cat.'+str(i)+'.jpg'
    dogImageName = 'train/train/dog.'+str(i)+'.jpg'
        
    catImage = imread(catImageName)
    dogImage = imread(dogImageName)
    
    #Convert images to tensors
    dogTensor = convert_tensor(dogImage)
    catTensor = convert_tensor(catImage)
    
    #Change image to grayscale and resize
    catTensor = transforms.Grayscale()(transforms.functional.center_crop(catTensor,output_size=[300,300])) 
    dogTensor = transforms.Grayscale()(transforms.functional.center_crop(dogTensor,output_size=[300,300]))
    
    #Create the entries for the training dataset and append them
    dogEntry = (dogTensor,1)
    catEntry = (catTensor,0)
    train_dataset.append(dogEntry)
    train_dataset.append(catEntry)
    
    if (i+1)%500==0: print(i) #Had added this to keep track of progress for loading the training dataset

batch_size = 100    

#Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#Define hyperparameters

input_size = 300*300 #Input dimensions:: 300x300
num_classes = 1 #2 Single binary output: 1=dog,0=cat
hidden_size= 100
alpha = 0.5e-4 #Learning rate
num_epochs = 3
bias = 0.1

class LogisticRegression(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(LogisticRegression, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        output = torch.sigmoid(out)
        return output

#Initialize model
model = LogisticRegression(input_size,hidden_size,num_classes)

#Initialize loss function and optimizer
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)  

#Start the training loop

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        
        #Set all the gradient parameters to 0 at the start of the loop
        optimizer.zero_grad()
        
        images = images.reshape(-1,300*300) #So that now each image becomes a 1D array
        preds = model(images)
        preds = preds.reshape(100) #Reshape the preds tensor
        labels = torch.tensor(labels,dtype = torch.float) #Changed dtype to float
        cost = loss(preds,labels)
        
        #Calculate gradient and update the parameterss
        cost.backward()
        optimizer.step()
        
        if (i+1)%100==0:
            output = 'Epoch: '+str(epoch)+', Iter: '+str(i)+', Loss: '+str(cost)
            print(output)

#Testing the code
def classifier(prob):
    '''
    Function to classify image as cat or dog based on the value predicited by the model
    '''
    if prob<0.5:
        print("Cat")
    else:
        print("Dog")
i=24 #Choose any i from 1 to 12500 and test the model
testImage = imread('test1/'+'test1/'+str(i)+'.jpg')
plt.imshow(testImage)
prob = model(test_dataset[i].reshape(300*300))
print(prob.item)
classifier(prob)

'''
********************NOTE: As there were no labels provided for the testing dataset uploaded on the website,
I haven't calculated accuracy for this model on  the training dataset.************************************************
'''
