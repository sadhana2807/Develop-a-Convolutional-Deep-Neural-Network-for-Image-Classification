# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
<img width="712" height="481" alt="image" src="https://github.com/user-attachments/assets/acb7bbb2-0124-4143-8c4e-39417e583c9a" />


## DESIGN STEPS
### STEP 1: 
Load the Fashion-MNIST training and testing datasets and preprocess the images by converting them into tensors and normalizing the pixel values.

### STEP 2: 

Create DataLoaders to divide the datasets into batches for efficient training and testing.

### STEP 3: 
Build a CNN model with three convolutional layers, ReLU activation, max-pooling layers, and fully connected layers for classifying images into 10 categories.


### STEP 4: 
Initialize the Cross-Entropy Loss function and Adam optimizer, then train the CNN for multiple epochs using forward propagation, loss calculation, backpropagation, and weight updates.

### STEP 5: 

Evaluate the trained model using the test dataset and calculate the classification accuracy.

### STEP 6: 

Select a single test image, pass it through the trained CNN, and display the actual class and predicted class.



## PROGRAM

### Name: SADHANA K

### Register Number: 212225240128

```python
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

train_set = torchvision.datasets.FashionMNIST(root='./data', train=True , download=True , transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,transform=transform)

img , lbl = train_set[0]
print(img.shape)
print(len(train_set))
print(len(test_set))

trl = DataLoader(train_set,batch_size=32,shuffle=True)
tstl = DataLoader (test_set,batch_size=32,shuffle=False)

class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x

from torchsummary import summary

model = CNNclassifier1()

criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(),lr=0.001)

if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device = torch.device('cuda')
    model.to(device)

summary(model,input_size=(1,28,28))

epochs = 3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a , b in trl:
        op.zero_grad()
        pred = model(a)
        loss = criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss{i}:", running_loss/len(trl))
        
t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

with torch.no_grad():
    img1,label=test_set[9]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")


```

### OUTPUT

## Training Loss per Epoch

<img width="655" height="80" alt="image" src="https://github.com/user-attachments/assets/8aaf2296-57c7-48f5-b3cc-4d47e17e3cbd" />


## Confusion Matrix

<img width="741" height="631" alt="image" src="https://github.com/user-attachments/assets/ffb49220-8c3f-4896-8ffc-9ddff4e101cd" />


## Classification Report
<img width="788" height="416" alt="image" src="https://github.com/user-attachments/assets/8e98e6a6-c620-4acd-a891-d452a12f2e0b" />


### New Sample Data Prediction
<img width="680" height="631" alt="image" src="https://github.com/user-attachments/assets/6a25a134-09f9-400a-a872-03c6817412e9" />


## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
