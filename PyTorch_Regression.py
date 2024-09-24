import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index].reshape(128, 128, 3)
        x = np.uint8(x * 255)  # Convert normalized values to original range
        x = transforms.ToPILImage()(x)
        if self.transform:
            x = self.transform(x)
        y = self.labels[index].astype(np.float32)
        return x, y

    def __len__(self):
        return len(self.data)

class MultipleRegression(nn.Module):
    def __init__(self, inputFeatures):
        super(MultipleRegression, self).__init__()
        self.layer1 = nn.Linear(inputFeatures, 16)
        self.layer2 = nn.Linear(16, 32)
        self.layer3 = nn.Linear(32, 16)
        self.outputLayer = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.outputLayer(x)

def trainModel(trainLoader, valLoader, model, criterion, optimizer, numEpochs=15):
    '''
    Trains the regression model with validation. Returns both training and validation losses.
    '''
    model.train()
    trainLosses = []
    valLosses = []

    for epoch in range(numEpochs):
        # Training phase
        model.train()
        epochTrainLoss = 0
        for inputs, targets in trainLoader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            epochTrainLoss += loss.item()

        avgTrainLoss = epochTrainLoss / len(trainLoader)
        trainLosses.append(avgTrainLoss)

        # Validation phase
        model.eval()
        epochValLoss = 0
        with torch.no_grad():
            for inputs, targets in valLoader:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                epochValLoss += loss.item()

        avgValLoss = epochValLoss / len(valLoader)
        valLosses.append(avgValLoss)

        print(f'Epoch {epoch + 1}, Train Loss: {avgTrainLoss}, Val Loss: {avgValLoss}')
    
    return trainLosses, valLosses


def testModel(testLoader, model, criterion):
    '''
    Test the trained model on the test dataset and calculate evaluation metrics.
    Returns test loss, MSE, and R² score.
    '''
    model.eval()
    testLoss = 0.0
    yTrue, yPred = [], []

    with torch.no_grad():
        for inputs, targets in testLoader:
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            testLoss += loss.item()

            # Collect predictions and true values for evaluation
            yTrue.extend(targets.cpu().numpy())
            yPred.extend(outputs.cpu().numpy())

    mse = mean_squared_error(yTrue, yPred)
    r2 = r2_score(yTrue, yPred)

    print(f"Test MSE: {mse}, R2: {r2}")
    return testLoss / len(testLoader), mse, r2


def initializePretrainedModel(learningRate=0.001):
    '''
    Initialize a pretrained DenseNet161 model with a custom regression head.
    '''
    # Get pretrained DenseNet161 model
    model = models.densenet161(pretrained=True)
    
    # Turn off training for all parameters in the base model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier with a custom regression head
    classifierInput = model.classifier.in_features
    classifier = MultipleRegression(classifierInput)

    model.classifier = classifier
    model = model.to('cuda')  # Move model to GPU

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    return model, criterion, optimizer
