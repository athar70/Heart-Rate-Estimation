from ECG_HR_analysis import processEcgData
from Face_Crop_and_HR_Assignment import assignHRToFrames, writeDataToH5File
from PyTorch_Regression import loadDataset, MyDataset, RegressionModel, trainModel, testModel, initializePretrainedModel
from Visualization import plotLoss, plotRmse, plotMse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

def processAllUsers(startUser=1, endUser=40):
    '''
    Process ECG data, crop faces, and assign HR for users P01 to P40.
    Saves the data to H5 files for each user.
    '''
    for userNum in range(startUser, endUser + 1):
        user = f'P{userNum:02d}'  # Format user number as P01, P02, ..., P40
        print(f"Processing data for {user}...")

        # Step 1: Process ECG data
        userDataPath = f'./Data/Data_Preprocessed_{user}.mat' #ECG data
        minHR, maxHR, error, videoList = processEcgData(userDataPath)

        # Step 2: Crop faces and assign heart rates
        print(f"Cropping faces and assigning heart rates for {user}...")
        hrData = {}  # Load your HR data appropriately for the current user
        videoIDs = videoList  # List of video IDs from the ECG processing step
        listImages, listLabels = assignHRToFrames(user, videoIDs, hrData)

        # Step 3: Save the cropped faces and HR into H5 files for each user
        print(f"Saving data into H5 files for {user}...")
        h5FilePath = f'data/{user}_face_data.h5'
        writeDataToH5File(h5FilePath, listImages, listLabels)

def loadAllUserData(startUser=1, endUser=40):
    '''
    Load data from H5 files for all users (P01 to P40) and return the combined dataset.
    '''
    xData, yData = [], []
    for userNum in range(startUser, endUser + 1):
        user = f'P{userNum:02d}'  # Format user number as P01, P02, ..., P40
        print(f"Loading data for {user}...")
        xUser, yUser = loadDataset(f'data/{user}_face_data', 1, 1)  # Load data for each user
        xData.extend(xUser)
        yData.extend(yUser)
    
    return xData, yData

def main():
    # Process all users from P01 to P40
    print("Processing all users from P01 to P40...")
    processAllUsers(startUser=1, endUser=40)

    # Load data for model training from users P01 to P35
    print("Loading dataset for model training from users P01 to P35...")
    xTrainData, yTrainData = loadAllUserData(startUser=1, endUser=35)

    # Load data for model testing from users P36 to P40
    print("Loading dataset for model testing from users P36 to P40...")
    xTestData, yTestData = loadAllUserData(startUser=36, endUser=40)

    # Prepare data and transformations
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(127),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and split into training and validation sets
    trainDataset = MyDataset(xTrainData, yTrainData, transform=transform)
    testDataset = MyDataset(xTestData, yTestData, transform=transform)

    # Split the training data into training and validation sets
    trainSize = int(0.8 * len(trainDataset))  # 80% for training
    valSize = len(trainDataset) - trainSize   # 20% for validation
    trainSubset, valSubset = random_split(trainDataset, [trainSize, valSize])

    trainLoader = DataLoader(trainSubset, batch_size=32, shuffle=True)
    valLoader = DataLoader(valSubset, batch_size=32, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False)

    # Initialize pretrained DenseNet161 model with custom regression head
    print("Initializing pretrained model...")
    model, criterion, optimizer = initializePretrainedModel(learningRate=0.001)

    # Train regression model with validation
    print("Training the regression model...")
    trainLosses, valLosses = trainModel(trainLoader, valLoader, model, criterion, optimizer)

    # Test the trained model on test users
    print("Testing the model on users P36 to P40...")
    testLoss, mse, r2 = testModel(testLoader, model, criterion)

    # Visualize training and validation losses
    print("Visualizing results...")
    plotLoss(trainLosses, valLosses)  # Plot both train and validation losses

    # Visualize MSE for the test set (replace with real values)
    plotMse([mse])  # Plot MSE
    plotRmse([mse]) # Plot RMSE

    print(f"Test MSE: {mse}, R2: {r2}")
    print(f"Test Loss: {testLoss}")

if __name__ == "__main__":
    main()
