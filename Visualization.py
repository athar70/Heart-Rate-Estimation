import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plotLoss(trainLoss, valLoss):
    '''
    Plots training and validation loss.
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(trainLoss, label='Train Loss')
    plt.plot(valLoss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def plotRmse(mseValues):
    '''
    Plots RMSE for each user based on the provided MSE values.
    '''
    # Compute RMSE as the square root of MSE
    rmseValues = np.sqrt(mseValues)

    # Plot RMSE values
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(rmseValues)), rmseValues)
    plt.xlabel('User')
    plt.ylabel('RMSE')
    plt.title('RMSE for Unseen Participants')
    plt.show()

def plotMse(mseValues):
    '''
    Plots MSE for each user.
    '''
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(mseValues)), mseValues)
    plt.xlabel('User')
    plt.ylabel('MSE')
    plt.title('MSE for Unseen Participants')
    plt.show()
