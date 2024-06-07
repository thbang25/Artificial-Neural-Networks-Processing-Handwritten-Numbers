#Thabang Sambo
#10/04/2024
#processing hand written numbers from images and predicting the values written

import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

# Constants
DATA_DIR = "."  # the directory
download_dataset = False  # download the dataset if it does not exist

# load the data from the dataset
train_mnist = datasets.MNIST(DATA_DIR, train=True, download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, download=download_dataset)

# Store as X_train, y_train, X_test, y_test
X_train = train_mnist.data.float() / 255.0 #reduce the values down to a nuber between 0 and 1
y_train = train_mnist.targets
X_test = test_mnist.data.float() / 255.0 #reduce the values down to a nuber between 0 and 1
y_test = test_mnist.targets

# we try to define a maltitude of transformations for the image and resize it 28x28
image_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((28, 28)),transforms.ToTensor(),])

# the ANN class function to try to forwad the neural network
class ArtificialNeuralNetwork(nn.Module):
    def __init__(self):
        super(ArtificialNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10) #we have 10 elements

    def forward(self, x_set):
        x_set = x_set.view(-1, 28 * 28)
        x_set = F.relu(self.fc1(x_set))
        x_set = F.relu(self.fc2(x_set))
        x_set = self.fc3(x_set)
        return F.log_softmax(x_set, dim=1)

# Initialise the model
model = ArtificialNeuralNetwork()

# configure the training parameters
optimizer = optim.Adam(model.parameters(), lr=0.001) #efficient in training neural networks
criterion = nn.NLLLoss() #nll loss criterion ofr classification tasks with softmax

# get the batches of data that provide the training and the testing parameters
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=64, shuffle=False)

#goes over the data analyzes the paremeters and updates them using optimizer   
def ModelTraining(model, optimizer, criterion, train_loader):
    model.train() #train the model
    loss_in_processing = 0.0 #initialize the variable to store the loss
    for data, target in train_loader:
        optimizer.zero_grad()
        ModelProduction = model(data) #the ouput produced
        loss = criterion(ModelProduction, target)
        loss.backward()
        optimizer.step()
        #calculate the training loss
        loss_in_processing += loss.item() * data.size(0)
    #calculate training loss
    loss_in_training = loss_in_processing / len(train_loader.dataset)
    return loss_in_training

# evaluate the peerfomance from the models on that given set of data and perfom calculations
def ModelTesting(model, criterion, test_loader):
    model.eval()
    loss_in_testing = 0.0 #initialize the variable to store the loss
    cal_accuracey = 0 #the accuracy of the predictions
    with torch.no_grad():
        for data, target in test_loader:
            ModelProduction = model(data) #the data that is produced by the model output
            #the amount of loss from the test
            loss_in_testing += criterion(ModelProduction, target).item() * data.size(0)
            classificationMade = ModelProduction.argmax(dim=1, keepdim=True)
            #how many times is the classification valid
            cal_accuracey += classificationMade.eq(target.view_as(classificationMade)).sum().item()
    #calculate the loss in testing
    loss_in_testing /= len(test_loader.dataset)
    #Calculate the the accuracy
    test_accuracy = cal_accuracey / len(test_loader.dataset)
    return loss_in_testing, test_accuracy

# set the number of iterations
set_epoch_amount = 8
for epoch in range(set_epoch_amount):
    loss_in_training = ModelTraining(model, optimizer, criterion, train_loader)
    loss_in_testing, test_accuracy = ModelTesting(model, criterion, test_loader)
    #display results of training and test
    print(f'Epoch {epoch+1}/{set_epoch_amount}, Calculated Training Loss: {loss_in_training:.2f}, Calculated Testing Loss: {loss_in_testing:.2f}, Calculated Testing Accuracy: {test_accuracy:.2f}')
#show done
print("Done!") #epoch count has finished now allow user input

# image classifier function
def classify(model):
    while True:
        #the file path from user
        user_file_path = input("Please enter a filepath:\n")
        #if the user says exit quit
        if user_file_path.lower() == 'exit':
            print("Exiting...")
            break
        #catch block
        try:
        #read the image from the user input path
            image = Image.open(user_file_path)
            image = image_transform(image).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                ModelProduction = model(image)  #the output from the data
                #the classified value
                classification = torch.argmax(ModelProduction, 1).item()
            print("Classifier: ", classification)
            #error exception
        except Exception as e:
            print("Error:", e)

# classify the Numbers in the images
classify(model)
