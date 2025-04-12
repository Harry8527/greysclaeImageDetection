# Import Dependencies

import torch
from torch import nn, save, load
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Get data
training_data = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(training_data, 32)
# MNIST images are in the shape of 1*28*28
# It has total of 10 class as 0-9


# Image Classifier Neural network
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            # Since MNIST output images shape is 28*28, that's why 28 here. 
            # Each of the Conv2D neural network shave off images by 2 pixels each time, and a total of 6 . Hence -6 we will do.
            # 10 because there are total 10 image class in MNIST dataset. (0-9)
            nn.Linear(64*(28-6)*(28-6), 10) 
        )
    def forward(self, x):
        return self.model(x)
    
# Create an instance of the ImageClassifier class.
clf = ImageClassifier().to('cpu')
# Instantiate our optimizer, i.e. Adam, which we have imported above
opt = Adam(clf.parameters(), lr=1e-3)
# Initialize our loss function
loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    """First we are opening the trained model in read mode, and loading it into our classifier."""
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))
    
    """Storing the image for detection into image variable."""
    image = Image.open('number_6.png')  # Loading the image.
    image = image.convert('L')  # Convert the image to greyscale
    image = image.resize((28,28), Image.Resampling.BILINEAR) # Resize the image to 28 X 28 pixels
    """Converting the image to Tensor object"""
    image_tensor = ToTensor()(image).unsqueeze(0).to('cpu') # converting the image to tensor object
    print(torch.argmax(clf(image_tensor))) # Passing the image forward to the image classifier for prediction.


    """Commenting this training section of the code, since the model is trained already."""
    # train for 10 epochs
    # for epoch in range(10):
    #     for batch in dataset:
    #         x,y = batch
    #         x,y = x.to('cpu'), y.to('cpu')
    #         yhat = clf(x)
    #         loss = loss_fn(yhat, y)

    #         #Apply backprop
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #     print(f"Epoch: {epoch} loss is {loss.item()}")

with open('model_state.pt', 'wb') as f:
    save(clf.state_dict(), f)
