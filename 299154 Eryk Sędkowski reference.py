''' basic pytorch tools'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import time


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")

default_training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

default_test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

class NeuralNetwork(nn.Module):
    ''' Neural Netowork frame '''
    def __init__(self, num_of_channels=1, num_of_labels=10):
        super().__init__()
        #self.conv_and_pooling = nn.Sequential(
            # nn.Conv2d(in_channels=num_of_channels, out_channels=2, kernel_size=5),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            # nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(5, 5)),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        #)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 200),
            # nn.Linear(784, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, num_of_labels),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, feed):
        ''' using cnn for prediction '''
        #feed = self.conv_and_pooling(feed)
        feed = self.flatten(feed)
        feed = self.linear_relu_stack(feed)
        logits = self.log_softmax(feed)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    ''' traing function '''
    size = len(dataloader.dataset)
    for batch, (image, label) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(image)
        loss = loss_fn(pred, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(image)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    ''' testing loop '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for image, label in dataloader:
            pred = model(image)
            test_loss += loss_fn(pred, label).item()
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct


def training(model, train_dataloader, test_dataloader, learning_rate=0.01):
    ''' whole training function '''
    previous_accuracy = -1
    accuracy = 0
    epoch_num = 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    while epoch_num < 10:
        previous_accuracy = accuracy
        print(f"Epoch {epoch_num+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        accuracy = test_loop(test_dataloader, model, loss_fn)
        # print("accuracy difference:", (accuracy - previous_accuracy))
        epoch_num += 1

    torch.save(model, 'model.pth')
    print("Done!")

def cnn(file='load.pth', training_data=default_training_data,
    test_data=default_test_data, num_of_channels=1, num_of_labels=10):
    ''' creation and training convolutional neural network '''
    model = NeuralNetwork(num_of_channels, num_of_labels)
    train_dataloader = DataLoader(training_data, 64)
    test_dataloader = DataLoader(test_data, 64)
    training(model, train_dataloader, test_dataloader)
    torch.save(model, file)


def main():
    ''' default program behavior '''
    cnn()

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))