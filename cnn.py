import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels for RGB images and 6 output channels
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with a 2x2 window
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension, acts as a flattening layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # or (2, 2)
        x = x.view(-1, self.num_flat_features(x)) # 16 * 5 * 5 for flattening
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    net = LeNet()
    print(net)                         # what does the object tell us about itself?

    # Chaining transformations using Compose
    transform = transforms.Compose(
        [transforms.ToTensor(), # "torchvision's ToTenor" -> converts a PIL image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]) # (mean_r, mean_g, mean_b), (std_r, std_g, std_b)
    # mean = torch.mean(torch.stack([sample[0] for sample in ConcatDataset([trainset])]), dim=(0,2,3))
    # std = torch.std(torch.stack([sample[0] for sample in ConcatDataset([trainset])]), dim=(0,2,3))
    # stack all train images together into a tensor of shape
    # (50000, 3, 32, 32)

    # Load the loaded CIFAR-10 dataset from torchvision
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # Organizes the input tensors served by the Dataset into batches with the parameters
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    # The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    # Visualization of the dataset
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


    # Load the test dataset and construct the corresponding DataLoader
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    # Instructed as the last step before training
    criterion = nn.CrossEntropyLoss() # Loss Function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # Gradient Descent
    epochs = 5

    # Training the network
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        # Iterate over the DataLoader for batch processing
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients as an update only happens once per batch
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels) # CrossEntropyLoss internally applies F.log_softmax to the logits and then computes the negative log-likelihood loss against the labels (i.e., the class indices)
            loss.backward() # Gradient Computations
            optimizer.step() # Parameters Update

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # Evaluation
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))