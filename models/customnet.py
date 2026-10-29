from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.globAvgPool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 224 x 224

        x = self.conv2(x).relu() # B x 128 x 224 x 224

        x = self.maxpool(x) # B x 128 x 112 x 112

        x = self.conv3(x).relu() # B x 256 x 112 x 112

        x = self.globAvgPool(x) # B x 256 x 1 x 1

        x = self.flatten(x) # B x 256

        x = self.fc1(x) # B x 200

        return x