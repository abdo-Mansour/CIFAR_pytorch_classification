import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from torchsummary import summary
from torchvision import transforms
from torchmetrics import Accuracy, Precision, Recall





################################################################################### Model 1
class FirstModel(nn.Module):
    def __init__(self,num_classes,device,dim = 32):
        super().__init__()
        self.num_of_classes = num_classes
        self.device = device
        self.dim = dim
        # Debugging
        self.DEBUG = False
        # Hyperparameters
        self.num_epochs = 10
        self.learning_rate = 0.001

        # History while Training
        self.model_loss_history = []
        self.model_train_acc_history = []
        self.model_val_acc_history = []
        self.model_val_precision_history = []
        self.model_val_recall_history = []
        self.model_lr_history = []

        # Model Attributes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.accuracy = Accuracy(task= 'multiclass', num_classes=self.num_of_classes, average='macro').to(self.device)
        self.precision = Precision(task= 'multiclass', num_classes=self.num_of_classes, average='macro').to(self.device)
        self.recall = Recall(task= 'multiclass', num_classes=self.num_of_classes, average='macro').to(self.device)
        # Model Architecture
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1000),
            nn.BatchNorm1d(1000),
            
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),

            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),

            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            
            nn.ReLU(),
            nn.Linear(1000, self.num_of_classes),
            nn.BatchNorm1d(self.num_of_classes),
        )
        
    def forward(self, x):
        x = self.feature_extract(x)
        x = self.classifier(x)
        return x
    
    def predict(self, img):
        '''
        returns the predicted classes for the given images
        '''
        self.eval()
        with torch.no_grad():
            img = img.to(self.device)
            output = self(img)
            _, predicted = torch.max(output, 1)
            return predicted
        

    
    def eval_val(self, data_loader):
        '''
        returns accuracy, precision and recall
        '''
        self.eval()
        with torch.no_grad():
            for images, labels in data_loader:
                
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)
                self.accuracy(outputs, labels)
                self.precision(outputs, labels)
                self.recall(outputs, labels)

        return self.accuracy.compute(), self.precision.compute(), self.recall.compute()
    
    def train_model(self, train_loader, val_loader):

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0

            for i, (images, labels) in enumerate(train_loader):

                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i%100 == 0 and self.DEBUG:
                    print(" Step [{}/{}] Loss: {}".format(i, len(train_loader), loss.item()))
                    
            val_acc, val_precision, val_recall = self.eval_val(val_loader)
            train_acc, _, _ = self.eval_val(train_loader)

            self.model_loss_history.append(running_loss/len(train_loader))
            self.model_train_acc_history.append(train_acc.item())
            self.model_val_acc_history.append(val_acc.item())
            self.model_val_precision_history.append(val_precision.item())
            self.model_val_recall_history.append(val_recall.item())
            self.model_lr_history.append(self.optimizer.param_groups[0]['lr'])

            print(f'Epoch: {epoch+1}/{self.num_epochs}, Loss: {loss.item()},Train Acc: {train_acc}, Val Acc: {val_acc}, Val Precision: {val_precision}, Val Recall: {val_recall}')
        
        print('Finished Training')

    def plot_history(self):
        # making two plots one for loss and other for accuracy
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Training History')
        axs[0, 0].plot(self.model_loss_history)
        axs[0, 0].set_title('Model Loss')
        axs[0, 0].set_xlabel('Epochs')
        axs[0, 0].set_ylabel('Loss')

        axs[0, 1].plot(self.model_train_acc_history, label='Train')
        axs[0, 1].plot(self.model_val_acc_history, label='Val')
        axs[0, 1].set_title('Model Accuracy')
        axs[0, 1].set_xlabel('Epochs')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].legend()

        axs[1, 0].plot(self.model_val_precision_history)
        axs[1, 0].set_title('Model Precision')
        axs[1, 0].set_xlabel('Epochs')
        axs[1, 0].set_ylabel('Precision')
        
        axs[1, 1].plot(self.model_val_recall_history)
        axs[1, 1].set_title('Model Recall')
        axs[1, 1].set_xlabel('Epochs')
        axs[1, 1].set_ylabel('Recall')

        axs[0, 2].plot(self.model_lr_history)
        axs[0, 2].set_title('Learning Rate')
        axs[0, 2].set_xlabel('Epochs')
        axs[0, 2].set_ylabel('Learning Rate')
        
        
        # axs[1, 2].axis('off')

        plt.show()
    
    def save_model(self):
        torch.save(self.state_dict(),type(self).__name__+'.pth')

    def print_summary(self):
        summary(self, (3, self.dim, self.dim))

################################################################################### Model 2
class SecondModel(FirstModel):
    def __init__(self,num_classes,device,dim = 32):
        super().__init__(num_classes,device,dim)
        self.dropout_ratio = .2
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten()
            )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_of_classes),            
        )
    
################################################################################### Model 3
class ThirdModel(FirstModel):
    def __init__(self,num_classes,device,dim = 32):
        super().__init__(num_classes,device,dim)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(p=0.2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1000),
            nn.BatchNorm1d(1000),
            
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),

            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),

            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, self.num_of_classes),
            nn.BatchNorm1d(self.num_of_classes),
        )

################################################################################### Model 4
class FourthModel(FirstModel):
    def __init__(self,num_classes,device,dim = 32):
        super().__init__(num_classes,device,dim)
        self.feature_extract = nn.Sequential(
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1000),
            nn.BatchNorm1d(1000),
            
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),

            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),

            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            
            nn.ReLU(),
            nn.Linear(1000, self.num_of_classes),
            nn.BatchNorm1d(self.num_of_classes),
        )
################################################################################### Model 5


def main():
    # Hyperparameters for all models
    batch_size = 32
    num_of_classes = 10
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Dimensions for the input images
    dim = 32

    # Data augmentation and normalization for training
    train_transform = transforms.Compose([ 
        transforms.Resize((dim, dim)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),  # TODO: See what this does
    ])

    # Just normalization for validation
    test_transform = transforms.Compose([
        transforms.Resize((dim, dim)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])


    # Loading the Datasets CIFAR 10
    train_ratio = 0.8

    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=test_transform)
    # Splitting the dataset into training and validation
    train_size = int(train_ratio * len(cifar_dataset))
    val_size = len(cifar_dataset) - train_size
    indices = torch.randperm(len(cifar_dataset))

    train_dataset = torch.utils.data.Subset(train_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[train_size:])


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2 )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    # Loading the test dataset and creating a dataloader
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





    model_1 = FirstModel(num_of_classes,device,dim)
    model_1.to(device)
    model_1.print_summary()

    model_2 = SecondModel(num_of_classes,device,dim)
    model_2.to(device)
    model_2.print_summary()

    model_3 = ThirdModel(num_of_classes,device,dim)
    model_3.to(device)
    model_3.print_summary()

    model_4 = FourthModel(num_of_classes,device,dim)
    model_4.to(device)
    model_4.print_summary()

    models = [model_1, model_2, model_3, model_4]


    results = []
    print("Training the Model 1")
    model_1.train_model(train_loader, val_loader)
    acc, prec, rec = model_1.eval_val(test_loader)
    print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}')
    results.append(([acc, prec, rec], model_1))

    print("Training the Model 2")
    model_2.train_model(train_loader, val_loader)
    acc, prec, rec = model_2.eval_val(test_loader)
    print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}')
    results.append(([acc, prec, rec], model_2))

    print("Training the Model 3")
    model_3.train_model(train_loader, val_loader)
    acc, prec, rec = model_3.eval_val(test_loader)
    print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}')
    results.append(([acc, prec, rec], model_3))

    print("Training the Model 4")
    model_4.train_model(train_loader, val_loader)
    acc, prec, rec = model_4.eval_val(test_loader)
    print(f'Accuracy: {acc}, Precision: {prec}, Recall: {rec}')
    results.append(([acc, prec, rec], model_4))

    # sort the results based on the acc
    results = sorted(results, key=lambda x: x[0][0], reverse=True)
    print("Here is the Ranking of the Models: ")
    for data , model in results:
        print(data[0] , type(model).__name__)
        model.save_model()


if __name__ == '__main__':
    print("Program Started")
    main()