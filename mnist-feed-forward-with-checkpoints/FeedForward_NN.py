import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import jovian

# Configuring the GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('GPU')
else:
    device = torch.device('cpu')
    print('CPU')
    

# Defining a Feed Forward NN class
class FeedForward_NN(nn.Module):
    
    def __init__(self, input_size, hidden_layer, output_size):
        super(FeedForward_NN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_layer, output_size)
        
    def forward(self, X):
        out = self.layer1(X)
        out = self.relu(out)
        out = self.layer2(out)
        return out
        
# Defining Hyper-Parameters for the Training Loop
lr = 0.01
input_size = 784 # MNIST input-size = 28*28
hidden_size = 20
output_size = 10 # MNIST has 10 output labels
batch_size = 100
num_epochs = 3

# Initializing Model
model = FeedForward_NN(input_size, hidden_size, output_size).to(device)
        

# Downloading MNIST dataset and dividing the dataset into Batches
train_data = torchvision.datasets.MNIST(root = './data', train = True, download = True, transform = transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root = './data', train = False, download = True, transform = transforms.ToTensor())
train_loader = DataLoader(dataset = train_data, batch_size = batch_size, shuffle = False) 
test_loader = DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False) 

#print('Training Data Shape is : ',train_data)
#print('Training Data Shape is : ',test_data)

# Checking the Images and Labels shape
# iters = iter(train_loader)
# data, labels = iters.next()
# print(data.shape, labels.shape)


# Saving the Checkpoint of the Model
def save_checkpoint(checkpoint, epoch):
    File = 'checkpoint.pth.tar'
    print(f'\n\n=> Saving Checkpoint after Epoch {epoch}')
    torch.save(checkpoint, File)

# Loading the Saved Checkpoint of the Model
def load_checkpoint(checkpoint, model, optimizer):
    print('=> Loading Checkpoint')
    model.load_state_dict(checkpoint['saved_model'])
    optimizer.load_state_dict(checkpoint['saved_optimizer'])


# Defining Loss and Optimizer Function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


# Defining Training Loop
total_step = len(train_loader)
loading_checkpoint = True

if loading_checkpoint:
    load_checkpoint(torch.load('checkpoint.pth.tar'), model, optimizer)

# Experimenting : Printing the model parameters
# model.eval()
# print('\n\nModel Parameters : \n',model.state_dict())
# print('\n\nOptimizer Parameters : \n',optimizer.state_dict())


for epochs in tqdm(range(num_epochs)):
    #print('\nEpoch number is : ',epochs+1)
    
    # Defining the Checkpoint Structure
    checkpoint = {'saved_model' : model.state_dict(), 'saved_optimizer' : optimizer.state_dict()}
    save_checkpoint(checkpoint,epochs+1)

    for i, (imgs, labels) in enumerate(train_loader):
        
        # Reshaping the images from (100,1,28,28)  to  (100,784)
        imgs = imgs.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward Prop
        outputs = model(imgs)
        
        # Calling Loss Function
        loss = criterion(outputs, labels)
        
        # Backward Prop
        loss.backward()
        
        # Updating Paramters
        optimizer.step()
        
        # Zeroing the grad property
        optimizer.zero_grad()
        
        # Printing Model info after every 100 batches processed
        if (i+1) % 100 == 0:
            print(f'\nEpochs : {epochs+1}/{num_epochs}, batch : {i+1}/{total_step}, Loss : {loss.item():.3f}')
            #pass

 # Defining the Checkpoint Structure
checkpoint = {'saved_model' : model.state_dict(), 'saved_optimizer' : optimizer.state_dict()}
save_checkpoint(checkpoint,epochs+1)


# Evaluating the Test Data
with torch.no_grad():
    
    correct = 0
    total_samples = 0
    
    # Evaluation Loop
    for i,(imgs, labels) in enumerate(test_loader):
        
        # Reshaping the images from (100,1,28,28)  to  (100,784)
        imgs = imgs.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Finding the outputs of the Test Data
        outputs = model(imgs)
        
        # Accumulating the Total Samples in the Test Data
        total_samples += labels.shape[0]
        
        # Finding the Predicted Labels
        _, predictions = torch.max(outputs, 1)
        
        # Accumulating the total Correct prediction
        correct += (predictions == labels).sum().item()
        
    acc = ( correct / total_samples ) * 100
    print(f'Accuracy of the Test Data is {acc}')

# Experimenting : Printing the model parameters
# model.eval()
# print('\n\nModel Parameters : \n',model.state_dict())
# print('\n\nOptimizer Parameters : \n',optimizer.state_dict())


# Committing the code into the Jovian.ml cloud
# jovian.commit()