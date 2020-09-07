import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 12)

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)



# NN Class
class NN(nn.Module):
    def __init__(self, input, hidden, output):
        super(NN, self).__init__()
        self.lin1 = nn.Linear(input, hidden)
        self.lin2 = nn.Linear(hidden, output)

    def forward(self, X):
        out = torch.sigmoid(self.lin1(X))
        out = torch.sigmoid(self.lin2(out))
        return out

# Hyper-Parameters
lr = 0.01
hidden_size = 4
output = 1
num_epochs = 100

# Model Initialization
model = NN(n_features, hidden_size, output)

# Loss and Optimizer Function
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


# Saving the Checkpoint of the Model
def save_checkpoint(checkpoint, epoch):
    File = 'checkpoint1.pth.tar'
    print(f'=> Saving Checkpoint after Epoch {epoch}')
    torch.save(checkpoint, File)

# Loading the Saved Checkpoint of the Model
def load_checkpoint(checkpoint, model, optimizer):
    print('=> Loading Checkpoint')
    model.load_state_dict(checkpoint['saved_model'])
    optimizer.load_state_dict(checkpoint['saved_optimizer'])


loading_checkpoint = True
if loading_checkpoint:
    load_checkpoint(torch.load('checkpoint1.pth.tar'), model, optimizer)

temp = True

# Training Loop
for epochs in range(num_epochs):
    
    # Forward Prop
    outputs = model.forward(X_train)
    loss = criterion(outputs, y_train)

    if temp:
        print(f'\nDummy Epochs : {epochs}/{num_epochs}, Loss : {loss.item():.3f}')
        temp = False
    
    # Backward Prop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epochs+1) % 10 == 0:
        # Defining the Checkpoint Structure
        checkpoint = {'saved_model' : model.state_dict(), 'saved_optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint,epochs+1)        
        print(f'Epochs : {epochs+1}/{num_epochs}, Loss : {loss.item():.3f}')

print('\nTraining End...')
