import torch
from torchsummary import summary
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm


#Function for calculating correct predictions
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

#Model training
def train(model, device, train_loader, optimizer, criterion):
  model.train() #Setting model in train mode
  pbar = tqdm(train_loader) #Loading bar

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad() #To prevent gradient accumulation

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  return (100*correct/processed,train_loss/len(train_loader))

#Model testing
def test(model, device, test_loader, criterion):
    model.eval() #Put model in test mode

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return (100. * correct / len(test_loader.dataset),test_loss)

#Create summary of pytorch model
def get_model_summary(model):
  return summary(model, input_size=(1, 28, 28))

#Transformation of data
def get_transforms(train=True):
  if(train):
    transformation = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
  else:
    transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

  return transformation

def get_datasets():
  train_data = datasets.MNIST('../data', train=True, download=True, transform=get_transforms(train=True))
  test_data = datasets.MNIST('../data', train=False, download=True, transform=get_transforms(train=False))

  return (train_data,test_data)

def get_data_loaders(batch_size):
  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
  train_data, test_data = get_datasets()

  test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
  train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

  return (train_loader, test_loader)
