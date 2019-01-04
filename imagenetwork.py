import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


# return 'cuda' if GPU is available otherwise, 'cpu'
def defaultDevice(requireCpu):
    return torch.device('cuda' if torch.cuda.is_available() and not requireCpu else 'cpu')

# create a network from a pretrained torchvision model with a custom classifier
def createNetwork(architecture, outputSize, hiddenSize = 256, learningRate = 0.0015, device = ''):
    if (device == ''):
        device = defaultDevice(False)
    network = pretrainModel(architecture)
    if (network == None):
        return None
    if (architecture.startswith('alexnet') or architecture.startswith('vgg')):
        network.classifier[6] = createClassifier(inputSize = network.classifier[6].in_features,
                                                 hiddenSize = hiddenSize, outputSize = outputSize)
        network.to(device)
        network.optimizer = optim.Adam(network.classifier[6].parameters(), lr = learningRate)
    elif (architecture.startswith('densenet')):
        network.classifier = createClassifier(inputSize = network.classifier.in_features,
                                              hiddenSize = hiddenSize, outputSize = outputSize)
        network.to(device)
        network.optimizer = optim.Adam(network.classifier.parameters(), lr = learningRate)
    elif (architecture.startswith('resnet')):
        network.fc = createClassifier(inputSize = network.fc.in_features,
                                      hiddenSize = hiddenSize, outputSize = outputSize)
        network.to(device)
        network.optimizer = optim.Adam(network.fc.parameters(), lr = learningRate)
    network.criterion = nn.NLLLoss()
    network.trainedEpochs = 0
    return network


# create a pretrained torchvision model
def pretrainModel(architecture = 'vgg11'):
    switcher = {
        'alexnet': models.alexnet,
        'densenet121': models.densenet121, 'densenet161': models.densenet161,
        'densenet169': models.densenet169, 'densenet201': models.densenet201,
        'resnet18': models.resnet18, 'resnet34': models.resnet34, 'resnet50': models.resnet50,
        'resnet101': models.resnet101, 'resnet152': models.resnet152,
        'vgg11': models.vgg11_bn, 'vgg13': models.vgg13_bn,
        'vgg16': models.vgg16_bn, 'vgg19': models.vgg19_bn
    }
    # get the function from switcher dictionary
    func = switcher.get(architecture, None)
    if (func == None):
        print(f'Architecture {architecture} is not supported')
        return None
    local_model = func(pretrained = True)
    # freeze parameters so we don't backprop through them
    for param in local_model.parameters():
        param.requires_grad = False
    local_model.architecture = architecture
    return local_model


# create a classifier model that can replace the last step from one of the torchvision models
def createClassifier(inputSize = 1024, hiddenCount = 1, hiddenSize = 128, outputSize = 10, dropout = 0.2):
    modules = []
    if (hiddenCount < 1):
        modules.append(nn.Linear(inputSize, outputSize))
    else:
        modules.append(nn.Linear(inputSize, hiddenSize))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(dropout))
        for i in range(hiddenCount - 1):
            modules.append(nn.Linear(hiddenSize, hiddenSize))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(hiddenSize, outputSize))
    modules.append(nn.LogSoftmax(dim = 1))
    return nn.Sequential(*modules)


# train the model and print loss and accuracy statistics at intervals
def trainModel(model, epochs, printrate, trainloader, validloader, device):
    model.train()
    model.to(device)
    steps = 0
    running_loss = 0
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the device
            inputs, labels = inputs.to(device), labels.to(device)
        
            model.optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = model.criterion(logps, labels)
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()
        
            if steps % printrate == 0:
                test_loss, accuracy = testModel(model, validloader, device)
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/printrate:.3f}.. "
                      f"Test loss: {test_loss:.3f}.. "
                      f"Test accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()
        model.trainedEpochs += 1
        
        
# run a test of the model over the testloader on the device
def testModel(model, testloader, device):
    test_loss = 0
    accuracy = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            test_loss += model.criterion(logps, labels).item()

            # Calculate accuracy
            ps = torch.exp(logps)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()
    return test_loss/len(testloader), accuracy/len(testloader)


# save the network to a checkpoint file
def saveCheckpoint(model, filepath):
    checkpoint = {
        'architecture': model.architecture,
        'hiddenCount': sum([1 for x in model.classifier if type(x) == nn.modules.linear.Linear]) - 1,
        'trainedEpochs': model.trainedEpochs,
        'classToIndex': model.classToIndex,
        'modelState': model.state_dict(),
        'optimizerState': model.optimizer.state_dict()
    }
    if (model.architecture.startswith('alexnet') or model.architecture.startswith('vgg')):
        checkpoint['hiddenSize'] = model.classifier[6][0].out_features
        checkpoint['outputSize'] = model.classifier[6][-2].out_features
        checkpoint['dropout'] = next((x.p for x in model.classifier[6] if type(x) == nn.modules.dropout.Dropout), 0.2)
    elif (model.architecture.startswith('densenet')):
        checkpoint['hiddenSize'] = model.classifier[0].out_features
        checkpoint['outputSize'] = model.classifier[-2].out_features
        checkpoint['dropout'] = next((x.p for x in model.classifier if type(x) == nn.modules.dropout.Dropout), 0.2)
    elif (model.architecture.startswith('resnet')):
        checkpoint['hiddenSize'] = model.fc[0].out_features
        checkpoint['outputSize'] = model.fc[-2].out_features
        checkpoint['dropout'] = next((x.p for x in model.fc if type(x) == nn.modules.dropout.Dropout), 0.2)
    torch.save(checkpoint, filepath)
    
    
# load checkpoint file and rebuild the network model
def loadCheckpoint(checkpointFilePath):
    try:
        checkpoint = torch.load(checkpointFilePath)
        model = createNetwork(architecture = checkpoint['architecture'],
                              hiddenSize = checkpoint['hiddenSize'],
                              outputSize = checkpoint['outputSize'])
        model.load_state_dict(checkpoint['modelState'])
        model.optimizer.load_state_dict(checkpoint['optimizerState'])
        model.trainedEpochs = checkpoint['trainedEpochs']
        model.classToIndex = checkpoint['classToIndex']
        return model
    except FileNotFoundError:
        print('Checkpoint file not found')
    return None


# load data from torchvision ImageFolder directories
def loadData(data_dir, means = [0.485, 0.456, 0.406], std_devs = [0.229, 0.224, 0.225]):
    image_datasets = dict()
    loaders = dict()
    
    # define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = means, std = std_devs)]),
        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = means, std = std_devs)]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = means, std = std_devs)]),
    }
    for k, v in data_transforms.items():
        # load the datasets with ImageFolder
        image_datasets[k] = datasets.ImageFolder(data_dir + '/' + k, transform = data_transforms[k])
        # using the image datasets and the transforms, define the dataloaders
        loaders[k] = torch.utils.data.DataLoader(image_datasets[k], batch_size = 64, shuffle = (k == 'train'))
    return loaders, image_datasets['train'].class_to_idx
