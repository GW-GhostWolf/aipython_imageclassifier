import argparse
import imagenetwork as net

printFreqency = 8

parser = argparse.ArgumentParser(
    description = 'This program will train an AI model to catagorize images'
)

parser.add_argument('data_dir',
                    help = 'Image folder. Should have 3 subfolders (train, test, valid) that contain torchvision ImageFolder compatible folder structures')
parser.add_argument('--save_dir', '-sd',
                    help = 'Directory to save checkpoint file(s)',
                    default = './')
parser.add_argument('--arch', '-a',
                    help = 'Torchvision pretrained architecture to use',
                    default = 'densenet161')
parser.add_argument('--learning_rate', '-lr',
                    help = 'Learning rate for the model to use during training',
                    default = 0.0015,
                    type = float)
parser.add_argument('--hidden_units', '-hu',
                    help = 'Number of nodes to use in the hidden layer of the model',
                    default = 512,
                    type = int)
parser.add_argument('--epochs', '-e',
                    help = 'Number of epochs to train the model',
                    default = 1,
                    type = int)
parser.add_argument('--cpu',
                    help = 'Train the model on the CPU even if GPU is available',
                    action = 'store_true',
                    default = False)

args = parser.parse_args()

if (not args.save_dir.endswith('/')):
    args.save_dir += '/'

device = net.defaultDevice(args.cpu)

loaders, classIndexMap = net.loadData(args.data_dir)

model = net.createNetwork(architecture = args.arch, hiddenSize = args.hidden_units,
                          outputSize = len(classIndexMap), learningRate = args.learning_rate,
                          device = device)
model.classToIndex = classIndexMap

net.trainModel(model, args.epochs, printFreqency, loaders['train'], loaders['valid'], device)

net.saveCheckpoint(model, f'{args.save_dir}{args.arch}_{args.hidden_units}.pth')