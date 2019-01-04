import argparse
import json
import numpy as np
from PIL import Image
import torch

import imagenetwork as net


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    means = [0.485, 0.456, 0.406]
    std_devs = [0.229, 0.224, 0.225]

    np_image = np.array(image.resize((256, 256)).crop((16, 16, 240, 240)))
    np_image = (np_image / 255 - means) / std_devs
    return torch.from_numpy(np_image.transpose((2,0,1)))


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implement the code to predict the class from an image file
    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    model.eval()
    img = process_image(Image.open(image_path))
    img = img.unsqueeze(0).float().to(device)
    ps = torch.exp(model.forward(img.float()))
    probs, idx = ps.topk(topk)
    idxToClass = {v: k for k, v in model.classToIndex.items()}
    classes = [ idxToClass[v] for v in idx.tolist()[0] ]
    return probs.tolist()[0], classes
    

parser = argparse.ArgumentParser(
    description = 'This program takes an image and an AI model checkpoint to predict the most likely category'
)

parser.add_argument('image',
                    help = 'Image to classify')
parser.add_argument('checkpoint',
                    help = 'Model checkpoint to load and use for classification')
parser.add_argument('--top_k', '-k',
                    help = 'Return the top K most likely classifications',
                    default = 3,
                    type = int)
parser.add_argument('--category_names', '-c',
                    help = 'JSON file to map class values to category names',
                    default = '')
parser.add_argument('--cpu',
                    help = 'Use the CPU even if GPU is available',
                    action = 'store_true',
                    default = False)

args = parser.parse_args()

device = net.defaultDevice(args.cpu)

m = net.loadCheckpoint(args.checkpoint)

if (args.category_names != ''):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

probs, classes = predict(args.image, m, args.top_k)
for i in range(args.top_k):
    if (args.category_names != ''):
        print(f'{cat_to_name[classes[i]]} ({classes[i]}): {probs[i]:.3f}')
    else:
        print(f'{classes[i]}: {probs[i]:.3f}')
