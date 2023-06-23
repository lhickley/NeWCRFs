import torch
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import transforms
from networks.NewCRFDepth import NewCRFDepth
from dataloaders.dataloader import NewDataLoader

def load_image(image_path):
    # Load image, then convert it to RGB and normalize it to [0, 1]
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

transform = transforms.Compose([
    transforms.ToTensor(),
    ])

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='newcrfs')
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--image_path',                type=str,   help='path to the image', default='/Users/lewishickley/Downloads/InitialTestImage.jpg') #TODO change this default and the handling around it.
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='/Users/lewishickley/BBK/MscThesis/models/initialSet/NeWCRFs/models/model_nyu.ckpt')

args = parser.parse_args()

args.mode = 'test'
#dataloader = NewDataLoader(args, 'test')

model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
model = torch.nn.DataParallel(model)

checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint['model'])
model.eval()
#model.cuda()

num_params = sum([np.prod(p.size()) for p in model.parameters()])

img = load_image(args.image_path)
img = transform(img)
img = img.unsqueeze(0).float()

with torch.no_grad():
    #image = Variable(sample['image'].cuda())
    # Predict
    output = model(img)

# The output is a depth map, it's up to us how to process it
# For simplicity, let's convert it to numpy and squeeze unnecessary dimensions
output = output.cpu().numpy().squeeze()

#Return a plot of the data so we can visualise how it is doing.
#TODO Write a function to save this image.
plt.imshow(output, cmap='inferno')
plt.colorbar()
plt.show()