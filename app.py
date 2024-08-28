import os
from PIL import Image
import numpy as np
import torch
import torchvision
from torchvision import transforms
from skimage import color

# Directory containing images
images_dir = "./images"

# List of image filenames
images = os.listdir(images_dir)

# Load the ResNet18 model with default pre-trained weights
model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
model.eval()

# Transformations to apply to each image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.avgpool.register_forward_hook(get_activation("avgpool"))

def calculate_histogram(image):
    image = np.array(image)
    image = color.rgb2lab(image)  # Convert to LAB color space for better similarity comparison
    hist = np.histogram(image, bins=50, range=(0, 100))[0]  # Adjust bins and range as needed
    hist = hist / hist.sum()  # Normalize the histogram
    return hist

all_names = []
all_vecs = None
all_histograms = []

with torch.no_grad():
    for i, file in enumerate(images):
        try:
            img_path = os.path.join(images_dir, file)
            img = Image.open(img_path)
            img = img.convert("RGB")  # Ensure image is in RGB format

            # Calculate color histogram
            hist = calculate_histogram(img)
            all_histograms.append(hist)

            img = transform(img)
            out = model(img[None, ...])
            vec = activation["avgpool"].numpy().squeeze()[None, ...]

            if all_vecs is None:
                all_vecs = vec
            else:
                all_vecs = np.vstack([all_vecs, vec])
            all_names.append(file)
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
        
        if i % 100 == 0 and i != 0:
            print(f"{i} images processed")

# Save vectors, histograms, and names to .npy files
np.save("all_vecs.npy", all_vecs)
np.save("all_histograms.npy", np.array(all_histograms))
np.save("all_names.npy", all_names)
