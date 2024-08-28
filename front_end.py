import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import torch
import torchvision
from torchvision import transforms

# Function to read precomputed vectors and names
@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    return all_vecs, all_names

# Load the vectors and names
vecs, names = read_data()

# Initialize session state variables if not already present
if "disp_img" not in st.session_state:
    st.session_state["disp_img"] = ""

# Layout for the Streamlit app
_, fcol2, _ = st.columns(3)

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert the uploaded file to an image
    img = Image.open(uploaded_file)
    img = img.convert("RGB")  # Ensure image is in RGB format
    fcol2.image(img, caption="Uploaded Image")

    # Process the uploaded image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)[None, ...]
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.eval()
    with torch.no_grad():
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        model.avgpool.register_forward_hook(get_activation("avgpool"))
        
        out = model(img_tensor)
        target_vec = activation["avgpool"].numpy().squeeze()[None, ...]

        # Find the top 5 similar images
        top5 = cdist(target_vec, vecs).squeeze().argsort()[1:6]

        # Display the top 5 similar images
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.image(Image.open("./images/" + names[top5[0]]))
        c2.image(Image.open("./images/" + names[top5[1]]))
        c3.image(Image.open("./images/" + names[top5[2]]))
        c4.image(Image.open("./images/" + names[top5[3]]))
        c5.image(Image.open("./images/" + names[top5[4]]))
else:
    st.write("Please upload an image to start.")
