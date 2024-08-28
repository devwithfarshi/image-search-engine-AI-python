import streamlit as st
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
import torch
import torchvision
from torchvision import transforms
from skimage import color

# Function to calculate the color histogram for an uploaded image
def calculate_histogram(image):
    image = np.array(image)
    image = color.rgb2lab(image)  # Convert to LAB color space for better similarity comparison
    hist = np.histogram(image, bins=50, range=(0, 100))[0]  # Adjust bins and range as needed
    hist = hist / hist.sum()  # Normalize the histogram
    return hist

# Function to read precomputed vectors, histograms, and names
@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_histograms = np.load("all_histograms.npy")
    all_names = np.load("all_names.npy")
    return all_vecs, all_histograms, all_names

# Load the vectors, histograms, and names
vecs, histograms, names = read_data()

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

    # Calculate histogram for uploaded image
    uploaded_hist = calculate_histogram(img)

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

        # Calculate distances (feature similarity and color histogram similarity)
        vec_distances = cdist(target_vec, vecs).squeeze()
        hist_distances = cdist(uploaded_hist[None, ...], histograms).squeeze()

        # Combine distances with a weighting factor (e.g., 0.5 for each)
        combined_distances = 0.5 * vec_distances + 0.5 * hist_distances

        # Dropdown for user to select number of top similar images to display
        num_results = st.selectbox("Number of top similar images to display:", range(5, 11))

        # Find the top N similar images based on user selection
        top_n = combined_distances.argsort()[:num_results]

        # Display the top N similar images in a grid with 5 images per row
        cols_per_row = 5
        rows = (num_results + cols_per_row - 1) // cols_per_row  # Calculate the number of rows needed

        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col in range(cols_per_row):
                index = row * cols_per_row + col
                if index < num_results:
                    cols[col].image(Image.open("./images/" + names[top_n[index]]))
else:
    st.write("Please upload an image to start.")
