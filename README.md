# Image Search Engine AI With Python

**Description:**

This repository implements a user-friendly image search engine using Python and Streamlit. It allows users to upload an image and retrieve similar images from a pre-built database based on visual content.

**Features:**

- Leverages pre-trained ResNet18 model for efficient feature extraction.
- Utilizes cosine similarity for accurate image comparison.
- Streamlit-based interface for easy image upload and result visualization.

**Requirements:**

- Python 3.x
- Streamlit (`pip install streamlit`)
- NumPy (`pip install numpy`)
- Pillow (`pip install Pillow`)
- PyTorch (`pip install torch torchvision`)
- SciPy (`pip install scipy`)

**Setup:**

1. Clone this repository.

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv image-search-engine
   source image-search-engine/bin/activate  # Linux/macOS
   image-search-engine\Scripts\activate.bat  # Windows
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare Your Image Database:

   i. Create a directory named `images` within the project directory.

   ii. Place your `images` inside the images directory.

5. Run the app.py:

   ```bash
    python app.py
   ```

6. Run the application:

   ```bash
    streamlit run front_end.py
   ```

**Usage:**

1. Upload an image using the file uploader.

2. The application will display the uploaded image.

3. The top 5 visually similar images from the database will be retrieved and displayed.
