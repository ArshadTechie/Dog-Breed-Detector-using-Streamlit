# Streamlit-App-for-Dog-Breed-Prediction

# Dog Breed Identification using Convolutional Neural Network (CNN)

This project implements a Convolutional Neural Network (CNN) using Keras and TensorFlow to identify the breed of a dog from an input image. This is a supervised machine learning task, specifically a multiclass classification problem.

## Setup

### Create Conda Environment

1. **Clone the Repository**

git clone https://github.com/yourusername/dog-breed-classifier.git   
cd dog-breed-classifier
2. **Create Conda Environment**

  Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt


## Steps Implemented

### Data Acquisition from Kaggle
- The dataset containing dog images and corresponding labels (breeds) was obtained from Kaggle.

### Label Preparation
- A CSV file containing image IDs and their respective breed labels was loaded to associate each image with its breed.

### Data Exploration
- The distribution of dog breeds in the dataset was analyzed to understand the dataset's class balance.

### One-Hot Encoding (OHE)
- The breed labels were one-hot encoded to transform categorical labels into a format suitable for model training.

### Image Loading and Preprocessing
- Dog images were loaded, converted into arrays, and normalized to ensure consistent input to the model.

### Data Validation
- Checks were performed on the dimensions and size of the input data arrays (X) and their corresponding labels (Y).

### Model Architecture Design
- A CNN architecture was designed using Keras to learn features from the input images and predict the breed labels.

### Model Training
- The dataset was split into training and validation sets to train the model. An accuracy plot was generated to visualize model performance during training.

### Model Evaluation
- The trained model was evaluated on the validation set to determine its accuracy in predicting dog breeds.

### Prediction Using the Model
- The trained model was utilized to make predictions on new dog images, enabling identification of the breed from unseen images.

## Integration with Streamlit App

This project also includes a Streamlit web application for interactive dog breed prediction. Users can upload images of dogs, and the app will display the uploaded image along with the predicted breed using the trained CNN model.

### How to Run the Streamlit App
1. **Install Dependencies**
   - Ensure 'requirements.txt' file is installed.

## Usage

1. Place your pre-trained model file `dog_breed_classifier_model.h5` in the repository directory.
2. Update the path to your dataset in the `data_dir` variable inside the `dog_breed_classifier.py` file:
    ```python
    data_dir = r'C:\path\to\your\dataset'
    ```
3. Run the Streamlit app:
    ```bash
    streamlit run dog_breed_classifier.py
    ```

4. Open your web browser and go to `http://localhost:8501` to view the app.

## Folder Structure


