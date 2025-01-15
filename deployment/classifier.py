#import libraries
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow_hub.keras_layer import KerasLayer
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

#import pickle
import pickle

#load model
def run():
    # set title
    st.title('AI Generated vs Real Picture')
    st.write('---')
    
    # set banner image
    st.image('https://www.boldbusiness.com/wp-content/uploads/2024/04/featured-Fake-News.jpeg', caption='boldbusiness.com')

    # description
    st.write('AI and Real. Can you tell the difference?')
    
    file = st.file_uploader("Upload an image", type=["jpg", "png"])

    # Load model once outside of the file upload handler
    model = load_model('model_best.keras', custom_objects={'KerasLayer': KerasLayer})
    target_size = (128, 128)

    def import_and_predict(image_data, model):
        # Convert the uploaded image to PIL image
        image = Image.open(image_data)
        image = image.resize(target_size)  # Resize image to match the model's expected input size
        
        # Convert image to array
        img_array = img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        
        # Normalize the image
        img_array = img_array / 255.0
        
        # Make prediction
        predictions = model.predict(img_array)

        # For binary classification, the output will have 1 neuron (sigmoid output)
        if predictions.shape[-1] == 1:  # Sigmoid output for binary classification
            idx = 1 if predictions[0][0] >= 0.5 else 0
        else:  # Softmax output (multi-class classification)
            idx = np.argmax(predictions)

        labels = ['AI Generated Image', 'Real Image']
        result = f"Prediction: {labels[idx]}"

        return result

    # File handling and predictions
    if file is None:
        st.text("Please upload an image file")
    else:
        try:
            # Run prediction
            result = import_and_predict(file, model)
            
            # Display the image and result
            st.image(file)
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")
        
if __name__ == "__main__":
    run()
