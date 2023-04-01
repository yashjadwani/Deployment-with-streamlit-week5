import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

fig = plt.figure(figsize=(3,3))

st.set_page_config(layout="wide")

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('MNIST Predictor')

st.markdown("Upload a grayscale image of a handwritten digit (28x28 pixels)")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    if file_uploaded is not None:    
        image = tf.keras.preprocessing.image.load_img(file_uploaded, target_size=(28, 28), color_mode='grayscale')
    class_btn = st.button("Classify")
        
    predictions = None

    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
            
                predictions = predict(image)
                time.sleep(1)

                st.success(predictions)

    
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<h3 style="text-align:centre;" >Uploaded Image</h3>',unsafe_allow_html=True)
                    st.image(file_uploaded, use_column_width= True)
                with col2:
                    if predictions is not None:
                        st.markdown('<h3 style="text-align:centre;" >HeatMap of the uploaded Image</h3>',unsafe_allow_html=True)
                        plt.imshow(image)
                        plt.axis("off")
                        st.pyplot(fig)
                        

                


def predict(image):
    classifier_model = "model.h5"
    
    model = load_model(classifier_model,compile=False,)

    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    
    predict_prob = model.predict(image)
    prediction=np.argmax(predict_prob,axis=1)
    
    result = f"The Given Image is of number: {prediction[0]}" 
    return result



if __name__ == "__main__":
    main()
