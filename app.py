import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
#import cv2

fig = plt.figure()

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('MNIST Predictor')

st.markdown("Welcome to web application that classifies hand Writted Digits")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.pyplot(fig)
                st.write(predictions)


def predict(image):
    classifier_model = "model.hs"
    IMAGE_SHAPE = (28, 28,1)
    model = load_model(classifier_model,compile=False,)
    
    data = np.ndarray(shape=(1,28,28,1))
    test_image = image
    size=(28,28)
    test_image = ImageOps.fit(test_image,size,Image.ANTIALIAS)
    
    image_array = np.asarray(test_image)
    
    normalized_image_array = image_array /255.0
    
    data[0]= normalized_image_array
    
    #test_image = image.resize((28,28),Image.ANTIALIAS)
    #test_image = preprocessing.image.img_to_array(test_image)
    #test_image = test_image / 255.0
    #test_image = np.expand_dims(test_image, axis=0)
    
    
    #test_image = np.repeat(test_image,-1, axis = 0)
    class_names = ['No retinopathy: class 0',
          'Mild retinopathy: class 1',
          'Moderate retinopathy: class 2',
          'Severe retinopathy: class 3',
          'Proliferative: class 4 '
          ]
    predict_prob = model.predict(data)
    prediction=np.argmax(predict_prob,axis=1)
    
    result = f"The Given Image is of number: {prediction[0]}" 
    return result



if __name__ == "__main__":
    main()
