from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import os
import PIL
import csv
import pandas as pd
import streamlit as st
import cv2
from io import StringIO


model=load_model('Bioiatriki_project_binary.h5')
labels = {0: 'Normal', 1: 'Pathological'}

img_size=224

def get_prediction(image_path,my_model,labels):        
        image_loaded = PIL.Image.open(image_path)
        image_loaded = image_loaded.resize((img_size, img_size))
        image_loaded = np.asarray(image_loaded)
      
        if len(image_loaded.shape) < 3:
          image_loaded = np.stack([image_loaded.copy()] * 3, axis=2)
        
        preprocessed_image = preprocess_input(image_loaded)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        
        predictions=my_model.predict(preprocessed_image)
        class_predicted = np.argmax(predictions[0])
        class_predicted_name = labels[class_predicted]                                  
        

        return class_predicted_name
		
st.title("Massive Prediction of Chest X-RAYs")

with st.beta_container():
  bio_image= cv2.imread('ISCA_Logo_small2.png')
  bio_image = cv2.cvtColor(bio_image, cv2.COLOR_BGR2RGB)
  st.image(bio_image)

images_coded= st.file_uploader("Choose files to predict",accept_multiple_files=True)
images=[i.name for i in images_coded]

all_diagnoses=[]
xray_names =[]

if st.button('Press for massive diagnosis'):
	for i,j  in zip(images_coded,images):
	  diagnosis = (get_prediction  (i,model,labels))
	  all_diagnoses.append(diagnosis)
	  xray_names.append(j[:-4])
	  
	table=list(zip(xray_names,all_diagnoses))

	with open('table.csv', 'w', encoding='UTF8', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(['Patient','Diagnosis'])
		writer.writerows(table)
		
	df=pd.read_csv('table.csv')
	st.dataframe(data=df,width = 500,height =500)
	
# 	st.download_button(
# 	     label="Download the diagnosis",
# 	     data=csv,
# 	     file_name='table.csv',
# 	     mime='text/csv',
# 	 )

