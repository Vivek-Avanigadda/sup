import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
animal_info = {"cat":"lives home","dog":"lives outside"}


def get_animal_info(animal_name, animal_info_dict):
    # التحقق مما إذا كان اسم الحيوان موجود في الـ dictionary
    if animal_name in animal_info_dict:
        # استخراج المعلومات
        info = animal_info_dict[animal_name]
        # طباعة المعلومات
        print(info)

    else:
        print("null")


def set_background(image_file):
  
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    
    # convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # convert image to numpy array
    image_array = np.asarray(image)

    # normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # set model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    #index = 0 if prediction[0][0] > 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score





import streamlit as st
from keras.models import load_model
from PIL import Image
#import numpy as np
#import os
#os.chdir(r"D:\Data YOLOv8")
set_background(r"C:\Users\zbook 17 g3\Downloads\animals\my BG.jpg")

# set title
st.title('Animals classification')

# set header
st.header(' upload your input image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model(r"C:\Users\zbook 17 g3\Downloads\animals\keras_model.h5")

# load class names
with open(r"C:\Users\zbook 17 g3\Downloads\animals\labels.txt", 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)
    get_animal_info(str(class_name), animal_info)


    # write classification
    st.write("## {}".format(class_name,get_animal_info(str(class_name), animal_info)))
    st.write("### score: {}%".format(int(conf_score * 100) / 100))
