import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
animal_info = {
    "Cat": {
        "Habitat": "Indoors",
        "Diet": "Carnivore",
        "Average Lifespan": "12-15 years",
        "Behavior": "Cuddly and independent",
        "Number of Bones": 230,
        "Average Adult Weight (kg)": 4,
        "Distribution": "Worldwide, often kept as pets"
    },
    "Dog": {
        "Habitat": "Varies (commonly domesticated)",
        "Diet": "Omnivore",
        "Average Lifespan": "10-13 years",
        "Behavior": "Loyal and social",
        "Number of Bones": 319,
        "Average Adult Weight (kg)": 25,
        "Distribution": "Worldwide, often kept as pets"
    },
    "Deer": {
        "Habitat": "Forests, grasslands",
        "Diet": "Herbivore",
        "Average Lifespan": "6-14 years",
        "Behavior": "Herbivore",
        "Number of Bones": 200,
        "Average Adult Weight (kg)": 70,
        "Distribution": "North and South America, Europe, Asia, Africa"
    },
    "Cow": {
        "Habitat": "Farms, grasslands",
        "Diet": "Herbivore",
        "Average Lifespan": "18-22 years",
        "Behavior": "Domesticated",
        "Number of Bones": 206,
        "Average Adult Weight (kg)": 725,
        "Distribution": "Worldwide, domesticated for various purposes"
    },
    "Lion": {
        "Habitat": "Grasslands, savannas",
        "Diet": "Carnivore",
        "Average Lifespan": "8-12 years",
        "Behavior": "Social predators",
        "Number of Bones": 230,
        "Average Adult Weight (kg)": 190,
        "Distribution": "Africa, parts of Asia"
    },
    "Tiger": {
        "Habitat": "Forests, grasslands",
        "Diet": "Carnivore",
        "Average Lifespan": "10-15 years",
        "Behavior": "Solitary hunters",
        "Number of Bones": 230,
        "Average Adult Weight (kg)": 300,
        "Distribution": "Asia"
    },
    "Duck": {
        "Habitat": "Lakes, ponds, rivers",
        "Diet": "Omnivore",
        "Average Lifespan": "2-12 years",
        "Behavior": "Waterfowl",
        "Number of Bones": 100,
        "Average Adult Weight (kg)": 2,
        "Distribution": "Worldwide, near freshwater habitats"
    },
    "Frog": {
        "Habitat": "Wetlands, ponds",
        "Diet": "Insectivore",
        "Average Lifespan": "2-15 years",
        "Behavior": "Amphibian",
        "Number of Bones": 40,
        "Average Adult Weight (kg)": 0.1,
        "Distribution": "Worldwide, often near water sources"
    },
    "Horse": {
        "Habitat": "Grasslands, pastures",
        "Diet": "Herbivore",
        "Average Lifespan": "25-30 years",
        "Behavior": "Domesticated, used for riding and work",
        "Number of Bones": 205,
        "Average Adult Weight (kg)": 500,
        "Distribution": "Worldwide, domesticated for various purposes"
    },
    "Goat": {
        "Habitat": "Mountains, grasslands",
        "Diet": "Herbivore",
        "Average Lifespan": "15-18 years",
        "Behavior": "Domesticated, used for milk and meat",
        "Number of Bones": 220,
        "Average Adult Weight (kg)": 70,
        "Distribution": "Worldwide, domesticated for various purposes"
    }
}

# Test the get_animal_info function with one of the animals



from prettytable import PrettyTable

def get_animal_info(animal_name, animal_info_dict):
    # التحقق مما إذا كان اسم الحيوان موجود في الـ dictionary
    if animal_name in animal_info_dict:
        # استرجاع المعلومات
        info = animal_info_dict[animal_name]
        # إرجاع المعلومات بدلاً من طباعتها
        return info
    else:
        return "Null"


def display_table(animal_info_dict, selected_animal):
    # إعداد العناوين والبيانات للـ PrettyTable
    table = PrettyTable()
    table.field_names = ["Attribute", "Value"]

    # عرض معلومات الحيوان المحدد فقط
    animal_info = get_animal_info(selected_animal, animal_info_dict)
    if animal_info != "Null ":
        for attribute, value in animal_info.items():
            table.add_row([attribute, value])

        # عرض الجدول باستخدام st.write بدلاً من print
        st.write(f"\n information: {selected_animal}")
        st.write(table)
    else:
        st.write(f"\n{selected_animal} none")


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
set_background("BG")

# set title
st.title('Animals classification')

# set header
st.header(' upload your input image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model("keras_model.h5")

# load class names
with open("labels.txt", 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)
    animal_info_output = display_table(animal_info, class_name)



    animal_info = get_animal_info(str(class_name), animal_info)
    st.write("## {}".format(class_name))
    st.write("### Score: {}%".format(int(conf_score * 100) / 100))
    #st.write("### Animal Info: {}".format(animal_info_output))
