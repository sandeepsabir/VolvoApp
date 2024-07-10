import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from functions import matchedTemplates, modelPrediction
import joblib
from datetime import datetime
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Dummy function to simulate model prediction
def predict(frame):
    global finished_left, best_right_match, best_left_match_img, best_right_match_img, curr_max_value, curr_max_value_right, threshold

    img_template_left = cv2.imread('TempLeft.jpg', 0)
    img_template_right = cv2.imread('TempRight.jpg', 0)
    
    #result = modelPrediction(matchedTemplates(frame, img_template_left, img_template_right))

    croppedimg = matchedTemplates(frame, img_template_left, img_template_right)

    if (not np.array_equal(croppedimg, img_template_left)):
        expanded = modelPrediction(croppedimg)
        #print(expanded)
    else:
        expanded = ['9']

    if (expanded[0]=='0'):
        image_placeholder.image(croppedimg, caption=None, use_column_width=True)
        # Display caption with color
        st.markdown(f"<h3 style='color: red; font-size: 18px; font-weight: bold; text-align: center;'>Last Battery Lid: Unexpanded</h3>", unsafe_allow_html=True)
    elif (expanded[0]=='1'):
        image_placeholder.image(croppedimg, caption=None, use_column_width=True)
        st.markdown(f"<h3 style='color: green; font-size: 18px; font-weight: bold; text-align: center;'>Last Battery Lid: Expanded</h3>", unsafe_allow_html=True)

   # if expanded[0] != '9':
    #    print(expanded, datetime.now())

    return expanded

def modelPrediction_file(input_frame):
    img = preprocess_image_file(input_frame)

    # Load the model from the file
    loaded_model = joblib.load("RF_original_model.joblib")
    expanded = loaded_model.predict(img)

    if (expanded[0]=='0'):
        st.caption('Unexpanded')
    elif (expanded[0]=='1'):
        st.caption('Expanded')

    return expanded

def modelPrediction_file_cnn(file_path):  
    loaded_model_cnn = tf.keras.models.load_model("model_original.hdf5")

    img = tf.keras.preprocessing.image.load_img(file_path)
    
    st.image(img)
    
    #, target_size=(150, 150)
       
    # Predict probabilities for test data
    expanded = preprocess_image_cnn(img, target_size=(150, 150))

    # Convert probabilities to class labels (if needed)
    # For binary classification:
    expanded = (expanded > 0.5).astype(int)

    if (expanded==0):
        st.image(img, caption='Unexpanded', use_column_width=False)
    elif (expanded==1):
        st.image(img, caption='Expanded', use_column_width=False)

    print(expanded)

    return expanded    
    

def preprocess_image_file(file_path, target_size=(150, 150)):
    img = cv2.imread(file_path)  # Read image
    img = cv2.resize(img, target_size)  # Resize image to a uniform size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = img.flatten()  # Flatten image into a 1D array
    img = img.reshape(1, -1)
    return img

# Example preprocessing function to load and resize images
def preprocess_image_cnn(img, target_size):    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    global image_placeholder

    # Set the configuration for the Streamlit page
    st.set_page_config(
        page_title="Volvo Battery Lid Defect Detection App",
        page_icon="volvo-icon.ico"  # Provide the path to your icon file
    )

    st.title("Battery Lid Defect Detection")
    
    st.session_state.finished_left = 0
    st.session_state.curr_max_value = 0
    best_right_match = 0
    image_placeholder = st.empty()
       
    # Add a select box to choose between URL and local video file
    #option = st.selectbox("Select Video Source", ("URL", "Local Video File"))
    option = st.selectbox("Select Video Source", ("URL", "Local Video File", "Image"))

    video_source = None
    img_file_path = None

    if option == "URL":
        # Input field for the video stream URL
        video_url = st.text_input("Enter the video stream URL", "http://your-video-stream-url")
        if st.button('Start Video Stream'):
            video_source = video_url

    elif option == "Local Video File":
        # File uploader for local video file
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if video_file is not None:
            video_source = video_file.name
            with open(video_source, 'wb') as f:
                f.write(video_file.read())

    
    elif option == "Image":
        # File uploader for image file
        img_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "gif", "bmp"])
        #img_file_path = "D:\\Volvo\\BatteryLid Images Sealing\\" + img_file.name
        #img_file_path = "D:\\Volvo\\BatteryLid Images Sealing Cycle\\" + img_file.name
        img_file_path = "D:\\Volvo\\" + img_file.name
    
        
    if video_source:
        # Start video capture from the selected source
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            st.error("Unable to access the video stream. Please check the URL or file.")
            return

        # Setting a placeholder for the video frames
        video_placeholder = st.empty()
       
        # Create a placeholder in the Streamlit app
        image_placeholder = st.empty()

        # Run the loop until the user stops the app
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from the video stream. Stopping...")
                break
            
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imwrite(f"D:\Volvo App\Test Frame.jpg", frame)

            # Process the frame using the prediction function
            result = predict(frame)

            #if result == 0:
             #   print(result)
                       
            # Convert the frame to Image format
            img = Image.fromarray(frame)
            
            # Display the frame in the Streamlit app
            video_placeholder.image(img)
            
            # Adding a delay for visibility
            time.sleep(0.05)

        # Release the video capture object
        cap.release()

    elif img_file_path:
        print(type(img_file_path))
        print(img_file_path)
        expanded = modelPrediction_file(img_file_path)  
        #expanded = modelPrediction_file_cnn(img_file_path)  

        print(expanded[0])
        print(type(expanded[0]))


if __name__ == "__main__":
    main()

