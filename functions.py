### add imports
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import joblib

def matchedTemplates(orig_img_color, temp_img_left, temp_img_right):


    global finished_left, best_left_match, best_right_match, best_left_match_img, best_right_match_img, curr_max_value, curr_max_value_right, threshold, null_img

    final_cropped_img = temp_img_left
        
    orig_img_gray = cv2.cvtColor(orig_img_color, cv2.COLOR_BGR2GRAY)
    orig_img_rgb = cv2.cvtColor(orig_img_color, cv2.COLOR_BGR2RGB)

    h, w = temp_img_left.shape[::]
        
    sealing_top = h - 25 #- 30
    #sealing_left = 20
    sealing_left = 25
    sealing_bottom = h + 3 #- 2
            
    res = cv2.matchTemplate(orig_img_gray, temp_img_left, cv2.TM_CCOEFF_NORMED)
    
    if st.session_state.finished_left == 0:
        threshold = 0.93 #Pick only values above 0.93. For TM_CCOEFF_NORMED, larger values = good fit.
    
        
    #loc = np.where(res >= threshold)  
    #Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    #print(min_val, max_val, min_loc, max_loc)
    
    if max_val >= threshold:
     #   min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        threshold = max_val * 0.96
        #print(min_val, max_val, min_loc, max_loc, orig_img_file, temp_img_file, threshold)
        #print(min_val, max_val, min_loc, max_loc, threshold, st.session_state.finished_left)
        
        top_left = max_loc  #Change to max_loc for all except for TM_CCOEFF_NORMED        
        cropped_img = orig_img_rgb[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
        cropped_img_sealing = orig_img_rgb[(top_left[1] + sealing_top):(top_left[1] + sealing_bottom), (top_left[0] + sealing_left):(top_left[0] + w)]

       # st.image(cropped_img, caption='Left Part')
       # st.image(cropped_img_sealing, caption='Left Part Sealing')
                
        if st.session_state.finished_left == 0:
            st.session_state.finished_left = 1
            best_right_match = 0
            curr_max_value_right = 0
        
        if max_val > st.session_state.curr_max_value:
            st.session_state.curr_max_value = max_val
            best_left_match_img = orig_img_color            
        
    #else :
        #print("Nothing detected", orig_img_file, temp_img_file, max_val, threshold)
        #print("Nothing detected", max_val, threshold, st.session_state.finished_left)
        
    if st.session_state.finished_left == 1:
        h_right, w_right = temp_img_right.shape[::]
            
        res_right = cv2.matchTemplate(orig_img_gray, temp_img_right, cv2.TM_CCOEFF_NORMED)
        
        threshold_right = 0.90 #Pick only values above 0.90. For TM_CCOEFF_NORMED, larger values = good fit.
    
        #loc_right = np.where(res_right >= threshold_right)  
        #Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.
    
        min_val_right, max_val_right, min_loc_right, max_loc_right = cv2.minMaxLoc(res_right)

        #print(max_val_right, threshold_right, st.session_state.finished_left)
        
        if max_val_right >= threshold_right:
            #st.write(max_val_right, threshold_right, 2)
            #print(min_val_right, max_val_right, min_loc_right, max_loc_right, orig_img_file, temp_img_file_right)
            #print(min_val_right, max_val_right, min_loc_right, max_loc_right, st.session_state.finished_left)
            
            cropped_img = orig_img_rgb[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]
            cropped_img_sealing = orig_img_rgb[(top_left[1] + sealing_top):(top_left[1] + sealing_bottom), (top_left[0] + sealing_left):(top_left[0] + w)]

            #st.image(cropped_img, caption='Right Part', use_column_width=True)
            #st.image(cropped_img_sealing, caption='Right Part Sealing', use_column_width=True)                      
            
            if max_val_right > curr_max_value_right:
                curr_max_value_right = max_val_right
                best_right_match_img = orig_img_color
                best_right_match = 1

        #print(best_right_match, max_val, threshold, 0)        
                  
        if (best_right_match == 1 and max_val < threshold): 
         #   print(best_right_match, max_val, threshold, 1)        

            best_left_match_img = orig_img_color            
            orig_img_first_color = best_left_match_img
            orig_img_first_rgb = cv2.cvtColor(orig_img_first_color, cv2.COLOR_BGR2RGB)
            orig_img_first_gray = cv2.cvtColor(orig_img_first_color, cv2.COLOR_BGR2GRAY)
                        
            res = cv2.matchTemplate(orig_img_first_gray, temp_img_left, cv2.TM_CCOEFF_NORMED)    
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)           
            top_left = max_loc  #Change to max_loc for all except for TM_CCOEF            
                       
            orig_img_second_rgb = cv2.cvtColor(best_right_match_img, cv2.COLOR_BGR2RGB)
            cropped_img_second = orig_img_second_rgb[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w]         
                                   
                                   
            cropped_img_sealing_second = orig_img_second_rgb[(top_left[1] + sealing_top):(top_left[1] + sealing_bottom), (top_left[0] + sealing_left):(top_left[0] + w)]
            #cropped_img_sealing_second_file = f"D:\Volvo\BatteryLid Images Sealing Cycle\BatteryLid000001_11.jpg"
            #cv2.imwrite(cropped_img_sealing_second_file, cropped_img_sealing_second)
            
                        
            st.session_state.finished_left = 0
            best_left_match_img = 0
            curr_max_value = 0
                                 
            final_cropped_img = cropped_img_sealing_second              
            
    return final_cropped_img

def modelPrediction(input_frame):
    img = preprocess_image(input_frame)

    # Load the model from the file
    loaded_model = joblib.load("RF_original_model.joblib")

    expanded = loaded_model.predict(img)
    # expanded_int = int(expanded[0])

    return expanded


# Function to load and preprocess images
def preprocess_image(img, target_size=(150, 150)):
    img = cv2.resize(img, target_size)  # Resize image to a uniform size
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = img.flatten()  # Flatten image into a 1D array
    img = img.reshape(1, -1)
    return img






