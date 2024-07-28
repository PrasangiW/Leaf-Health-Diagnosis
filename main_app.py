import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
import time
import plotly.graph_objects as go

# Loading the Model
try:
    model = load_model('plant_disease_model.h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Set page config
st.set_page_config(page_title="Leaf Health Diagnosis", page_icon="🌿", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: green;
    }
    .stApp {
        background-color: #C9C8D8;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black;
    }
    
    .stButtonContainer {
        display: flex;
        justify-content: space-between;
        gap: 10px; /* Adjust the gap between buttons as needed */
    }
    .stButton > button {
        background-color: #012405;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        flex: 1; /* Makes both buttons take equal space */
    }
    .stButton > button:hover {
        background-color: white;
        color:black;
    }
    }
    .stFileUploader > div > button {
        background-color: #2196F3;
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
    }
    .stFileUploader > div > button:hover {
        background-color: #0b7dda;
        color: white;
    }
    .stProgress > div > div > div > div {
        background-color: #0B0851;
    }
    .stMarkdown > div > div {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .css-1d391kg {  /* This is the sidebar class in Streamlit */
        background-color: black !important;
    }
    .css-1d391kg .css-145kmo2, .css-1d391kg .css-145kmo2 * {
        color: white !important;
    }
    
   .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        text-align: center;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Sidebar for input widgets
with st.sidebar:
    
    st.title("Leaf Health Diagnosis")
    st.title("🌿🌿🌿🌿🌿🌿🌿🌿")
    st.markdown("### Upload an image file of the plant leaf to detect the disease")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file...", type=['jpg', 'jpeg', 'png'])
    
    # Buttons for prediction and reset
    submit_button = st.button('Predict Disease')
    reset_button = st.button('Delete')

# Main container
with st.container():
    st.markdown('<h1 style="text-align: center;">🌿🌿🌿 Leaf Health Diagnosis 🌿🌿🌿</h1>', unsafe_allow_html=True)

    if reset_button:
        st.empty()

    if submit_button:
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.type

                # Handle image files
                if file_type in ['image/jpeg', 'image/png']:
                    # Convert the file to an opencv image.
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    
                    # Display the image
                    st.image(opencv_image, channels="BGR", caption='Uploaded Image', use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Resizing the image
                    opencv_image_resized = cv2.resize(opencv_image, (256, 256))
                    
                    # Convert image to 4 Dimension
                    opencv_image_resized = np.expand_dims(opencv_image_resized, axis=0)
                    
                    # Show progress bar and status updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                        status_text.text(f"Processing: {i + 1}%")
                    
                    progress_bar.progress(100)
                    status_text.text("Processing Complete!")
                    
                    # Make Prediction
                    tf.keras.backend.clear_session()
                    try:
                        Y_pred = model.predict(opencv_image_resized)[0]
                        # Display prediction result
                        result = CLASS_NAMES[np.argmax(Y_pred)]
                        st.markdown(f"<h2 style='text-align: center;'>This is a {result.split('-')[0]} leaf with {result.split('-')[1]}</h2>", unsafe_allow_html=True)
                        
                        # Plot prediction probabilities
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=CLASS_NAMES, y=Y_pred, marker=dict(color=['#FF9999', '#FF6666', '#FF3333'])))
                        fig.update_layout(title='Prediction Probabilities',
                                          xaxis_title='Classes',
                                          yaxis_title='Probability',
                                          plot_bgcolor='rgba(0,0,0,0)',
                                          paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in model prediction: {e}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a file before clicking 'Predict Disease'.")
