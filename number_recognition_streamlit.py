import pandas as pd
import numpy as np
import streamlit as st
import torch
import torchvision
import plotly.express as px
from torchvision import transforms
import io
from PIL import Image
from torchvision.io import read_image
from streamlit_drawable_canvas import st_canvas

# ========== Intro Text ==========

st.title('MNIST-Trained Handwritten Digit OCR')
st.write('GitHub: https://github.com/cwfrock')
st.write('Using a convolutional neural network (CNN) trained on the famous MNIST dataset, harness the power of AI to recognize your handwriting!')
st.subheader('Note About Data Privacy')
st.write("""Any upload, drawing, or camera input is hosted by Streamlit\'s third-party cloud providers 
         and is encrypted in transit via 256-bit encryption. No data will be visible or accessible by the owner of this app.
         See https://docs.streamlit.io/streamlit-community-cloud/get-started/trust-and-security for more information.""")
st.subheader("ML Model Background")
st.write("""
        The model used to calculate probabilities and predictions for handwriting input is a simple CNN trained on the standard MNIST dataset. Each
         image in MNIST is 28 x 28 pixels. 
         """)

# ========== Credits ==========

st.subheader("Credits")
st.markdown(" * `streamlit-drawable-canvas`: https://github.com/andfanilo/streamlit-drawable-canvas")
st.markdown(" * **Streamlit**: https://streamlit.io/")
st.markdown(" * *Machine Learning with PyTorch and Scikit-Learn* by Sebastian Raschka, Yuxi (Hayden) Liu, and Vahid Mirjalili: https://sebastianraschka.com/books/#machine-learning-with-pytorch-and-scikit-learn")

# ========== User Input ==========

st.subheader('Handwriting Input')
input_type = st.radio("Choose how you would like to input a sample of your handwritten digit:",
                      ["Upload a .png!", "Draw your digit!"])

# ========== Case: PNG ==========

if input_type == "Upload a .png!":

    st.write('Upload a sample of a handwritten digit (0-9) in .png format.')
    uploaded_file = st.file_uploader(label = 'Upload your file in .png format!', type = 'png')
    if uploaded_file is not None:
        user_img = Image.open(uploaded_file)
        convert_user_input_to_tensor = torchvision.transforms.Compose([
        
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        input_tensor = convert_user_input_to_tensor(user_img)
        st.image(user_img)
    else:
        st.warning("No sample has been found!")
        st.stop()

# ========== Case: Draw ==========

elif input_type == "Draw your digit!":
    st.subheader('Tip!')
    st.write('Change the stroke width and see if this changes the model\'s prediction.')
    stroke_width = st.slider("Stroke width: ", 1, 100, 50)
    canvas_result = st_canvas(background_color = "#ffffff", 
                              stroke_width = stroke_width, 
                              update_streamlit = True, 
                              height = 600, 
                              width = 600, 
                              display_toolbar = True)
    
    form = st.form(key='my_form')
    submitted_input = form.form_submit_button('Submit!')
    if submitted_input:
        new_result_to_tensor = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((28, 28))
        ])
        new_result = new_result_to_tensor(canvas_result.image_data)
        input_tensor_2 = new_result[0, :, :] * 255
        input_tensor_2 = input_tensor_2.unsqueeze(0)
    else:
        st.stop()

else:
    st.warning("Waiting for input.")
    #st.stop()

# ========== Load Torch model and predict ==========

model = torch.load('C://Users//cwfro//OneDrive - cfrock//Machine Learning Resources//streamlit_apps//mnist_number_recognition_app//mnist_model.pt')
    #model.eval()

if input_type == "Upload a .png!":
    pred = model(input_tensor.unsqueeze(0))
elif input_type == "Draw your digit!":
    pred = model(input_tensor_2.unsqueeze(0))
user_number_pred = torch.argmax(pred, dim = 1).detach().numpy()[0]
    #st.write('The model predicts that the number you wrote is a {user_number_pred}!')
    #(input_tensor) * 255
softmax_probas = np.round(np.ravel(torch.nn.functional.softmax(pred, dim = 1).detach().numpy()), 2)
    #softmax_probas
digits = np.arange(0, len(softmax_probas))
    #digits
df = pd.DataFrame({"digits": digits, "probability": softmax_probas})

# ========== The Prediction ==========

st.subheader('Prediction')
st.write('The model predicts that you wrote the number ' + str(user_number_pred) + ' with a probability of ' + str(round(df['probability'].loc[user_number_pred] * 100, 2)) + '%!')

# ========== Plotly Charts ==========

st.subheader('Softmax Probabilities')

fig = px.bar(df, x = "digits", y = "probability", color = "probability", text = "probability", color_continuous_scale = "thermal")
fig.update_traces(textposition = 'inside')
st.plotly_chart(fig)