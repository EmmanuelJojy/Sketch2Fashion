import streamlit as st
from PIL import Image
import os
from sidebar import sidebar  
from transform import sketch2fashion
# to run type python3.11 -m streamlit run main.py
sidebar()

st.title(':red[Generating Clothing Visualization Using Sketches] :dress:')
st.subheader(
    'Upload your sketch here or take a picture to see the results:exclamation::exclamation:')

uploaded_file = st.file_uploader(" ", type=['jpg', 'png', 'jpeg'])

picture = st.camera_input("Take a picture ! :camera_with_flash:")

if picture is not None and uploaded_file is None:
    # with open(os.path.join("C:/Users/hp/streamlit-test/",picture.name),"wb") as f:
    #       f.write(picture.getbuffer())
    sketch2fashion(picture.name, "converted.png")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Before :pencil:")
        st.image(picture)

        btn = st.download_button(
            label="Download image :arrow_down:",
            data=picture,
            file_name="sketch.png",
            mime="image/png"
        )
    with col2:
        st.header("After :lower_left_paintbrush:")
        progress_text = "Operation in progress. Please wait. :clock2:"
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1, text=progress_text)

        sketch = Image.open("converted.png")
        st.image(sketch)

        btn = st.download_button(
            label="Download image :arrow_down:",
            data=picture,
            file_name="dress.png",
            mime="image/png"
        )


if uploaded_file is not None:  # and picture is none:
    image = Image.open(uploaded_file)
    with open(os.path.join(uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

    sketch2fashion(uploaded_file.name, 'converted.png')

    col1, col2 = st.columns(2)
    with col1:
        st.header("Before :pencil:")
        st.image(image)

    with col2:
        st.header("After :lower_left_paintbrush:")
        progress_text = "Operation in progress. Please wait. :clock2:"
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            my_bar.progress(percent_complete + 1, text=progress_text)

        sketch = Image.open("converted.png")
        st.image(sketch)
        if st.button("Regenerate respone :arrows_counterclockwise:"):
            sketch2fashion("converted.png", "converted.png")
            sketch = Image.open("converted.png")
            st.image(sketch)
