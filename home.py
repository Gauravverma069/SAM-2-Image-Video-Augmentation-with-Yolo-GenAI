import streamlit as st



def home_page():
    st.title("Welcome to MetaMorphix AI")
    st.write("""
    This application uses the **META Sam-2 model** to perform advanced augmentation on images and videos.,
             \n**YOLO** trained and pretrained model for Object Detection.
             \n**Stability AI API** for Generative AI - Image to Image generation on mask.
             \n**Image Annoter** for YOLO training Folder Input, Process Replica That of Roboflow app.

    Navigate to the desired section using the sidebar.

    \nScroll down to see the tutorial.

    """)
    st.divider()
    st.header("For Image Augmentation")
    st.write("""1. Navigate to Image Augmentation page & Upload a Image.
            \n2. Mark coordinates on canvas **(green for Inclusive points & red for Exclusive points).**
            \n3. Select Augmentaion method [Pixelated, Hue Change, Mask Replacement, Img2Img Generation] and proceed.""")
    st.video("images/image_aug.mp4")

    st.divider()
    st.header("For Image Annotation on an Image Directory")
    st.write("""1. Navigate to Video Augmentation page & Paste Local Directory link where train images are to annoted.
            \n2. create Bounding box on canvas.
            \n3. click on save annoptation and navigate through next button""")
    st.video("images/image_annote.mp4")

    st.warning("As of now Video Augmentation can only be happen on Jupyter notebook due to certain Limitation")
    st.write("Go to following link to access Notebook and Use Kaggle GPU")
    # Define the profile link
    profile_url = "https://www.kaggle.com/code/gauravverma069/sam-2-meta-video-augmentation-with-yolo-and-genai"
    st.markdown(f"[Visit my Kaggle Notebook link]({profile_url})")

    
    

  