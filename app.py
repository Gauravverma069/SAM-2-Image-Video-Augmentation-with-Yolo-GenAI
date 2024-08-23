import streamlit as st
import base64

# Set the page configuration
st.set_page_config(
    page_title="MetaMorph AI",
    page_icon="🌉",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'Get help': 'https://www.linkedin.com/in/gaurav-verma-4696bb106/',
        'About': "MetaMorph: Revolutionize your media with cutting-edge image and video augmentation using the META Sam-2 model for stunning visual transformations!"
    }
)

# Function to load video as base64
def get_base64_video(video_path):
    with open(video_path, 'rb') as video_file:
        video_bytes = video_file.read()
    return base64.b64encode(video_bytes).decode('utf-8')

# Video file path
video_path = 'images/background.mp4'

# Get the base64 video
video_base64 = get_base64_video(video_path)

# Add video as background
background_video = f"""
    <style>
    .stApp {{
        background: transparent;
    }}
    .video-container {{
        position: fixed;
        top: 0;
        left: 0;
        min-width: 100%;
        min-height: 100%;
        z-index: -1;
        overflow: hidden;
    }}
    .video-container video {{
        position: absolute;
        top: 50%;
        left: 50%;
        width: auto;
        height: auto;
        min-width: 100%;
        min-height: 100%;
        transform: translate(-50%, -50%);
        opacity: 0.5;
    }}
    .content {{
        position: relative;
        z-index: 1;
        padding-top: 50px;
    }}
    </style>
    <div class="video-container">
        <video autoplay loop muted>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
    </div>
    """
st.markdown(background_video, unsafe_allow_html=True)

# Content goes here
with st.container():
    
    # Title
    html_code = """
    <div class="content">
        <div class="title-container">
          <h1 class="neon-text">
            MetaMorphix AI 🐦‍🔥
          </h1>
        </div>
    </div>

    <style>
    @keyframes rainbow-text-animation {
      0% { color: white; }
      16.67% { color: grey; }
      33.33% { color: grey; }
      50% { color: black; }
      66.67% { color: grey; }
      83.33% { color: white; }
      100% { color: black; }
    }

    .title-container {
      text-align: center;
      margin: 1em 0;
      padding-bottom: 10px;
      border-bottom: 4px solid #fcdee9;
    }

    .neon-text {
      font-family: Trebuchet MS , sans-serif;
      font-size: 4em;
      margin: 0;
      animation: rainbow-text-animation 5s infinite linear;
      text-shadow: 0 0 5px rgba(0, 255, 0, 0.8),
                   0 0 10px rgba(0, 255, 255, 0.7),
                   0 0 20px rgba(0, 255, 255, 0.6),
                   0 0 40px rgba(0, 0, 0, 0.6),
                   0 0 80px rgba(0, 0, 0, 0.6),
                   0 0 90px rgba(0, 0, 0, 0.6),
                   0 0 100px rgba(0, 0, 255, 0.6),
                   0 0 150px rgba(0, 0, 255, 0.6);
    }
    </style>
    """
    st.markdown(html_code, unsafe_allow_html=True)

    # Additional content
    
# Functionality for pages
from home import home_page
from image_augmentation import image_augmentation_page
from video_augmentation import image_annoter
from use_cases import use_case
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ("Home","Use Cases", "Image Augmentation", "Video Augmentation"))

    if page == "Home":
        home_page()
    elif page == "Use Cases":
        use_case()
    elif page == "Image Augmentation":
        image_augmentation_page()
    elif page == "Video Augmentation":
        image_annoter()

if __name__ == "__main__":
    main()
