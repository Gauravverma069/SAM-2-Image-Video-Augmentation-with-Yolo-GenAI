import streamlit as st

def use_case():
    st.title("Video Augmentation Use Cases")

    st.markdown("### 1. Face Blur in Privacy Protection")
    st.write("""
    In scenarios where privacy is a concern, such as public surveillance or social media content, blurring faces is crucial to protect identities. 
    Video augmentation techniques can automatically detect and blur faces in video footage, ensuring compliance with privacy regulations and protecting individuals' identities.
    """)
    st.video('images/pix_output_video (1).mp4')

    st.markdown("### 2. Enhanced Video Editing and Post-Production")
    st.write("""
    In video production, object masks allow editors to isolate and manipulate specific elements within a scene. 
    Whether it’s changing backgrounds, applying effects, or removing unwanted objects, the app’s masking capabilities make complex editing tasks more accessible and efficient.
    """)
    col1 ,col2 = st.columns(2)
    with col1:
        st.video('images/zoe.mp4')
    with col2:
        st.video("images/redhulk.mp4")
    
    st.markdown("### 3. Content Creation and Entertainment")
    st.write("""
    In media and entertainment, content creators often need innovative and visually appealing effects in videos. 
    Video augmentation can apply artistic filters, color grading, and other visual effects, allowing creators to experiment with different styles and generate engaging content quickly.
    """)
    st.video('images/with_replacement_output_video.mp4')

    st.markdown("### 4. Creative Content Generation with Generative AI")
    st.write("""
    By leveraging generative AI on object masks, the app can transform or replace the masked areas with entirely new content. 
    For example, in advertising or entertainment, a product or character can be dynamically altered to fit different themes or environments, providing unique and personalized experiences for viewers.
    This technique can also be used in film production to create special effects, in digital art to generate novel compositions, or in marketing to produce customized visuals that resonate with diverse audiences.
    """)
    st.video('images/genai shaolin.mp4')

    st.divider()

    st.header("Uncharted Use cases")

    st.markdown("### 5. Data Augmentation for Limited Datasets")
    st.write("""
    When working with limited video data, augmentation can help create a larger and more diverse dataset without additional data collection. 
    Techniques like temporal jittering, speed variations, color adjustments, and geometric transformations can generate synthetic videos, improving model performance.
    """)

    st.markdown("### 6. Enhancing Security and Surveillance Systems")
    st.write("""
    Security and surveillance systems rely on accurate detection and tracking of objects or individuals in various environments. 
    Video augmentation simulates different lighting conditions, weather effects (rain, fog), and camera angles, improving detection algorithms' robustness in real-world scenarios.
    """)

    st.markdown("### 7. Improving Autonomous Driving Systems")
    st.write("""
    Autonomous vehicles need to understand and react to various road conditions, lighting scenarios, and unexpected obstacles. 
    Video augmentation techniques like altering weather conditions, introducing random obstacles, and simulating different times of day help train resilient autonomous driving systems.
    """)

    

    st.markdown("### 8. Medical Video Analysis")
    st.write("""
    In medical diagnostics, especially in video-based analysis like endoscopy or ultrasound, the quality and diversity of video data are critical for accurate diagnoses. 
    Video augmentation can create variations in medical videos by adjusting contrast, adding noise, or simulating different imaging conditions, helping train more robust and accurate AI models.
    """)

    st.markdown("### 9. Sports Analytics and Player Performance Evaluation")
    st.write("""
    Sports analytics involve evaluating player performance from video footage, which can vary across games and conditions. 
    Augmenting videos with changes in speed, perspective, or focus can simulate different game scenarios, improving player tracking, action recognition, and strategy analysis.
    """)

    st.markdown("### 10. E-commerce and Virtual Try-Ons")
    st.write("""
    E-commerce platforms offering virtual try-ons for clothing or accessories need to simulate various conditions such as different lighting or angles. 
    Augmenting product videos helps create a more realistic virtual try-on experience, improving customer engagement and satisfaction.
    """)
