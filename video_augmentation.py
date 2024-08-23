import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os


def image_annoter():
    st.title("Image Annoter for YOLO")
    st.write("Enter image files folder Location.")

    # Upload video
    file_location = st.text_input("Enter File folder Location",None)
    
    if file_location is not None:

        # Folder containing your images
        label_folder = file_location

        # Get a list of all images in the folder
        image_files = [f for f in os.listdir(label_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

        # Initialize session state to keep track of the current image index
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0

        # Set the downscaling factor
        downscale_factor = 0.5  # Adjust the downscale factor as needed

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        if col1.button("Previous"):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1

        if col3.button("Next"):
            if st.session_state.current_index < len(image_files) - 1:
                st.session_state.current_index += 1

        # Display the current image
        current_image_file = image_files[st.session_state.current_index]
        st.write(f"Annotating: {current_image_file}")
        image_path = os.path.join(label_folder, current_image_file)
        image = Image.open(image_path)

        # Downscale the image for the canvas
        scaled_width = int(image.width * downscale_factor)
        scaled_height = int(image.height * downscale_factor)
        scaled_image = image.resize((scaled_width, scaled_height))

        # Display the image on the canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",  # Fill color for the bounding box
            stroke_width=3,
            stroke_color="#ff0000",
            background_image=scaled_image,
            height=scaled_height,
            width=scaled_width,
            drawing_mode="rect",
            key=current_image_file
        )

        # Save annotations
        if st.button("Save Annotation"):
            if canvas_result.json_data is not None:
                # Extract coordinates of the bounding box
                for obj in canvas_result.json_data["objects"]:
                    if obj["type"] == "rect":
                        left = obj["left"] / downscale_factor
                        top = obj["top"] / downscale_factor
                        width = obj["width"] / downscale_factor
                        height = obj["height"] / downscale_factor
                        x_center = left + width / 2
                        y_center = top + height / 2

                        # Normalize the coordinates (YOLO format)
                        x_center /= image.width
                        y_center /= image.height
                        width /= image.width
                        height /= image.height

                        # Save the annotation to a .txt file
                        annotation_path = os.path.join(label_folder, current_image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                        with open(annotation_path, 'w') as f:
                            f.write(f"0 {x_center} {y_center} {width} {height}\n")

                st.success(f"Annotation saved for {current_image_file}")
