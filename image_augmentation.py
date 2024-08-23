import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import image_mask_gen
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import io
import warnings
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

import streamlit as st
import base64


# Function to display points on the image using matplotlib
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def remove_duplicates(coords, labels):
    unique_coords = []
    unique_labels = []
    seen = set()

    for coord, label in zip(coords, labels):
        coord_tuple = tuple(coord)
        if coord_tuple not in seen:
            seen.add(coord_tuple)
            unique_coords.append(coord)
            unique_labels.append(label)
            
    return unique_coords, unique_labels


def image_augmentation_page():
    pass
    st.title("Image Augmentation")
    st.write("Upload an image to apply augmentation techniques.")

    # Initialize session state variables
    if "inclusive_points" not in st.session_state:
        st.session_state.inclusive_points = []
    if "exclusive_points" not in st.session_state:
        st.session_state.exclusive_points = []
    
    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)

        # Set the maximum width for display
        max_display_width = 700  # You can adjust this value

        # Calculate the scaling factor
        scale_factor = min(max_display_width / image.size[0], 1)

        # Resize the image for display
        display_width = int(image.size[0] * scale_factor)
        display_height = int(image.size[1] * scale_factor)
        resized_image = image.resize((display_width, display_height))

        # Inclusive Points Phase
        st.subheader("Select Inclusive Points (Green)")
        canvas_inclusive = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
            stroke_width=1,                # Stroke width for drawing
            stroke_color="blue",           # Color for the outline of clicks
            background_image=resized_image,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="circle",         # Drawing mode to capture clicks as circles
            point_display_radius=3,        # Radius of the circle that represents a click
            key="canvas_inclusive"
        )

        # Process inclusive clicks
        if canvas_inclusive.json_data is not None:
            objects = canvas_inclusive.json_data["objects"]
            new_clicks = [[(obj["left"] + obj["radius"]) / scale_factor, (obj["top"] + obj["radius"]) / scale_factor] for obj in objects]
            st.session_state.inclusive_points.extend(new_clicks)

        # Plot the inclusive points on the original image using Matplotlib
        fig_inclusive, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')  # Hide the axes

        # Prepare data for plotting
        inclusive_points = np.array(st.session_state.inclusive_points)
        labels_inclusive = np.array([1] * len(st.session_state.inclusive_points))

        # Call the function to show inclusive points
        if len(inclusive_points) > 0:
            show_points(inclusive_points, labels_inclusive, ax)

        st.pyplot(fig_inclusive)

        # Divider
        st.divider()

        # Exclusive Points Phase
        st.subheader("Select Exclusive Points (Red)")
        canvas_exclusive = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
            stroke_width=1,                # Stroke width for drawing
            stroke_color="blue",           # Color for the outline of clicks
            background_image=resized_image,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="circle",         # Drawing mode to capture clicks as circles
            point_display_radius=3,        # Radius of the circle that represents a click
            key="canvas_exclusive"
        )

        # Process exclusive clicks
        if canvas_exclusive.json_data is not None:
            objects = canvas_exclusive.json_data["objects"]
            new_clicks = [[(obj["left"] + obj["radius"]) / scale_factor, (obj["top"] + obj["radius"]) / scale_factor] for obj in objects]
            st.session_state.exclusive_points.extend(new_clicks)

        # Plot the exclusive points on the original image using Matplotlib
        fig_exclusive, ax = plt.subplots()
        ax.imshow(image)
        ax.axis('off')  # Hide the axes

        # Prepare data for plotting
        exclusive_points = np.array(st.session_state.exclusive_points)
        labels_exclusive = np.array([0] * len(st.session_state.exclusive_points))

        # Call the function to show exclusive points
        if len(exclusive_points) > 0:
            show_points(exclusive_points, labels_exclusive, ax)

        st.pyplot(fig_exclusive)

        # Grouping coordinates and labels
        coordinates = st.session_state.inclusive_points + st.session_state.exclusive_points
        labels = [1] * len(st.session_state.inclusive_points) + [0] * len(st.session_state.exclusive_points)

        # # Display grouped coordinates and labels
        # st.subheader("Coordinates and Labels")
        # st.write("Coordinates: ", tuple(coordinates))
        # st.write("Labels: ", labels)

        # Provide an option to clear the coordinates
        if st.button("Clear All Points"):
            st.session_state.inclusive_points = []
            st.session_state.exclusive_points = []
        # global unique_coordinates, unique_labels
        unique_coordinates, unique_labels = remove_duplicates(coordinates, labels)

        st.write("Unique Coordinates:", tuple(unique_coordinates))
        st.write("Unique Labels:", tuple(unique_labels))

        # image_mask_gen.show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)
        sam2_checkpoint = "sam2_hiera_base_plus.pt"
        model_cfg = "sam2_hiera_b+.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")

        predictor = SAM2ImagePredictor(sam2_model)

        image = image
        predictor.set_image(image)

        input_point = np.array(unique_coordinates)
        input_label = np.array(unique_labels)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        mask_input = logits[np.argmax(scores), :, :]

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
        image_mask_gen.show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

        
        # Get masked images
        original_image = Image.open(uploaded_file)
        # st.image(original_image, caption='Original Image', use_column_width=True)

        with st.container(border=True):# Display masked images
            col1, col2 = st.columns(2)
            with col1:
                mask_images = image_mask_gen.show_masks_1(original_image, masks, scores)
                for idx, (img, score) in enumerate(mask_images):
                    st.image(img, caption=f'Mask {idx+1}, Score: {score:.3f}', use_column_width=True)
            with col2:
                inverse_mask_images = image_mask_gen.show_inverse_masks(original_image, masks, scores)
                for idx, (img, score) in enumerate(inverse_mask_images):
                    st.image(img, caption=f'Inverse Mask {idx+1}, Score: {score:.3f}', use_column_width=True)
        
        if st.checkbox("Proceed to Image Augmentation"):
        
            image_aug_select = st.sidebar.selectbox("Select Augmentation for Mask",["Pixelate","Hue Change","Mask Replacement","Generative Img2Img"])
            if image_aug_select == "Pixelate":
                
                if st.sidebar.toggle("Proceed to Pixelate Mask"):
                    pixelation_level = st.slider("Select Pixelation Level", min_value=5, max_value=50, value=10)
                    combined_image = image_mask_gen.combine_pixelated_mask(original_image, masks[0], pixelation_level)
                    st.image(combined_image, caption="Combined Pixelated Image", use_column_width=True)
            elif image_aug_select == "Hue Change":

                if st.sidebar.toggle("Proceed to Hue Change"):
                    # Hue shift slider
                    hue_shift = st.slider("Select Hue Shift", min_value=-180, max_value=180, value=0)
                    # Apply hue change and show the result
                    combined_image = image_mask_gen.combine_hue_changed_mask(original_image, masks[0], hue_shift)  # Assuming single mask
                    st.image(combined_image, caption="Combined Hue Changed Image", use_column_width=True)
            elif image_aug_select == "Mask Replacement":

                if st.sidebar.toggle("Proceed to replace Mask"):
                    replacement_file = st.file_uploader("Upload the replacement image", type=["png", "jpg", "jpeg"])
                    if replacement_file is not None:
                        replacement_image = Image.open(replacement_file) #.convert("RGBA")
                        combined_image = image_mask_gen.combine_mask_replaced_image(original_image, replacement_image, masks[0])  # Assuming single mask
                        st.image(combined_image, caption="Masked Area Replaced Image", use_column_width=True)
            elif image_aug_select == "Generative Img2Img":
        
                msk_img = None
                mask_images_x = image_mask_gen.show_masks_1(original_image, masks, scores)
                for idx, (img, score) in enumerate(mask_images_x):
                    msk_img = img
                    # st.image(img, caption=f'Mask {idx+1}, Score: {score:.3f}', use_column_width=True)

                rgb_image = msk_img.convert("RGB")
                # st.image(rgb_image)
                resized_image = image_mask_gen.resize_image(rgb_image)
                # st.image(resized_image, caption=f"Resized size: {resized_image.size[0]}x{resized_image.size[1]}", use_column_width=True)
                width, height = resized_image.size
                
                # User input for the prompt and API key
                prompt = st.text_input("Enter your prompt:", "A Beautiful day, in the style reference of starry night by vincent van gogh")
                api_key = st.text_input("Enter your Stability AI API key:")

                if prompt and api_key:
                    # Set up our connection to the API.
                    os.environ['STABILITY_KEY'] = api_key
                    stability_api = client.StabilityInference(
                        key=os.environ['STABILITY_KEY'], # API Key reference.
                        verbose=True, # Print debug messages.
                        engine="stable-diffusion-xl-1024-v1-0", # Set the engine to use for generation.
                    )
                    style_preset_selector = st.sidebar.selectbox("Select Style Preset",["3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk",
                                                                "origami", "photographic", "pixel-art", "tile-texture"],index = 5)
                    if st.sidebar.toggle("Proceed to Generate Image"):
                        # Set up our initial generation parameters.
                        answers2 = stability_api.generate(
                            prompt=prompt,
                            init_image=resized_image, # Assign our uploaded image as our Initial Image for transformation.
                            start_schedule=0.6,
                            steps=250,
                            cfg_scale=10.0,
                            width=width,
                            height=height,
                            sampler=generation.SAMPLER_K_DPMPP_SDE,
                            style_preset=style_preset_selector
                        )

                        # Process the response from the API
                        for resp in answers2:
                            for artifact in resp.artifacts:
                                if artifact.finish_reason == generation.FILTER:
                                    warnings.warn(
                                        "Your request activated the API's safety filters and could not be processed."
                                        "Please modify the prompt and try again.")
                                if artifact.type == generation.ARTIFACT_IMAGE:
                                    img2 = Image.open(io.BytesIO(artifact.binary))
                                    # Display the generated image
                                    st.image(img2, caption="Generated Image", use_column_width=True)

                                    # Combine the generated image with the original image using the mask
                                    combined_img = image_mask_gen.combine_mask_and_inverse_gen(original_image, img2, masks[0])
                                    st.image(combined_img, caption="Combined Image", use_column_width=True)