import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_mask(image_cv, mask, color=(0, 255, 0), alpha=0.5):
    """ Apply a mask to an image with given color and alpha blend """
    mask_bgr = np.zeros_like(image_cv)
    mask_bgr[mask > 0] = color
    return cv2.addWeighted(image_cv, 1 - alpha, mask_bgr, alpha, 0)

def draw_points(image_cv, points, labels):
    """ Draw points on the image with different colors based on labels """
    for coord, label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)  # Green for inclusive, Red for exclusive
        cv2.circle(image_cv, tuple(map(int, coord)), 5, color, -1)
    return image_cv

def draw_boxes(image_cv, boxes):
    """ Draw boxes on the image """
    for box in boxes:
        x, y, w, h = map(int, box)
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red boxes
    return image_cv

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    image_cv = np.array(image.convert("RGB"))[..., ::-1]  # Convert PIL image to BGR format for OpenCV

    for i, (mask, score) in enumerate(zip(masks, scores)):
        image_with_mask = apply_mask(image_cv, mask)
        
        if point_coords is not None:
            assert input_labels is not None
            image_with_mask = draw_points(image_with_mask, point_coords, input_labels)

        if box_coords is not None:
            image_with_mask = draw_boxes(image_with_mask, box_coords)

        # Convert back to RGB and then to PIL for Streamlit
        image_with_mask = cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_with_mask)
        
        # Display the final image with all overlays
        st.image(image_pil, caption=f"Mask {i+1}, Score: {score:.3f}", use_column_width=True)


def apply_mask_to_image(image, mask):
    # Ensure the image is a NumPy array in BGR format
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create an alpha channel based on the mask
    alpha_channel = (mask * 255).astype(np.uint8)

    # Create an image with the mask applied only on masked areas
    masked_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    for c in range(3):  # Apply the mask only to the RGB channels
        masked_image[..., c] = image[..., c] * mask

    # Add the alpha channel to make areas outside the mask transparent
    masked_image[..., 3] = alpha_channel

    return masked_image

def show_masks_1(image, masks, scores):
    mask_images = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Apply the mask to the image
        masked_image = apply_mask_to_image(image, mask)

        # Convert the masked image to PIL format for Streamlit
        pil_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGRA2RGBA))
        mask_images.append((pil_image, score))

    return mask_images


def apply_inverse_mask_to_image(image, mask):
    # Ensure the image is a NumPy array in BGR format
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create an alpha channel that is transparent inside the mask and opaque outside
    alpha_channel = (1 - mask) * 255

    # Create an image with the mask applied to the inverse areas
    inverse_masked_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    for c in range(3):  # Apply the inverse mask to RGB channels
        inverse_masked_image[..., c] = image[..., c] * (1 - mask)

    # Add the alpha channel to make areas inside the mask transparent
    inverse_masked_image[..., 3] = alpha_channel.astype(np.uint8)

    return inverse_masked_image

def show_inverse_masks(image, masks, scores):
    mask_images = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Apply the inverse mask to the image
        inverse_masked_image = apply_inverse_mask_to_image(image, mask)

        # Convert the masked image to PIL format for Streamlit
        pil_image = Image.fromarray(cv2.cvtColor(inverse_masked_image, cv2.COLOR_BGRA2RGBA))
        mask_images.append((pil_image, score))

    return mask_images

import streamlit as st
import cv2
import numpy as np
from PIL import Image

def combine_mask_and_inverse(image, mask):
   
    # Ensure the image is a NumPy array in BGR format
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    # Apply the mask to get the masked region (in original color)
    masked_region = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

    # Apply the inverse mask to get the inverse-masked region (in original color)
    inverse_mask = 1 - mask
    inverse_masked_region = cv2.bitwise_and(image, image, mask=inverse_mask.astype(np.uint8))

    # Combine both masked and inverse-masked regions
    combined_image = cv2.add(masked_region, inverse_masked_region)

    # Convert to RGBA format for transparency
    combined_image_rgba = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGBA)

    return combined_image_rgba

def show_combined_masks(image, masks, scores):
    
    mask_images = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Combine masked and inverse masked areas
        combined_image = combine_mask_and_inverse(image, mask)

        # Convert the combined image to PIL format for Streamlit
        pil_image = Image.fromarray(combined_image)
        mask_images.append((pil_image, score))

    return mask_images


def pixelate_area(image, mask, pixelation_level):
    """
    Apply pixelation to the masked area of an image.
    """
    pixelated_image = image.copy()
    h, w, _ = image.shape

    for y in range(0, h, pixelation_level):
        for x in range(0, w, pixelation_level):
            block = (slice(y, min(y + pixelation_level, h)), slice(x, min(x + pixelation_level, w)))
            if np.any(mask[block]):
                mean_color = image[block].mean(axis=(0, 1)).astype(int)
                pixelated_image[block] = mean_color

    return pixelated_image

def combine_pixelated_mask(image, mask, pixelation_level=10):
    """
    Combine the pixelated masked areas with the original image.
    """
    image_np = np.array(image)
    mask_np = np.array(mask)

    pixelated_mask = pixelate_area(image_np, mask_np, pixelation_level)
    combined_image = Image.fromarray(pixelated_mask)
    return combined_image


def change_hue(image, mask, hue_shift):
   
    # Convert the image from RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_RGB2HSV)

    # Apply the hue shift to the masked area
    hsv_image[..., 0] = (hsv_image[..., 0] + hue_shift) % 180

    # Convert back to RGB format
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Combine the hue-changed area with the original image using the mask
    hue_changed_image = np.array(image).copy()
    hue_changed_image[mask] = np.concatenate((rgb_image[mask], hue_changed_image[mask][..., 3:]), axis=-1)

    return hue_changed_image

def combine_hue_changed_mask(image, mask, hue_shift):
   
    image_np = np.array(image)
    mask_np = np.array(mask).astype(bool)

    hue_changed_area = change_hue(image_np, mask_np, hue_shift)
    combined_image = Image.fromarray(hue_changed_area)

    return combined_image

def replace_masked_area(original_image, replacement_image, mask):
    # Ensure the replacement image is the same size as the original image
    replacement_image = cv2.resize(replacement_image, (original_image.shape[1], original_image.shape[0]))

    # Create a copy of the original image
    replaced_image = original_image.copy()

    # Replace the masked area with the corresponding area from the replacement image
    replaced_image[mask] = replacement_image[mask]

    return replaced_image

def combine_mask_replaced_image(original_image, replacement_image, mask):

    # Convert images to NumPy arrays
    original_np = np.array(original_image)
    replacement_np = np.array(replacement_image)
    mask_np = np.array(mask).astype(bool)

    # Replace the masked area
    replaced_area = replace_masked_area(original_np, replacement_np, mask_np)
    combined_image = Image.fromarray(replaced_area)

    return combined_image

import streamlit as st
from PIL import Image

def resize_image(image, max_size=1024):
    # Get the current width and height of the image
    width, height = image.size

    # Calculate the scaling factor
    if width > height:
        scaling_factor = max_size / width
    else:
        scaling_factor = max_size / height

    # Only resize if the image is larger than the max_size
    if scaling_factor < 1:
        # Calculate new dimensions
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Resize the image
        image_resized = image.resize((new_width, new_height))
        return image_resized
    else:
        # Return the original image if it's already within the size limits
        return image


def combine_mask_and_inverse_gen(original_img, generated_img, mask):
    # Ensure images are in RGBA mode
    original_img = original_img.convert("RGBA")
    generated_img = generated_img.convert("RGBA")
    
    # Resize the generated image to match the original image size
    generated_img = generated_img.resize(original_img.size)
    
    # Convert images to arrays
    orig_array = np.array(original_img)
    gen_array = np.array(generated_img)
    
    # Resize the mask to match the original image size
    mask = Image.fromarray((mask * 255).astype(np.uint8))  # Convert mask to image for resizing
    mask = mask.resize(original_img.size, Image.NEAREST)   # Resize the mask
    bool_mask = np.array(mask).astype(bool)
    
    # Ensure the mask has the correct shape (H, W, 1)
    if bool_mask.ndim == 2:
        bool_mask = bool_mask[:, :, np.newaxis]
    
    # Combine images using the mask
    combined_array = np.where(bool_mask, gen_array, orig_array)
    
    # Convert combined array back to image
    combined_img = Image.fromarray(combined_array, "RGBA")
    return combined_img
