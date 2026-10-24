import cv2
import numpy as np


def preprocess_image_from_bytes(image_bytes: bytes, target_size: int = 109):
  """Processes raw image bytes sent via FastAPI:

  1. Decodes uploaded bytes into an OpenCV image.
  2. Removes hair using fine-tuned Black-Hat filtering & inpainting.
  3. Pads to a square aspect ratio using the mode skin-tone color.
  4. Resizes to the model target resolution (109x109).
  """
  # 1. Convert bytes to NumPy array and decode
  nparr = np.frombuffer(image_bytes, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  if img is None:
    raise ValueError(
        "Could not decode uploaded image. Ensure it is a valid image file."
    )

  # 2. Hair Removal (Optimized for 109x109 scale)
  grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Scaled kernel for fine hair structures
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

  # Sensitive threshold and mask dilation for thin hairs
  _, thresh = cv2.threshold(blackhat, 4, 255, cv2.THRESH_BINARY)
  dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
  dilated_mask = cv2.dilate(thresh, dilation_kernel, iterations=1)

  hair_removed = cv2.inpaint(
      img, dilated_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA
  )

  # 3. Square Padding using Mode Skin-Tone Color
  h, w, c = hair_removed.shape
  max_dim = max(h, w)

  # Compute mode skin-tone color from the image pixels
  pixels = hair_removed.reshape(-1, 3)
  mode_color = np.array(
      [np.bincount(pixels[:, i]).argmax() for i in range(3)], dtype=np.uint8
  )

  # Create square canvas filled with mode color
  square_img = np.full((max_dim, max_dim, c), mode_color, dtype=np.uint8)

  # Center the image onto the canvas
  y_offset = (max_dim - h) // 2
  x_offset = (max_dim - w) // 2
  square_img[y_offset : y_offset + h, x_offset : x_offset + w] = hair_removed

  # 4. Final resize to target model resolution (109x109)
  processed_img = cv2.resize(
      square_img, (target_size, target_size), interpolation=cv2.INTER_AREA
  )

  return processed_img