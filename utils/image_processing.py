from PIL import Image
from io import BytesIO
import numpy as np
import base64
import albumentations as A
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

def process_image(jpg_image):
    """
    Processes JPG image data provided as bytes.

    Args:
        jpg_image_bytes (bytes): The raw bytes of the JPG image.

    Returns:
        numpy.ndarray or None: A processed NumPy array representing the JPG image,
                               or None if an error occurred.
    """

    

    try:
        transform = A.Compose([
            A.Resize(height=240, width=240),
            ToTensorV2()
        ])
        image = Image.open(BytesIO(jpg_image)).convert('RGB')

        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)
        image_np = np.array(image_tensor)
        if image_np.shape[0] == 3:
            image_np = np.transpose(image_np, (1, 2, 0))

        orginal_X = image_np.shape[1]
        orginal_Y = image_np.shape[0]

        image_final = transform(image=image_np)['image']

        return image_final, orginal_X, orginal_Y

    except Exception as e:
        print(f"Error processing JPG image bytes: {e}")
        return None


def scale_bbox(bbox, org_X, org_Y, current_X=240, current_Y=240):
    """
    Scales a bounding box from the current image dimensions to the original dimensions.

    Args:
        bbox (list or tuple): A bounding box in [x_min, y_min, x_max, y_max] format
                              relative to the current image dimensions.
        org_X (int): The width of the original image.
        org_Y (int): The height of the original image.
        current_X (int): The width of the current image (after transformation).
                         Defaults to 240.
        current_Y (int): The height of the current image (after transformation).
                         Defaults to 240.

    Returns:
        list: The scaled bounding box in [x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled]
              relative to the original image dimensions.
    """
    x_min, y_min, x_max, y_max = bbox

    scale_x = org_X / current_X
    scale_y = org_Y / current_Y

    x_min_scaled = x_min * scale_x
    y_min_scaled = y_min * scale_y
    x_max_scaled = x_max * scale_x
    y_max_scaled = y_max * scale_y

    return [x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled]
