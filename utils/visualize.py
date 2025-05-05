# utils/visualization.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
import io
import base64

def display_image(image_arr, boxes_minmax=None, box_color='lime'):
    """
    Displays an image with optional bounding boxes using Matplotlib.

    Args:
        image_arr (torch.Tensor): The image tensor (C, H, W).
        boxes_minmax (list of torch.Tensor, optional): A list of bounding box tensors
                                                       in [x_min, y_min, x_max, y_max] format.
                                                       Defaults to None.
        box_color (str, optional): The color of the bounding boxes. Defaults to 'lime'.
    """
    image_arr = image_arr.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image_arr)

    if boxes_minmax is not None:
        for box_tensor in boxes_minmax:
            # Move the tensor to CPU before converting to NumPy
            box = box_tensor.detach().cpu().numpy()
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def encode_matplotlib_to_base64():
    """
    Encodes the current Matplotlib figure to a Base64 string (PNG format).

    Returns:
        str: A Base64 encoded string of the Matplotlib figure, or None if no figure exists.
    """
    fig = plt.gcf()
    if fig is None:
        return None
    try:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        print(f"Error encoding Matplotlib to Base64: {e}")
        return None

def save_matplotlib_to_file(filepath):
    """
    Saves the current Matplotlib figure to a file.

    Args:
        filepath (str): The path to save the figure (e.g., 'static/plots/temp.png').
    """
    try:
        plt.savefig(filepath)
    except Exception as e:
        print(f"Error saving Matplotlib to file: {e}")

def clear_matplotlib_figure():
    """
    Clears the current Matplotlib figure to free up resources.
    """
    plt.clf()
    plt.close()