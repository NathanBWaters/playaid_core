import numpy as np
from bokeh.io.export import get_screenshot_as_png
import re


def graph_to_image(graph, width=400, height=360):
    return np.array(get_screenshot_as_png(graph, height=height, width=width))


def is_ascii(s=""):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in
    # python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def paste_on_top(src, dst, x, y):
    """
    Paste src onto dst at position (x, y) where src alpha > 1.
    Assumes both images are in RGBA format.
    """
    if x < 0 or y < 0:
        return dst

    # Get dimensions of source and destination images
    h_src, w_src = src.shape[:2]
    h_dst, w_dst = dst.shape[:2]

    if x >= w_dst or y >= h_dst:
        return dst

    # Calculate dimensions of the region in dst where src will be pasted
    h_paste = min(h_src, h_dst - y)
    w_paste = min(w_src, w_dst - x)

    # Crop src and mask to fit within dst
    src_cropped = src[:h_paste, :w_paste]
    dst_region = dst[y : y + h_paste, x : x + w_paste]

    # Ensure the images are in float32 format for accurate division
    src_cropped = src_cropped.astype(np.float32)
    dst_region = dst_region.astype(np.float32)

    # Split the images into RGB and alpha channels
    src_rgb = src_cropped[..., :3]
    src_alpha = src_cropped[..., 3:4] / 255.0  # Normalize the alpha channel to be in [0, 1]
    dst_rgb = dst_region[..., :3]
    dst_alpha = dst_region[..., 3:4] / 255.0  # Normalize the alpha channel to be in [0, 1]

    # Perform alpha blending
    out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
    out_rgb = (src_rgb * src_alpha + dst_rgb * dst_alpha * (1 - src_alpha)) / np.where(out_alpha == 0, 1, out_alpha)  # Avoid division by zero

    # Use np.where to fill the regions where alpha is zero with the original color
    out_rgb = np.where(out_alpha == 0, dst_rgb, out_rgb)

    # Concatenate the RGB and alpha channels to get the final image
    out = np.concatenate([out_rgb, out_alpha * 255], axis=-1)

    # Place the blended region back into the destination image
    dst[y : y + h_paste, x : x + w_paste] = out.astype(np.uint8)

    return dst

def split_text_emoji(text):
    pattern = re.compile(r"([\u263a-\U0001f645])|(\w+)|(\s+)|([0-9])")
    return [match.group() for match in pattern.finditer(text)]
