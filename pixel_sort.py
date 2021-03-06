import random
from math import ceil

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageOps
from skimage.util import random_noise

BLEND_MODES = {"none": (lambda img1, img2: img2), "add": ImageChops.add, "blend": (lambda img1, img2: Image.blend(img1, img2, 0.6)), "darker": ImageChops.darker, "difference": ImageChops.difference, "lighter": ImageChops.lighter,
               "multiply": ImageChops.multiply, "hardlight": ImageChops.hard_light, "softlight": ImageChops.soft_light, "overlay": ImageChops.overlay, "screen": ImageChops.screen, "subtract": ImageChops.subtract}
SORT_FIELDS = {"hue": 0, "saturation": 1, "value": 2}


def __pil_to_cv(img):
    numpy_img = np.array(img.convert("RGB"))
    opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    return opencv_img


def __cv_to_pil(img):
    color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)
    return pil_img


def cv_threshold(img):
    """ Seperates an image into foreground and background using thresholding and Otsu's algorithm.'

    Arguments:
        img: a BGR OpenCV image.

    Returns: 
        threshold: an OpenCV Binary image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gaussian blur image to remove noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # determine optimal threshold value using Otsu, and threshold using it
    ret, threshold = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold


def add_noise(img, amount=0.01, mean=0):
    # Wraps skimage function with auto conversion to/from PIL.
    # Mean = overall brightness, Varience = amount of noise
    sk_img = np.array(img)
    noise_img = random_noise(sk_img, mode="speckle", var=amount, mean=mean)
    # Convert -1->1 RGB to 0-> 255 RGB
    noise_img = (255*noise_img).astype(np.uint8)
    pil_img = Image.fromarray(noise_img)
    return pil_img


def get_largest_foreground_region(img, threshold_img=None):
    """ Creates an image containing only the largest forground region.

        Returns an image with the size of the input. 
        This image will contain the largest foreground contour (by area) and a transparent background.

        This function works best on portraits or other images with a subject that takes up most of the frame.

        Arguments:
            img: a PIL image.
            threshold_img: (optional). The image that will be passed to the thresholder. Defaults to img. Useful if
                                        img is noisy, and a less noisy version is available to threshold with.

        Returns: 
            img: an RGBA PIL image.
    """
    if threshold_img is None:
        threshold_img = img
    opencv_img = __pil_to_cv(threshold_img)
    threshold = cv_threshold(opencv_img)
    contours, hierarchy = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_byarea = sorted(contours, key=lambda n: cv2.contourArea(n))
    largest_contour = contours_byarea[-1]
    # Generate image to draw contour upon
    new_img = np.zeros((img.height, img.width, 3), np.uint8)
    # Draw contour onto image
    new_img = cv2.drawContours(
        new_img, [largest_contour], 0, (255, 255, 255), thickness=cv2.FILLED)
    mask = __cv_to_pil(new_img).convert("L")
    img = img.convert("RGBA")
    img.putalpha(mask)
    return img


def channel_shift(src_img, change_percent=-0.1, mode="none", channel=0):
    image = src_img.convert("RGB")
    change_x = change_percent*image.width
    channels = list(image.split())
    channels[channel] = channels[channel].rotate(0, translate=(change_x, 0))
    result = Image.merge(mode="RGB", bands=channels)

    blended_img = BLEND_MODES[mode](image, result)
    return blended_img


def pixel_sort(src_img, mode="none", sort_field="hue", skip_percent=0.3, flipped=False, region_max=0.3, region_min=0.1):

    img = src_img.convert(mode="HSV")
    img_array = np.array(img)

    # The alpha channel is used only as a mask to selectively apply pixel sort.
    # The final image is fully opaque and does use it.
    alpha_channel = src_img.convert("RGBA").getchannel("A")
    alpha_array = np.array(alpha_channel)

    result_array = []
    sort_key = SORT_FIELDS[sort_field]

    for i in range(src_img.height):

        start_pixel = 0
        startpixel_modified = False
        end_pixel = src_img.width + 1

        # Find start and end pixels.
        for j in range(src_img.width):
            # Start from the first non-transparent pixel
            if not startpixel_modified and alpha_array[i][j] != 0:
                start_pixel = j
                startpixel_modified = True
            # End at the last non-transparent pixel
            if alpha_array[i][j] != 0:
                end_pixel = j

        row_values = []

        # Is this row a skip row?
        is_skip_row = False
        # Pick a random number between 0 and 1
        num = random.random()
        # If number is below the percentage, skip it
        # As this uses a uniform distribution, we know that, overall, 30% of picks will be below 0.3, and so on.
        if num <= skip_percent:
            is_skip_row = True

        if (not is_skip_row) and startpixel_modified:
            # Fill the region up to the startpoint with the original pixels.
            row_values += [img_array[i][j] for j in range(0, start_pixel)]

            # Split pixels from start_pixel -> end_pixel into random length regions.
            regions = []
            current_pixel = start_pixel
            while current_pixel < end_pixel:
                region_length = random.randint(
                    ceil((end_pixel-start_pixel)*region_min), ceil((end_pixel-start_pixel)*region_max))
                region_end = current_pixel + region_length

                if region_end > end_pixel:
                    region_end = end_pixel

                regions.append(img_array[i][current_pixel:region_end])
                current_pixel = region_end

            # Sort each region and append.
            for region in regions:
                sorted_region = sorted(
                    region, key=lambda pix: pix[sort_key], reverse=flipped)
                row_values += sorted_region

            # Fill in the rest of the row after the endpoint with the original pixels.
            row_values += [img_array[i][j]
                           for j in range(end_pixel, src_img.width)]

            result_array.append(row_values)
            print("row {} out of {}".format(i, src_img.height))
            assert len(
                row_values) == src_img.width, "Row length not equal to image width"
            continue

        # Skipped row - populate the row with it's original pixels.
        result_array.append(img_array[i])
        print("row {} out of {} skipped".format(i, src_img.height))

    result = Image.fromarray(np.array(result_array),
                             mode="HSV").convert("RGB")
    blended_img = BLEND_MODES[mode](src_img.convert("RGB"), result)
    return blended_img
