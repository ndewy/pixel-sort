import cv2
import numpy as np
from PIL import Image, ImageChops

BLEND_MODES = {"blend": (lambda img1, img2: Image.blend(img1, img2, 0.6)), "darker": ImageChops.darker,
               "lighter": ImageChops.lighter, "none": (lambda img1, img2: img2), "multiply": ImageChops.multiply, "overlay": ImageChops.overlay, "hardlight": ImageChops.hard_light}
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


def get_largest_foreground_region(img):
    """ Creates an image containing only the largest forground region.

        Returns an image with the size of the input. 
        This image will contain the largest foreground contour (by area) and a transparent background.

        This function works best on portraits or other images with a subject that takes up most of the frame.

        Arguments:
            img: a PIL image.

        Returns: 
            img: an RGBA PIL image.
    """

    opencv_img = __pil_to_cv(img)
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


def pixel_sort(src_img, mode="none", sort_field="hue", skip_interval=0):
    image = src_img.convert(mode="HSV")
    image_array = np.array(image)
    results = []
    sort_key = SORT_FIELDS[sort_field]

    for i in range(image.height):
        if (skip_interval == 0) or not i % skip_interval == 0:
            sorted_row = sorted(image_array[i], key=lambda pix: pix[sort_key])
            results.append(sorted_row)
            print("row {} out of {}".format(i, image.height))
            continue
        # Skipped row
        results.append(image_array[i])
        print("row {} out of {} skipped".format(i, image.height))

    result = Image.fromarray(np.array(results), mode="HSV").convert("RGB")

    blended_img = BLEND_MODES[mode](src_img, result)
    return blended_img


source = Image.open("testimage2.jpg")

__cv_to_pil(cv_threshold(__pil_to_cv(source))).show()
get_largest_foreground_region(source).show()

image = channel_shift(source)
image.show()

image = pixel_sort(source)
image.show()
