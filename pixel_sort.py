import numpy as np
from PIL import Image, ImageChops

BLEND_MODES = {"blend": (lambda img1, img2: Image.blend(img1, img2, 0.6)), "darker": ImageChops.darker,
               "lighter": ImageChops.lighter, "none": (lambda img1, img2: img2), "multiply": ImageChops.multiply, "overlay": ImageChops.overlay, "hardlight": ImageChops.hard_light}
SORT_FIELDS = {"hue": 0, "saturation": 1, "value": 2}


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
image = pixel_sort(source)
image.show()
