import pixel_sort
from PIL import Image
source_1 = Image.open("examples/portrait1.jpg")


# === CHANNEL SHIFT ALL
# image = pixel_sort.channel_shift(source)
# image.show()

# === PIXEL SORT ALL ===
# image = pixel_sort.pixel_sort(source)
# image.show()

# === PIXEL SORT FACE ONLY + CHANNEL SHIFTS===
img = pixel_sort.channel_shift(source_1,change_percent=-0.01,channel=0)
img = pixel_sort.channel_shift(source_1,change_percent=0.02,channel=1)

masked_image = pixel_sort.get_largest_foreground_region(img)
masked_image.show()
image = pixel_sort.pixel_sort(masked_image,skip_interval=5,flipped=True)
image.show()
