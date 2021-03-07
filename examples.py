import pixel_sort
from PIL import Image
portrait1 = Image.open("examples/source-images/portrait1.jpg")
portrait2 = Image.open("examples/source-images/portrait2.jpg")
moon = Image.open("examples/source-images/moon1.jpg")
moonnoisy = Image.open("examples/source-images/moon1noisy.jpg")

# moon1
img = moon.transpose(Image.ROTATE_90)
img = pixel_sort.pixel_sort(img,sort_field="saturation",flipped=True,skip_percent=0,mode="none").transpose(Image.ROTATE_270)
img.save("examples/moon1.jpg","JPEG")

# moon1 - noisy
# Adding noise creates a substancially more exaggerated effect.
img = moonnoisy.transpose(Image.ROTATE_90)
img = pixel_sort.pixel_sort(img,sort_field="saturation",flipped=True,skip_percent=0,mode="none").transpose(Image.ROTATE_270)
img.save("examples/moon1noisy.jpg","JPEG")
