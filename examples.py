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

# portrait1
portrait1 = Image.open("examples/source-images/portrait1.jpg")
portrait1.thumbnail((1000,1000))
noisy = pixel_sort.add_noise(portrait1,0.005)
mask = pixel_sort.get_largest_foreground_region(noisy,threshold_img=portrait1).transpose(Image.ROTATE_90)
img = pixel_sort.pixel_sort(mask,region_max=0.5,region_min=0.2,sort_field="value",skip_percent=0)
img = img.transpose(Image.ROTATE_270)
img.save("examples/portrait1.jpg")