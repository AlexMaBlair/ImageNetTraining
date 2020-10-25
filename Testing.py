from PIL import Image
im = Image.open('cda.jpg')

width, height = im.size  # Get dimensions

# Crop the center of the image
left = (width - 300) / 2
top = (height - 300) / 2
right = (width + 300) / 2
bottom = (height + 300) / 2


im = im.crop((left, top, right, bottom))

im.show()