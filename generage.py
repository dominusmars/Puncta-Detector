from PIL import Image, ImageDraw
import random

# Image size
width = 800
height = 600

# Create a new image
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Number of dots
num_dots = 1000

# Draw random dots
for _ in range(num_dots):
    # Random position
    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)
    
    # Random size
    size = random.randint(1, 10)
    
    # Random brightness
    brightness = random.randint(0, 255)
    
    # Draw the dot
    draw.ellipse((x - size, y - size, x + size, y + size), fill=(brightness, brightness, brightness))

# Save the image
image.save("dots_picture.png")

print("Image generated successfully!")
