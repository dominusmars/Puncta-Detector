# Puncta Analyzer

This is a python script used to identify Puncta in fluorescence microscopy images.

## set up environment

use pip install -r requirements.txt
to install needed deps

## Help

usage: scan.py [-h] [--brightness BRIGHTNESS] [--tolerance TOLERANCE] [--surrounding SURROUNDING] [--blur BLUR] [--image IMAGE] [--gui GUI]

options:
-h, --help show this help message and exit
--brightness BRIGHTNESS
Set brightness level 0-255
--tolerance TOLERANCE
Upper level of brightness, brightness + tolerance
--surrounding SURROUNDING
The surrounding radius checked
--blur BLUR The blur set to the image, set to 1 for no blur
--image IMAGE Path to image
--gui GUI Interactive mode

## Interactive

Interactive mode lets you adjust parameters as seen fit. It gives you a bunch of windows to help you figure out whats being analyzed.

```sh
python ./src/scan.py --image {imagePath} --gui 1
```

To save, press the '-' key on the keyboard this will create a new folder called data and timestamp the save.

To quit, press the 'q' key on the keyboard.

# Methodology

Images are taken using a microscope. cropped images work best for this.

Images are then gray scaled to extract brightness values.

After gray scaling, Images are then normalized so that the brightest part of the image is 255 and darkest being 0.
This provides fine tuning of the brightness levels. To adjust as needed.

A Mask is created to identify Puncta based on the brightness threshold. Brightness threshold is based on the modified image not the original image.

Mask is then overlaid with the original image to find the true values of the puncta.
