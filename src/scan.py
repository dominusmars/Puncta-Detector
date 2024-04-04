#!/usr/bin/python3.11
#!python
from ast import arg
import csv
import cv2
import os
import time
import json
import keyboard
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

Mat = np.ndarray[np.uint8]

def displayImage(title, image):
    if not args.gui:
        return
    scaled_image = cv2.resize(image, None, fx=1, fy=1)
    cv2.imshow(title, scaled_image)    
    
def makeBoarder(image):
    return cv2.copyMakeBorder(image, 0, 0, 2, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def preProcessImage():
    global image
    global gray_image
    global normalized_image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX)



def processImage():
    global image
    global image_mask
    global analysis
    global gray_image
    global image_mask_color
    global normalized_image
    global overlaid_image
    global brightness_threshold
    global tolerance
    global average_intensity
    global average_surrounding_intensity
    global aiSD
    global asiSD 
    global blur
    global process_image

    time.sleep(0.2)

    process_image = cv2.blur(normalized_image, (blur,blur))
    
    displayImage("gray",process_image)

    image_mask = np.zeros_like(process_image, dtype=np.uint8)
    image_mask[(process_image >= brightness_threshold)  & (process_image <= brightness_threshold + tolerance)] = 255
    
    # Connect Pixel and analysis connected groups
    analysis  = cv2.connectedComponentsWithStats(image_mask, 8, cv2.CV_32S)
    average_intensity, aiSD, average_surrounding_intensity, asiSD =Compute( analysis,image, image_mask)
    fig = create_average_intensities_chart(analysis,average_intensity)
    figSD = create_average_intensities_SD_chart(analysis, aiSD)
    
    # average_surrounding_intensity, asiSD = average_intensity_surrounding_per_label(image, analysis, image_mask)
    
    fig_twoSD = create_average_surround_intensities_SD_chart(analysis, asiSD)
    
    
    fig_two = create_average_surrounding_intensities_chart(analysis, average_surrounding_intensity)
    
    plt.close(fig)
    plt.close(fig_two)
    plt.close(figSD)
    plt.close(fig_twoSD)
   
    
    
    (num_labels, label_map, stats, centroids) = analysis
    groups = num_labels - 1    

    
    # Color pixels on mask
    image_mask_color = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2BGR)

    mask = np.any(image_mask_color > 0, axis=2)
    image_mask_color[mask] = (blue,green,red)


    # Draw bounding boxes around each labeled component
    for label in range(1, num_labels):
        # Get the stats of the current component
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        # Put the label index on the image
        cv2.putText(image_mask_color, str(label), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (blue,green,red), 1)
    
    displayImage("Mask Image", image_mask_color)
    
    print(f"Number of pixel groups: {groups} Brightness: {brightness_threshold} Tolerance: {tolerance} Blur: {blur} Surround: {surrounding}")
      
    overlaid_image = cv2.addWeighted(image, 1, image_mask_color, 1, 0)

    # Merge the original image and the labeled image
    combined_image = np.hstack((makeBoarder(image), makeBoarder(overlaid_image)))
    
    displayImage('Original And Overlay', combined_image)
    if args.gui: 
        cv2.imshow('Average Intensity Chart', chart_image_average)
        cv2.imshow('Standard Deviation Average Intensity Chart', chart_image_average_SD)
        cv2.imshow('Standard Deviation Surround Average Intensity Chart', chart_image_average_surround_SD)
        cv2.imshow('Average Surround Intensity Chart', chart_image_average_surrounding)
        
    
    if not args.gui:
        saveData()
    
    
def Compute(analysis: tuple[int, Mat, Mat, Mat],image:np.ndarray ,mask:np.ndarray):
    (num_labels, label_map, stats, _) = analysis
    average_intensities = []
    sd = []
    average_surrounding_intensities = []
    sd_surrounding = []
    # Iterate through each label
    for label in range(1, num_labels):  # Skip background label 0
        average_intensity_label, sd_label = compute_average_intensity(label,label_map )
        average_intensities.append(average_intensity_label)
        sd.append(sd_label)
        average_intensities_surround, sd_surrounding_label = compute_average_intensity_surround(label,stats, image, mask)
        average_surrounding_intensities.append(average_intensities_surround)
        sd_surrounding.append(sd_surrounding_label)
        
    return average_intensities, sd, average_surrounding_intensities,sd_surrounding
        
def compute_average_intensity_surround(label, stats,image, mask):
     # Extract bounding box coordinates
    left = stats[label, cv2.CC_STAT_LEFT] + surrounding
    top = stats[label, cv2.CC_STAT_TOP] + surrounding 
    width = stats[label, cv2.CC_STAT_WIDTH] + surrounding
    height = stats[label, cv2.CC_STAT_HEIGHT] + surrounding

    # Calculate center of the square
    center_x = left + width // 2
    center_y = top + height // 2

    # Calculate radius of the circle
    radius = min(width, height) // 2

    # Extract pixels corresponding to the bounding box
    bounding_box_pixels = image[top:top+height, left:left+width]
    bounding_box_mask = mask[top:top+height, left:left+width]

    # Calculate indices of circle pixels
    yy, xx = np.ogrid[top:top+height, left:left+width]
    circle_indices = ((xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2)

    # Mask out circle pixels where mask value is 255
    circle_indices &= (bounding_box_mask != 255)

    # Extract circle pixels and calculate statistics
    circle_pixels = bounding_box_pixels[circle_indices]

    # Calculate the average intensity of the pixels within the circle
    average_intensity = np.mean(circle_pixels)

    return average_intensity, np.std(circle_pixels)
    
def compute_average_intensity(label,label_map):
    # Extract pixels corresponding to the label
    label_pixels = gray_image[label_map == label]

    # Calculate the average intensity and standard deviation of the pixels
    average_intensity = np.mean(label_pixels)
    standard_deviation = np.std(label_pixels)

    return average_intensity, standard_deviation


def create_average_intensities_chart(analysis, average_intensities):
    global chart_image_average
    (num_labels, _, _, _) = analysis

    # Create a DataFrame with label numbers and average intensities
    data = {'Label': range(1, num_labels), 'Average Intensity': average_intensities}
    df = pd.DataFrame(data)

    # Plot the DataFrame directly onto a Matplotlib figure
    fig, ax = plt.subplots()
    ax.bar(df['Label'], df['Average Intensity'], color='blue')
    ax.set_ylabel('Average Intensity (grayed image)')
    ax.set_xlabel('Label')
    ax.set_title('Average Intensity per Label')
    
    # Convert the Matplotlib figure to a NumPy array
    fig.canvas.draw()
    chart_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    chart_image = chart_image.reshape((fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4))
    chart_image_average = chart_image

    return fig

def create_average_intensities_SD_chart(analysis, sd):
    global chart_image_average_SD
    (num_labels, _, _, _) = analysis

    # Create a DataFrame with label numbers and average intensities
    data = {'Label': range(1, num_labels), 'Standard Deviation': sd}
    df = pd.DataFrame(data)

    # Plot the DataFrame directly onto a Matplotlib figure
    fig, ax = plt.subplots()
    ax.bar(df['Label'], df['Standard Deviation'], color='blue')
    ax.set_ylabel('Standard Deviation (grayed image)')
    ax.set_xlabel('Label')
    ax.set_title('Standard Deviation per Label')
    
    # Convert the Matplotlib figure to a NumPy array
    fig.canvas.draw()
    chart_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    chart_image = chart_image.reshape((fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4))
    chart_image_average_SD = chart_image

    return fig

def create_average_surround_intensities_SD_chart(analysis, sd):
    global chart_image_average_surround_SD
    (num_labels, _, _, _) = analysis

    # Create a DataFrame with label numbers and average intensities
    data = {'Label': range(1, num_labels), 'Surround Standard Deviation': sd}
    df = pd.DataFrame(data)

    # Plot the DataFrame directly onto a Matplotlib figure
    fig, ax = plt.subplots()
    ax.bar(df['Label'], df['Surround Standard Deviation'], color='blue')
    ax.set_ylabel('Surround Standard Deviation (grayed image)')
    ax.set_xlabel('Label')
    ax.set_title('Surround Standard Deviation per Label')
    
    # Convert the Matplotlib figure to a NumPy array
    fig.canvas.draw()
    chart_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    chart_image = chart_image.reshape((fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4))
    chart_image_average_surround_SD = chart_image
    
    return fig

def create_average_surrounding_intensities_chart(analysis, average_surrounding_intensities):
    global chart_image_average_surrounding

    (num_labels, _, _, _) = analysis

    # Create a DataFrame with label numbers and average intensities
    data = {'Label': range(1, num_labels), 'Average Surrounding Intensity': average_surrounding_intensities}
    df = pd.DataFrame(data)

    # Plot the DataFrame directly onto a Matplotlib figure
    fig, ax = plt.subplots()
    ax.bar(df['Label'], df['Average Surrounding Intensity'], color='blue')
    ax.set_ylabel('Average Intensity (grayed image)')
    ax.set_xlabel('Label')
    ax.set_title('Average Surrounding Intensity per Label')
    
    # Convert the Matplotlib figure to a NumPy array
    fig.canvas.draw()
    chart_image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    chart_image = chart_image.reshape((fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4))
    chart_image_average_surrounding= chart_image
 
    return fig
    
    
def saveData():
    now = time.time()
    
    now = datetime.now()
    try:
        os.mkdir("./data")
    except: 
        pass
    
    path = os.path.join("./data/" , get_filename_without_extension(filePath) + "_" + now.strftime("%m_%d_%Y_%H_%M_%S"))
    try:
        os.mkdir(path)
    except:
        print("error creating dir")
        pass
    (num_labels, label_map, stats, centroids) = analysis

    cv2.imwrite(path + "/image.png", image)
    cv2.imwrite(path + "/grayscaled.png", process_image)
    cv2.imwrite(path + "/mask.png", image_mask_color)
    cv2.imwrite(path + "/overlay.png", overlaid_image)
    
    cv2.imwrite(path + "/average_intensities_chart.png",chart_image_average )
    cv2.imwrite(path + "/average_intensities_SD_chart.png",chart_image_average_SD )
    cv2.imwrite(path + "/average_surrounding_intensities.png",chart_image_average_surrounding )
    cv2.imwrite(path + "/average_surrounding_SD_intensities.png",chart_image_average_surround_SD )


    data = {
        "Brightness": brightness_threshold,
        "Tolerance": tolerance,
        "Blur": blur,
        "Surround": surrounding,
        "Groups": num_labels,
        "Average Intensity": np.mean(average_intensity),
        "Average Intensity of Surround": np.nanmean(np.array(average_surrounding_intensity)),
        "Objects": 0
    }
    objects  = []
    
    # Analyze objects and add them to the data
    for label in range(1, num_labels):
        object_data = {}

        # Get the stats of the current component
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])

        # Calculate centroid coordinates
        centroid_x = centroids[label, 0]
        centroid_y = centroids[label, 1]

        # Add object properties to the object_data dictionary
        object_data["Label"] = label
        object_data['x'] = centroid_x
        object_data['y'] = centroid_y 
        # object_data["Position"] = {"X": centroid_x, "Y": centroid_y}
        object_data['Left'] = left
        object_data["Top"] = top
        object_data["Width"] = width
        object_data["Height"] = height
        object_data["Area"] = area
        
        # print(label)
        if(label < len(average_intensity) -1):
            object_data['Average Intensity'] = average_intensity[label]
            object_data["Average Intensity Standard Deviation"] = aiSD[label]
            object_data['Average Surrounding Intensity'] = average_surrounding_intensity[label]
            object_data["Average Surrounding Intensity Standard Deviation"] =asiSD[label]
            
        
        # Add the object data to the list of objects in the data dictionary
        objects.append(object_data)
    data['Objects'] = len(objects)
    
    
    try:
        with open(path+ '/data.json', 'w') as f:
            json.dump(data, f, indent=4)
        
        # array_to_csv(data, path+"/data.csv")
        array_to_csv(objects, path+"/objects.csv")
            
        print("Data successfully written to", path)
    except Exception as e:
        print("Error writing data to JSON file:", str(e))


def array_to_csv(array, file_name):
    with open(file_name, 'w', newline='') as file:
        # Create a CSV writer object
        csv_writer = csv.writer(file)
        
        # Write the header row based on the keys of the first JSON object
        header = array[0].keys()
        csv_writer.writerow(header)
        
        # Write the JSON objects as rows in the CSV file
        for item in array:
            csv_writer.writerow(item.values())

def createLabelImage(labels):
    global labeled_image
    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_image = cv2.merge([label_hue, blank_ch, blank_ch])
    # labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_HSV2BGR)
    labeled_image[labels > 0] = [blue, green, red]  # Set all labels to green

    # labeled_image[label_hue == 0] = 0
    labeled_image[label_hue == 0] = [0, 0, 0]

    return labeled_image

def onColorChangeRed(value):
    global red
    global ready
    red = value
    if ready:
        processImage()
    
def onBlurChange(value):
    global blur
    global ready
    
    blur = value
    if ready:
        processImage()
    
def onColorChangeBlue(value):
    global blue
    global ready
    
    blue = value
    if ready:
        processImage()
        
def onColorChangeGreen(value):
    global green
    global ready
    
    green = value
    if ready:
        processImage() 

# Callback function for the trackbar
def on_brightness(value):
    global brightness_threshold
    global ready
    
    brightness_threshold = value
    # Recreate the brightness mask
    if ready:
        processImage()

def on_tolerance(value):
    global tolerance
    global ready
    
    tolerance = value
    # Recreate the brightness mask
    if ready:
        processImage()
    
def on_surrounding(value):
    global surrounding
    global ready
    surrounding = value
    # Recreate the brightness mask
    if ready:
        processImage()

def get_filename_without_extension(file_path: str) -> str:
    # Get the base filename from the path
    base_filename = os.path.basename(file_path)

    # Split the filename into name and extension
    name, extension = os.path.splitext(base_filename)

    return name



parser = argparse.ArgumentParser()
parser.add_argument('--brightness', dest='brightness', type=int, help='Set brightness level 0-255')
parser.add_argument('--tolerance', dest='tolerance', type=int, help='Upper level of brightness, brightness + tolerance')
parser.add_argument('--surrounding', dest='surrounding', type=int, help='The surrounding radius checked')
parser.add_argument('--blur', dest='blur', type=int, help='The blur set to the image, set to 1 for no blur')
parser.add_argument('--image', dest='image', type=str, help='Path to image')
parser.add_argument('--gui', dest='gui', type=bool, help='Interactive mode')



args = parser.parse_args()


filePath = args.image

# Read the input image
image = cv2.imread(filePath)

# Set the initials


try:
    brightness_threshold = args.brightness
except:
    brightness_threshold = 150
if brightness_threshold is None:
    brightness_threshold = 150
    
try:
    tolerance = args.tolerance
except:
    tolerance = 255
if tolerance is None:
    tolerance = 255
        

try:
    surrounding = args.surrounding
except:
    surrounding = 3
if surrounding is None:
    surrounding = 3
# Set Colors for mask
red = 255
green = 255
blue = 255
ready = False
try:
    blur = args.blur
except:
    blur = 1
if blur is None:
    blur = 1



if __name__ == '__main__':

    # Display the original image
    # cv2.imshow('Original Image', image)

    # process Image
    preProcessImage()
    processImage()
    if args.gui:
        
    # Create a window to display the labeled image
        cv2.namedWindow('Options')
        cv2.namedWindow('MaskColor Setting')
        cv2.resizeWindow("Options", 900, 250)
        cv2.resizeWindow("MaskColor Setting", 900, 200)
        
        # Create a trackbar to adjust the brightness threshold
        cv2.createTrackbar('Brightness', 'Options', brightness_threshold, 255, on_brightness)
        cv2.createTrackbar('Tolerance', 'Options', tolerance, 255, on_tolerance)
        cv2.createTrackbar('Surrounding', 'Options', surrounding, 10, on_surrounding)
        
        cv2.createTrackbar('Mask Blue', 'MaskColor Setting', blue, 255, onColorChangeBlue)
        cv2.createTrackbar('Mask Green', 'MaskColor Setting', green, 255, onColorChangeGreen)
        cv2.createTrackbar('Mask Red', 'MaskColor Setting', red, 255, onColorChangeRed)
        cv2.createTrackbar('Blur', 'Options', blur, 10, onBlurChange)
        keyboard.on_press_key("-", lambda _:saveData())
        ready = True
        while True:
            # Wait for a key event
            key = cv2.waitKey(0)
            # If 'q' is pressed or data saving is complete, break the loop
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    else:
        print("Saved")


    
    
