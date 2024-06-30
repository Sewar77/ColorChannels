import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
image = cv.imread("resizes__face_frame_21.jpg")

# Convert the image to HSV color space
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Define the number of bins for H, S, and V channels
h_bins = 180
s_bins = 256
v_bins = 256

# Define the ranges for H, S, and V channels
h_range = [0, 180]
s_range = [0, 256]
v_range = [0, 256]

# Create a list of ranges for all channels
ranges = [h_range, s_range, v_range]

# Calculate the histogram
hist = cv.calcHist([hsv_image], [0, 1, 2], None, [h_bins, s_bins, v_bins], ranges)

# Plot the histogram
plt.figure()
plt.title("HSV Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.imshow(hist, interpolation="nearest")
plt.show()
