import cv2
import os

import numpy as np


def detect_circles(path):
	""" 
	Detects circles in the given images that have a radius up to 
	or bigger than 10px radius.

	:param path: path to the image to be processed.
	"""

	# Read the image.
	image = cv2.imread(path)
	# Make a copy for the final output.
	output = image.copy()
	# Convert it to grayscale.
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Apply erosion and then dilation to the image to remove outer noise.
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(24, 24))
	image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	# Apply some edge detection to further remove unwanted shapes.
	image = cv2.Canny(image, 225, 255)

	# Detect the circles.
	circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.4, 100,
                            param1=50, param2=60, minRadius=10, maxRadius=100)

	# Check if any circles were found.
	if circles is not None:
		# Convert values in circles array to integers.
		circles = np.round(circles[0, :]).astype("int")

		# Draw outlines over the output image in all found circles.
		for (x, y, r) in circles:
			cv2.circle(output, (x, y), r, (0, 255, 255), 2)

	cv2.imshow('Solution', output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':

	# The paths for the images to be processed.
	images = [os.path.abspath('../challenge/imagens/shapes_leo.jpg'),
			   os.path.abspath('../challenge/imagens/circles.png')]

	for i in images:
		detect_circles(i)