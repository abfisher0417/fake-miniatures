import os
import numpy as np
import cv2
import imutils


def updateYCoordinates(coords, y_value):
	"""Takes a list of (x, y) coordinates and set the y value to be a specific value
	"""
	return [(x, y_value) for x,y in coords]

def updateXCoordinates(coords, x_value):
	"""Takes a list of (x, y) coordinates and set the x value to be a specific value
	"""
	return [(x_value, y) for x,y in coords]

def centerPoint(image):
	"""Returns the middle point of an image in (x, y) format
	"""
	shape = image.shape
	cx = shape[1] // 2
	cy = shape[0] // 2
	return cx, cy

def rotatePoints(center_point, points, angle):
	"""Given a set of (x, y) coordinates, returns the set of (x, y) coordinates rotated
	   about a center point by a given angle.
	"""
	points = [rotatePoint(center_point[0], center_point[1], x, y, angle) for x,y in points]
	return points

def rotatePoint(cx, cy, x, y, angle):
	"""Rotates a point by a given angle around a center point
	   Source: https://stackoverflow.com/questions/2259476/rotating-a-point-about-another-point-2d
	"""
	s = np.sin(angle * np.pi / 180.)
	c = np.cos(angle * np.pi / 180.)
	x -= cx
	y -= cy
	xnew = x * c - y * s
	ynew = x * s + y * c
	x = xnew + cx
	y = ynew + cy
	return x, y

def rotateCanvas(image, angle, save_image=False):
	"""Rotates an image canvas in degrees
	   Source: https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
	"""
	rotated_image = imutils.rotate_bound(image, angle)

	if save_image:
		saveImage(rotated_image, 'rotateCanvas%dDegrees.png' % angle)

	return rotated_image

def saveImage(image, filename):
	"""Saves an image to the output folder
	"""
	cur_dir = os.getcwd()
	out_dir = os.path.join(cur_dir, "output")
	out_file = os.path.join(out_dir, filename)
	cv2.imwrite(out_file, image)

def consistentBlur(image, ksize, gaussian_or_box='gaussian', save_image=False):
	"""Applies a gaussian or box blur to an entire image
	"""
	height, width = image.shape[:2]

	if gaussian_or_box == 'gaussian':
		blur_img = cv2.GaussianBlur(image,(ksize,ksize),0)
		blur_img = cv2.GaussianBlur(blur_img,(ksize,ksize),0)
	elif gaussian_or_box == 'box':
		blur_img = cv2.blur(image,(ksize,ksize))
		blur_img = cv2.blur(blur_img,(ksize,ksize))

	if save_image:
		saveImage(blur_img, 'consistentBlur.png')

	return blur_img

def gradientMask(image, y1, y2, save_image):
	"""Creates a linear gradient mask
	"""
	height, width = image.shape[:2]
	mask_img = np.zeros((height, width), dtype=np.uint8)
	mask_img.fill(255)

	# Do a linear gradient from 255 to 0
	y1 = height - int(y1)
	for i in range(0, 40):
		mix_percent = i / 40
		color = (1 - mix_percent) * 255 + mix_percent * 0
		mask_img[y1 - 40 + i, :] = int(color)

	# Do a linear gradient from 0 to 255
	y2 = int(y2)
	mask_img[y1:height - y2, :] = 0

	for i in range(0, np.min([int(y2), 40])):
		mix_percent = i / np.min([int(y2), 40])
		color = (1 - mix_percent) * 0 + mix_percent * 255
		mask_img[height - y2 + i, :] = int(color)

	if save_image:
		saveImage(mask_img, 'gradientMask.png')

	return mask_img

def alphaBlend(image_1, image_2, mask_image, save_image=False):
	"""Performs an alpha blend between 2 images using a mask
	"""
	alpha = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR) / 255.0
	blended = cv2.convertScaleAbs(image_1 * (1 - alpha) + image_2 * alpha)
	if save_image:
		saveImage(blended, 'alphaBlend.png')
	return blended

def cropFromCenter(image, target_width, target_height, save_image=False):
	"""Crops an image from the center point
	"""
	cx, cy = centerPoint(image)
	target_top = cy - target_height // 2
	target_left = cx - target_width // 2
	target_bottom = target_top + target_height
	target_right = target_left + target_width
	cropped_image = image[target_top:target_bottom, target_left:target_right, :]
	if save_image:
		saveImage(cropped_image, 'cropFromCenter.png')
	return cropped_image

def cropImage(image, tolerance=0, save_image=False):
    """Crops an image using a bounding box
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = grayscale_image > tolerance
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    cropped_image = image[x0+16:x1-15, y0+16:y1-15, :]

    if save_image:
        saveImage(cropped_image, 'cropImage.png')
    return cropped_image

def increaseSaturation(image, multiplier, save_image=False):
	"""Increases saturation level of an image
       Source: http://answers.opencv.org/question/193336/how-to-make-an-image-more-vibrant-in-colour-using-opencv/
	"""
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hsv_image[...,1] = hsv_image[...,1] * multiplier
	image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
	if save_image:
		saveImage(image, 'increaseSaturation.png')
	return image

def applyEffect(img, y1, y2, theta, sat, gaussian_or_box='gaussian', save_image=False):
    """Applies the tilt-shift effect to an input image
    """
    image = np.copy(img)
    w, h = image.shape[:2]
    increased_saturation_img = increaseSaturation(image, sat, save_image=save_image)
    rotated_canvas_img = rotateCanvas(increased_saturation_img, -1 * theta, save_image=save_image)
    blur_img = consistentBlur(rotated_canvas_img, 7, gaussian_or_box=gaussian_or_box, save_image=save_image)
    gradient_mask = gradientMask(rotated_canvas_img, y1, y2, save_image=save_image)
    alpha_blend = alphaBlend(rotated_canvas_img, blur_img, gradient_mask, save_image=save_image)
    rotated_back_img = rotateCanvas(alpha_blend, 1 * theta, save_image=save_image)
    final_image = cropImage(rotated_back_img, tolerance=0, save_image=save_image)
    return final_image
