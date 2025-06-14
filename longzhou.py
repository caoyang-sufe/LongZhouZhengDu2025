# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import cv2
import time
import numpy
import pyautogui

from PIL import Image
from numpy.linalg import norm
from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse import Button, Controller as MouseController

import numpy as np

Y1, Y2, X1, X2 = 230, 330, 850, 1070	# 这组参数适配1920×1080

color_to_key = {'y': 'd', 'r': 'f', 'b': 'j', 'g': 'k'}

def easy_show(image: numpy.ndarray, window_title: str="image") -> None:
	"""
	Show a image in a new window.
	"""
	cv2.imshow(window_title,image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def get_screenshot():
	image = pyautogui.screenshot()
	image.save(f"./temp/sc.png")
	image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
	return image

def load_labels():
	labels = dict()
	for color in "yrbg":
		for n in "012":
			image = Image.open(f"./temp/{color}{n}.png")
			image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
			clipped_image = image[Y1: Y2, X1: X2, :]
			# jfeasy_show(clipped_image)
			labels[f"{color}{n}"] = clipped_image
	return labels

# 判断旗帜颜色及点击次数
def classify_label(screenshot, labels):
	clipped_image = screenshot[Y1: Y2, X1: X2, :]
	diffs = {}
	for key, label in labels.items():
		diff = norm(label - clipped_image)
		diffs[key] = diff
	print(diffs)
	predicted_label = min(diffs, key = lambda x: diffs[x])
	return predicted_label

def run():
	keyboard_controller = KeyController()
	while True:
		screenshot = get_screenshot()
		labels = load_labels()
		predicted_label = classify_label(screenshot, labels)
		color = predicted_label[0]
		key_char = color_to_key[color]
		n = int(predicted_label[1])
		if n == 0:
			# 禁止旗
			for key_char_rep in "dfjk":
				if key_char_rep != key_char:
					keyboard_controller.press(key_char_rep)
					time.sleep(.05)
					keyboard_controller.release(key_char_rep)
					time.sleep(.05)
					break
		else:
			for _ in range(n):
				keyboard_controller.press(key_char)
				time.sleep(.05)
				keyboard_controller.release(key_char)
				time.sleep(.05)
		time.sleep(.15)
	
time.sleep(2)
run()
# get_screenshot()