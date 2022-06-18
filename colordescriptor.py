import numpy as np
import cv2
import imutils


class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins
        
	
    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])
	
        hist = cv2.normalize(hist, hist).flatten()
        return hist


    def describe(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]

        for (startX, endX, startY, endY) in segments:
            
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        return features