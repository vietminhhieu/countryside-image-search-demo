import numpy as np
import cv2
import imutils


class ColorDescriptor:
    def __init__(self, bins):
        self.bins = bins
        
	
    def histogram(self, image, mask):
        # TS1: image, TS2: mỗi màu của hệ HSV, TS3: mask, TS4: bin, 
        # TS5: miền giá trị của từng kênh màu HSV của ảnh 8bit
        #tính tần suất xuất hiện của giá trị cường độ mức xám đối với từng kênh màu
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,[0, 180, 0, 256, 0, 256])

        # chuyển sang dạng xác suất để có thể dùng CT tri bình phương
        hist = cv2.normalize(hist, hist).flatten()

        return hist


    def describe(self, image):
        # chuyển từ RGB (tương đổi khác nhau trên nhiều hệ thông) 
        # -> HSV (tăng độ chính xác cho bài toán)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []    
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
            (0, cX, cY, h)]

        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            
            # tính vùng trắng của ảnh
            cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)

            # sau đó dùng histogram để tính tần suất xuất hiện 
            # của giá trị cường độ mức xám của từng kênh màu
            hist = self.histogram(image, cornerMask)

            #nối 4 vùng ảnh vào với nhau
            features.extend(hist)

        # return the feature vector (1152)
        return features 