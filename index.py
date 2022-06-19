from colordescriptor import ColorDescriptor
import argparse
import glob
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())
# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

# lưu các vector đặc trưng của từng bức ảnh ra 1 tệp
output = open(args["index"], "w")
for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# chỉ mục của hình ảnh
	imageID = imagePath[imagePath.rfind("/") + 1:]

	image = cv2.imread(imagePath)

	# tính vector đặc trưng của từng ảnh
	features = cd.describe(image)

	# ghi tên tệp vào vector đặc trưng
	features = [str(f) for f in features]

	output.write("%s,%s\n" % (imageID, ",".join(features)))

# close the index file
output.close()