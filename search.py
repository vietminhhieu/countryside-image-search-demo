from searcher import Searcher
from colordescriptor import ColorDescriptor
import argparse
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required = True,
	help = "Path to where the computed index will be stored")
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
ap.add_argument("-r", "--result-path", required = True,
	help = "Path to the result path")
args = vars(ap.parse_args())
# initialize the image descriptor
cd = ColorDescriptor((8, 12, 3))

# đọc một bức ảnh cần truy vấn
query = cv2.imread(args["query"])
features = cd.describe(query)

# tìm kiếm ảnh giống nhất
searcher = Searcher(args["index"])
results = searcher.search(features)
# display the query
cv2.imshow("Query", query)
# loop over the results
for (score, resultID) in results:
	# load the result image and display it
    print("rsid: ",resultID," score: ",score)
    result = cv2.imread(resultID)
    try:
        cv2.imshow("Result", result)
        cv2.waitKey(0)
    except:
        pass

cv2.destroyAllWindows()