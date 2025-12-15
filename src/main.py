# main_fixed.py
import cv2
import argparse
from pipeline_fixed import LogoDetectionPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

pipeline = LogoDetectionPipeline()
img = cv2.imread(args.image)

out = pipeline.run(img)
cv2.imshow("Result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
