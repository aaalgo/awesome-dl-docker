import os, sys
import numpy as np
sys.path.append(os.path.join(os.getcwd(), "python/"))
import darknet
import cv2
import argparse

# Input argument: threshold, weights
parser = argparse.ArgumentParser()
parser.add_argument("--thresh", type = float, default = 0.005)
parser.add_argument("--weights", type = str, default = "500")
args = parser.parse_args()

if __name__ == "__main__":
    net = darknet.load_net("cfg/yolov3_object.cfg", "weights/yolov3_object_" + args.weights + ".weights", 0)
    meta = darknet.load_meta("cfg/my.data")

    # If you want to predict the validation sets
    # folder = "trainval/val/images/"
    folder = "results/test/"
    files = os.listdir(folder)

    # Output path:
    folder_pred = "results/predictions/"
    count = 0
    
    labels_list =["carplate"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    chart_colors = [(255, 20, 147)]     

    # Detection multiple images in files 
    for f in files:
        if f.endswith(".jpg"):
            print(f)
            image_cv2 = cv2.imread(os.path.join(folder, f), cv2.IMREAD_COLOR)
            image_path = bytes(os.path.join(folder, f).encode("utf-8"))
            res = darknet.detect(net, meta, image_path, thresh = args.thresh)
	    cnt = 0
	    if res != []:
		while cnt < len(res):
		    name = res[cnt][0]
		    if name in labels_list:
			i = labels_list.index(name)
		    predict = res[cnt][1]
		    x = res[cnt][2][0]
		    y = res[cnt][2][1]
		    w = res[cnt][2][2]
		    h = res[cnt][2][3]		
		    print(x,y,w,h)			    
		    x_max = int(round((2*x+w)/2))
		    x_min = int(round((2*x-w)/2))
		    y_max = int(round((2*y+h)/2))
		    y_min = int(round((2*y-h)/2))
		    cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), chart_colors[i], 2)
		    cv2.putText(image_cv2, name, (x_min, y_min-12), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
	            cnt += 1
	    
		count += 1
	        saving_path = f + "_weights_"+ args.weights + ".jpg"
	        cv2.imwrite(saving_path, image_cv2)        















		
	     
	                
