from PIL import Image #reading images
import os #iterating over folders
import numpy as np #performing calculations
from sklearn.metrics import confusion_matrix #for getting tn, tp, fp, fn
import matplotlib.pyplot as plt

#ground truth labels, 1 for day and 0 for night
ground_truth_labels = []
path_to_images = []
for folder in os.listdir("day-night-images/training"):
    if folder=='day':
        for image in os.listdir(os.path.join("day-night-images/training",folder)):
            path_to_images.append(os.path.join("day-night-images/training",folder,image))
            ground_truth_labels.append(1)
    else:
        for image in os.listdir(os.path.join("day-night-images/training",folder)):
            path_to_images.append(os.path.join("day-night-images/training",folder,image))
            ground_truth_labels.append(0)


'''takes a list containing path to images and performs the following actions
1. Reads images
2. Resizes image to 800x800
3. Converts image to HSV channel style
4. Sums up the values in pixels of V channel, and takes mean by diving by image area which is 640000
5.
'''
def preprocess_image(image_path_list):
    v_values_images = []
    for image in image_path_list:
        img = Image.open(image)
        img = img.resize((800,800))
        img = img.convert('HSV')
        img = np.asarray(img)
        v = np.sum(img[:, :, 2])
        v = v/640000
        v_values_images.append(v)
    return v_values_images

v_value_for_images = preprocess_image(path_to_images)

'''geting threshold for classification into day and night
I use sklearn library to get the tp, fp, tn, and fn for a given threshold
This part is used to just get a suitable threshold and will not be used for testing'''

max_v = max(v_value_for_images)
min_v = min(v_value_for_images)
step = int((max_v-min_v)/100)
print("Calculating TP, FP, TN, FN at steps of: ",step)

tn_all = []
fp_all = []
fn_all = []
tp_all = []
threshold_all = []

for i in range(int(min_v),int(max_v+2),step):
    #getting predictions
    predicted_values = []
    threshold_all.append(i)
    for j in range(0, len(ground_truth_labels)):
        if v_value_for_images[j]<=i:
            predicted_values.append(0)
        else:
            predicted_values.append(1)    
    #calculating metrics using sklearn
    tn, fp, fn, tp = confusion_matrix(ground_truth_labels, predicted_values).ravel()
    tn_all.append(tn)
    fp_all.append(fp)
    fn_all.append(fn)
    tp_all.append(tp)

plt.plot(threshold_all, tn_all, label='true negative')
plt.plot(threshold_all, fp_all, label='false positive', linestyle='dotted')
plt.plot(threshold_all, fn_all, label='false negative', linestyle='dotted')
plt.plot(threshold_all, tp_all, label='true positive')
idx=np.argwhere(np.diff(np.sign(np.array(tp_all) - np.array(tn_all))) != 0).reshape(-1) + 0
plt.plot(threshold_all[idx[0]], tp_all[idx[0]], 'ro')
print("Optimal threshold is:",threshold_all[idx[0]])
plt.legend()
plt.xlabel("Threshold values")
plt.ylabel("Count of values")
plt.savefig("plot.jpg")