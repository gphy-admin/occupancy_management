import os
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image



# Create figure and axes
#fig, ax = plt.subplots()
plt.ion()

#PLease provide path for annotations here 
df_annotations= pd.read_csv("ssd_data/test_data.csv")
image_path="ssd_data/images/"

#this is How better we want our predicted cooresponding to ground truth, I chose this as 0.0 as task was people detection adn #we are only providing the output with threshold higher than 60 percent.
image_names= df_annotations['image_name'].unique()

window_name = 'Image'

for i in image_names:
    gt= df_annotations[df_annotations['image_name']==i]
    #print(gt.iat[0,0])
    print(len(gt['image_name']))
    if not os.path.isfile(image_path+gt.iat[0,0]):
    	continue
    im = Image.open(image_path+gt.iat[0,0])

    fig, ax = plt.subplots()
    for i in range(len(gt['image_name'])):
    	rect = patches.Rectangle((gt.iat[i,1], gt.iat[i,3]), gt.iat[i,2]-gt.iat[i,1], gt.iat[i,4]-gt.iat[i,3], linewidth=1, edgecolor='r', facecolor='none')
    	ax.add_patch(rect)

    	#print(gt.iat[i,0], end =" ") #image name
    	#print(gt.iat[i,1], end =" ") #xmin
    	#print(gt.iat[i,2], end =" ") #xman
    	#print(gt.iat[i,3], end =" ") #ymin
    	#print(gt.iat[i,4])           #ymax

    ax.imshow(im)
    plt.show()
    _ = input("Press [enter] to continue.") # wait for input from the user
    plt.close()
