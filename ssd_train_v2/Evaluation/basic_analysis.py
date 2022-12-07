import pandas as pd
from shapely.geometry import Polygon
#shapely input with all x y coordinates
#Note: The origin of Coordinate Systems in shapely library is left-bottom where origin in computer graphics is left-top. This #difference does not affect the IoU calculation, but if you do other yes

#PLease provide path for ground truth and predicted annotations here 

df_annotation_ground_truth= pd.read_csv("./test_data.csv")
# Uncomment this line to test accuracy with PB model
#df_annotations_predicted= pd.read_csv("./output.csv")
# Uncomment this line to test accuracy with Tflite Float Model NNtool converted
#df_annotations_predicted= pd.read_csv("./output_tflite.csv")
# Uncomment this line to test accuracy with Tflite Quantized Model NNtool converted
df_annotations_predicted= pd.read_csv("./output_tflite_quant.csv")


#this is How better we want our predicted cooresponding to ground truth, I chose this as 0.0 as task was people detection adn #we are only providing the output with threshold higher than 60 percent.
given_IOU= 0.0
SCORE_THR = 0.75
image_names_ground_truth= df_annotation_ground_truth['image_name'].unique()
image_names_predicted= df_annotations_predicted['image_name'].unique()
image_name= list(set(image_names_ground_truth) | set(image_names_predicted))

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    if poly_1.union(poly_2).area > 0:
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou
    else: return 0.0


def evaluaition_images(ground_truth,predicted):
    '''
    both ground_truth,predicted are sampled data frames for same image 
    first we create alist with all four coordiantes of bounding boxes 
    then recurssive IOU function is run on each ground truth to find best IOU with predictted values 
    the result of this is the list with index wise relation between ground_index, predicted_index, IOU
    #incase we have more predicted or ground truth that are associated with None eg;
    None, predicted_index, None or ground_index, None, None 
    
    At end we will return 3 lists 
    1; ground_truth_coordinate_list basically ground truth info in list format  with all four coordinates 
    

    '''
    # I am taking them in a list format 
    ground_truth_coordinate_list=[]
    predicted_coordinate_list=[]
    
    
    for i in range(len(ground_truth)):
        x1= ground_truth.iloc[i]['xmin']
        y1= ground_truth.iloc[i]['ymin']
        x2= ground_truth.iloc[i]['xmin']
        y2= ground_truth.iloc[i]['ymax']
        x3= ground_truth.iloc[i]['xmax']
        y3= ground_truth.iloc[i]['ymax']
        x4= ground_truth.iloc[i]['xmax']
        y4= ground_truth.iloc[i]['ymin']
            #4 coordinates 
        ground_truth_coordinate_list.append([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    
    for j in range(len(predicted)):

            
        X1= predicted.iloc[j]['xmin']
        Y1= predicted.iloc[j]['ymin']
        X2= predicted.iloc[j]['xmin']
        Y2= predicted.iloc[j]['ymax']
        X3= predicted.iloc[j]['xmax']
        Y3= predicted.iloc[j]['ymax']
        X4= predicted.iloc[j]['xmax']
        Y4= predicted.iloc[j]['ymin']
        if predicted.iloc[j]['score'] > SCORE_THR:
            predicted_coordinate_list.append([[X1,Y1],[X2,Y2],[X3,Y3],[X4,Y4]])

    
   

        
            
            
    def recurrsive_IOU_selection(gt, pd, final):
        #ground truth refers to gt the groiund truth list containing bounding box values in a image 
        #pd predicted input bounding value we are finding IOU with
        #final_related is the final list we will poroduce after IOU has crossed threshold 
        
        '''
        
        The reason we are doing this recurssive is we don't know which ground truth is assosiated to which predicted value as are output
        are just bounding boxes details hence what we do is coompare the best IOU between predictyed adn ground truth and assosicte those 
        ground truth with thoes predicted box values. This will be a recurssive process.
        
        Few heutrics could be applied to lessen the calculation like distance between center of predicted and ground truth if it is greater than 
        sum of diaganol of both then thye don't intersect but I haven't done that. You could try in future versions
        
        '''
        
        
        final_gt_pd= final.copy()
        
        for i in range(len(gt)):
            temp_iou= 0.0
            index_g= None
            index_p= None
            #taking the single value in ground truth list start by First
            box_1= gt[i]
            
            for j in range(len(pd)):
                box_2= pd[j] #excluding confidence_score as input 
                iou_calc= calculate_iou(box_1, box_2)
                
                #now I am am going to compare it with all the values in the predicted and take best
                if iou_calc>temp_iou:
                    
                    # now check if this predicted is associated with any previous associated ground truth value
                    for o in range(len(final_gt_pd)):
                        if (final_gt_pd[o][1] == j): 
                            if (final_gt_pd[o][2] < iou_calc):
                            #if yes is IOU grater than calculated
                            #if yes take it out 
                                final_gt_pd.pop(o)

                                temp_iou= iou_calc
                                index_p= j
                                index_g= i
                                
                            break
                        
                                
                    else:
                        
                       
                        temp_iou= iou_calc
                        index_p= j
                        index_g= i
                    
                
            if (index_p == None)| (index_g== None): continue
            else: final_gt_pd.append([index_g,index_p,temp_iou])
                
                
        for m in final_gt_pd:
            if m not in final:
                final_gt_pd=recurrsive_IOU_selection(gt, pd, final_gt_pd)
        for k in final:
            if k not in final_gt_pd: 
                final_gt_pd=recurrsive_IOU_selection(gt, pd, final_gt_pd)
        else: return final_gt_pd
           
    grnd_pretd_rel=[]
    
    grnd_pretd_rel_list= recurrsive_IOU_selection(ground_truth_coordinate_list, predicted_coordinate_list, grnd_pretd_rel)
    
    
    #those indexes not assosciated with anyone are associated with None 
    predicted_associated= list()
    ground_associated =list()
    for m in grnd_pretd_rel_list: predicted_associated.append(m[1])
    for n in grnd_pretd_rel_list: ground_associated.append(n[0])
    for k in range(len(ground_truth_coordinate_list)):
        if k not in ground_associated:
            grnd_pretd_rel_list.append([k, None, None])
    for f in range(len(predicted_coordinate_list)):
        if f not in predicted_associated:
            grnd_pretd_rel_list.append([None,f, None])
            
            

    return ground_truth_coordinate_list, predicted_coordinate_list, grnd_pretd_rel_list
def confusion_analysis(ground_truth_coordinate_list, predicted_coordinate_list, relationship_list, given_IOU):
    TP=list()
    FP=list()
    #TN=list() there are no true negatives we have only one class
    FN=list()
    for val in relationship_list:
        #val[2] refers to IOU calculated 
        #if it is None that means it is either FP or FN 
        if val[2]== None:
            if val[0]==None:FP.append(val)
            elif val[1]==None: FN.append(val)
        else:
            if val[2]>= given_IOU:
                TP.append(val)
            else: #if IOU calculated is less
                FN.append(val) # they will be seen as False negatives
    return TP, FP, FN

TP_count, FP_count, FN_count=0, 0, 0
for i in image_name:
    gt= df_annotation_ground_truth[df_annotation_ground_truth['image_name']==i]
    pd= df_annotations_predicted[df_annotations_predicted['image_name']==i]
    ground_truth_coordinate_list, predicted_coordinate_list, relationship_list= evaluaition_images(gt,pd)
    #print(relationship_list,gt,pd)
    TP, FP, FN= confusion_analysis(ground_truth_coordinate_list, predicted_coordinate_list, relationship_list, given_IOU)
    TP_count+= len(TP)
    FP_count+= len(FP)
    FN_count+= len(FN)


print('True Positive count is {}'.format(TP_count))
print('False Positive count  is {}'.format(FP_count))
print('False Negative count  is {}'.format(FN_count))
print('Precision is {}'.format(TP_count/(TP_count + FP_count)))
print('Recall  is {}'.format(TP_count/(TP_count + FN_count)))
print('F1 score  is {}'.format(TP_count/(TP_count +((FN_count + FP_count)/2))))
print('Net Accuracy is {}'.format(TP_count/(TP_count + FP_count + FN_count)))    
