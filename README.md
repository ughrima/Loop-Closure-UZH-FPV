
project
1-extracted poses from both 3 and 7 - 
Timestamp,PosX,PosY,PosZ,OriX,OriY,OriZ,OriW

2- converted them to pcds using caliberation information provided 

3- decided threshold value by generating trajecotry map and also histogram of minimum distance between them ie points bw bag 3 and 7

4 - Histogram gave us  - High frequency near 0–0.5m: This is clear evidence of spatial overlap → loop closures.

5 - hence took 0.5m as threshold for loop closure ground truth, computed ground truth got 5897 loop closures - Bag7_Timestamp,Bag7_PosX,Bag7_PosY,Bag7_PosZ,Bag3_Closest_Timestamp,Bag3_PosX,Bag3_PosY,Bag3_PosZ,Distance

6 - lpgw pipeline - bag 3 and bag 7 - to identify loop closure with minimum number of comparison basically optimize - loop closures prediction 
7 - changed the prediction.csv to match the ground truth file 
7 - comparison evaluation with ground truth 
results - 

Improved Evaluation Results:
True Positives (TP): 285
False Positives (FP): 698
False Negatives (FN): 47
True Negatives (TN): 1443
Precision: 0.2899
Recall: 0.8584
F1-Score: 0.4335
Accuracy: 0.6987

now need to do - 
identify threshold - why was 0.05m was taken for ground truth also the susequent threshold values
also the idea behind lpgw is to compare and bag 3 and 7, take only 1000 points or so ie less points and detect the loop closure, if that matches
then we can say that we optimised  loop closure detection, the idea is to give good results but with less points so that less computation, rn all points are being taken.





















6 - need to cluster these point-wise matches (ground truth) into temporal segments to align it with the way GW and similar algorithms work, so that we can compare them effectively and evaluate performance at the segment level.

7 - Segment 
Cluster the 5897 point-level ground truth loop closures into segment-level loop closures, where: Bag7 timestamps are temporally close (≤ 1s between consecutive points),
Their matched Bag3 timestamps are also temporally consistent (≤ 1s between),
Each segment contains at least 5 matches.


8 -  after segmentation got 62 high quality loop closure segments - Bag7_Start,Bag7_End,Bag3_Start,Bag3_End

9 -
Evaluation Results: {'precision': 0.42857142857142855, 'recall': 1.0, 'f1_score': 0.6, 'pr_auc': 0.20595238095238094, 'confusion_matrix': (2, 4, 0, 3)}