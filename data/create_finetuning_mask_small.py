import numpy as np
from PIL import Image
import random
import os
from tqdm import tqdm
import os
import json
import pydicom as dicom
from skimage.measure import label, regionprops, regionprops_table
import math

import pickle
from poly_utils import is_clockwise, close_polygon_contour, revert_direction, check_length, reorder_points, \
    approximate_polygons, interpolate_polygons, image_to_base64, polygons_to_string

combined_train_data = []
combined_test_data = []
combined_val_data = []

max_length = 400

sentence_dict = {
                '1':'anteroseptal',
                '2':'inferoseptal',
                '3':'inferior',
                '4':'inferolateral',
                '5':'anterolateral',
                '6':'anterior'
                }


def sort_counterclockwise(points, centre):
  if centre:
    centre_x, centre_y = centre
  else:
    raise ValueError("Center cannot be None. Please specify center point.")
  angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
  counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
  counterclockwise_points = [points[i] for i in counterclockwise_indices]
  return counterclockwise_points

def get_contours(path_json,path_masks_pred,path_lbl):
    """
    args:
    path_json: path to contour json file
    path_masks_pred: path to masks_pred file
    """
    contour = []
    img = np.load(path_masks_pred)['mask_2d']
    label_ = np.load(path_lbl)["label"]
    # read json file as dictionary
    with open(path_json) as f:
        contour = json.load(f)
    
    #center point
    c_x,c_y = contour['center'][0], img.shape[0] - np.array(contour['center'][1])
    
    contours_x = []
    contours_y = []

    
    for i in range(6):
      try:
          #contours_x = contour[str(i)][0]
          #contours_y = img.shape[0] - np.array(contour[str(i)][1])

          #first point for line1
          p1_x, p1_y = contour[str(i)][0][0], img.shape[0] - np.array(contour[str(i)][1])[0]
          

          #last point for line2
          p2_x, p2_y = contour[str(i)][0][-1], img.shape[0] - np.array(contour[str(i)][1])[-1]
          
          contours_x.extend([p1_x,p2_x])
          contours_y.extend([p1_y, p2_y])
          
      except KeyError as e: # if the number of regions is four, should not come here
          print(e)
    
    coordinates_contours =[[contours_x[i],contours_y[i]] for i in range(len(contours_x))]
    coordinates_contours = sort_counterclockwise(coordinates_contours,[c_x,c_y])
    contour = [np.array(coordinates_contours).reshape(-1,).tolist()]
    
    return contour,[c_x,c_y]

def get_bbox(path_lbl):
    label_ = np.load(path_lbl)["label"]
    lbl = label_.copy()
    lbl[lbl!=0] = 1
    label_img = label(lbl)
    regions = regionprops(label_img)
    for props in regions:     
        minr, minc, maxr, maxc = props.bbox
    return [minr, minc, maxr, maxc]

# set up image paths
dir_lbl='/home/shreya/scratch/Regional/regional_label/'
dir_contours = '/home/shreya/scratch/Regional/MRI_contours/MRI/'
dir_masks_pred = '/home/shreya/scratch/Regional/2d_nifti_masks'
    
# list subdirectories in dir_lbl
subdirs = os.listdir(dir_lbl)
subdirs.sort()

#Train-test split
train_split = subdirs[:43]
val_split =  subdirs[43:54]
test_split = subdirs[54:]

print(train_split)
print(val_split)
print(test_split)

id_ = 0

for subdir in subdirs:
    print(subdir)
    # Get the list of series in subdir
    dir_series = os.listdir(os.path.join(dir_lbl, subdir))
    
    # iterate the series name
    for series in dir_series:
        
        # get all label files in the path and sort them
        label_files = os.listdir(os.path.join(dir_lbl, subdir, series))
        #print(json_files)
        
        for file_label in list(label_files):
        
            fname = file_label.split("-")
            #print(fname)
            slice_number = int(fname.pop()[:-4][-2:])
            #print(slice_number)
            fname.extend(["_",str(slice_number),'.npz'])
            filename = subdir.replace("_","") + "".join(fname)
            path_masks_pred = os.path.join(dir_masks_pred, filename)
            path_json = os.path.join(dir_contours, subdir, series, file_label.replace('.npz', '.json'))
            path_label = os.path.join(dir_lbl, subdir, series, file_label)

            lbl = np.load(path_label)["label"]

            #ignore apical files
            if len(np.unique(lbl))<6:
                continue
            contour, center = get_contours(path_json,path_masks_pred,path_label)
            bbox_list = get_bbox(path_label)

            polygons = contour
            polygons_processed = []
            for polygon in polygons:
                # make the polygon clockwise
                if not is_clockwise(polygon):
                    polygon = revert_direction(polygon)

                # reorder the polygon so that the first vertex is the one closest to image origin
                polygon = reorder_points(polygon)
                #polygon = close_polygon_contour(polygon)
                polygons_processed.append(polygon)

            polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
            #polygons_interpolated = interpolate_polygons(polygons)

            #polygons = approximate_polygons(polygons, 5, max_length)

            pts_string = polygons_to_string(polygons)
            #pts_string_interpolated = polygons_to_string(polygons_interpolated)

            x1,y1,x2,y2 = bbox_list
            box_string = f'{x1},{y1},{x2},{y2}'

            uniq_id = f"{str(id_)}" 
            print(uniq_id)
            id_+=1

            instance = '\t'.join(
                [uniq_id, subdir, 'Generate the correct Regional points', box_string, pts_string, path_masks_pred, path_label,
                    pts_string]) + '\n'
            
            if subdir in train_split:
                combined_train_data.append(instance)
            elif subdir in val_split:
                combined_val_data.append(instance)
            else:
                combined_test_data.append(instance)

random.shuffle(combined_train_data)
file_name = os.path.join("/home/shreya/scratch/Regional/polygon-transformer/datasets/finetune/Regional_mask_small_train.tsv")
print("creating ", file_name)
writer = open(file_name, 'w')
writer.writelines(combined_train_data)
writer.close()

file_name_test = os.path.join("/home/shreya/scratch/Regional/polygon-transformer/datasets/finetune/Regional_mask_small_test.tsv")
print("creating ", file_name_test)
writer = open(file_name_test, 'w')
writer.writelines(combined_test_data)
writer.close()

file_name_val = os.path.join("/home/shreya/scratch/Regional/polygon-transformer/datasets/finetune/Regional_mask_small_val.tsv")
print("creating ", file_name_val)
writer = open(file_name_val, 'w')
writer.writelines(combined_val_data)
writer.close()

