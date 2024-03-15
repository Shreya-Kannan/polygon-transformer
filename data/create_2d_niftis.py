import nibabel as nib
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom

dir_dest = '/home/shreya/scratch/Regional/2d_nifti_masks'
#dir_masks = '/home/shreya/scratch/Regional/trained_models/nnUNet/ensembles/Task501_Basal/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/ensembled_postprocessed'
#dir_masks = '/home/shreya/scratch/Regional/trained_models/nnUNet/ensembles/Task502_Mid/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/ensembled_postprocessed'
all_dirs = ['/home/shreya/scratch/Regional/nnUnet_Predictions/Task502_Mid/Predictions_ensemble','/home/shreya/scratch/Regional/nnUnet_Predictions/Task501_Basal/Predictions_ensemble']
for dir_masks in all_dirs:
    for root, dirs, files in os.walk(dir_masks):
        files.sort()
        for file in files:
            if file.endswith(".nii.gz"):
                msk_path = os.path.join(root, file) 
                mask = nib.load(msk_path).get_fdata()

                for i in range(mask.shape[2]):
                    mask_2d = mask[:,:,i]
                    new_name = file[:-7]+"_"+str(i+1)+".npz"
                    dst_path = os.path.join(dir_dest, new_name) 
                    #print(dst_path)
                    np.savez(dst_path, mask_2d=mask_2d)
                
            

