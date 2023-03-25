from flax import linen as nn
import numpy as np
from typing import Any, Callable, Optional, Tuple, Type, List
from jax import lax, random, numpy as jnp
import einops
import torchio
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import jax
# import monai_swin_nD
import tensorflow as tf
# import monai_einops
import torch 
import einops
import torchio as tio
import optax
from flax.training import train_state  # Useful dataclass to keep train state
from torch.utils.data import DataLoader
import h5py
import SimpleITK as sitk

# f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/mytestfile.hdf5', 'w')

# pat_g=f.create_group(f"spleen/pat_0")
# pat_g.create_dataset("image", jnp.ones((10,10,10,1)))

# list(f["spleen"].keys())
# # f.create_group("spleen")
# pat_g=f.create_group("spleen/pat_1")
# f.create_dataset("spleen/pat_1/image",data= jnp.ones((10,10,10,1)))

# f.close()
# 

# slic = sitk.SLICImageFilter()
# seg  = slic.Execute(denoised_img)




spacing = (1.5,1.5,1.5)

def get_spleen_data():
    f = h5py.File('/workspaces/Jax_cuda_med/data/hdf5_loc/mytestfile.hdf5', 'r+')
    groups = list(f.keys())
    cached_subj=[]
    #if the dataset is not yet in hdf5 we will create it
    if("spleen" not in groups):
        print("spleen not in dataset")
        data_dir='/root/data'
        train_images = sorted(
            glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
        train_labels = sorted(
            glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))


        rang=list(range(0,len(train_images)))
        subjects_list=list(map(lambda index:tio.Subject(image=tio.ScalarImage(train_images[index],),label=tio.LabelMap(train_labels[index]),imagePath=train_images[index]),rang ))
        # subjects_list_train=subjects_list[:-9]
        # subjects_list_val=subjects_list[-9:]

        transforms = [
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.Resample(spacing),
            tio.transforms.CropOrPad((384,384,128)),
        ]
        transform = tio.Compose(transforms)
        subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

        cached_subj=[]
        training_loader = DataLoader(subjects_dataset, batch_size=1, num_workers=12)
        index =-1
        for subject in training_loader :
            index=index+1
            # pat_g=f.create_group(f"spleen/pat_{index}")
            # cached_subj.append(subject)
            image=subject['image'][tio.DATA].numpy()
            label=subject['label'][tio.DATA].numpy()
            # print(f"image shape {image.shape}")
            f.create_dataset(f"spleen/pat_{index}/image",data= image)
            f.create_dataset(f"spleen/pat_{index}/label",data= label)
            #### saving SLIC for some later pretraining
            slic = sitk.SLICImageFilter()
            # slic.SetMaximumNumberOfIterations(300)
            slic.SetEnforceConnectivity(True)
            image_sitk_nda=einops.rearrange(image,'bb cc a b c -> (bb cc c) b a')
            image_sitk = sitk.GetImageFromArray(image_sitk_nda)
            image_sitk.SetSpacing(spacing)

            rescalFilt=sitk.RescaleIntensityImageFilter()
            rescalFilt.SetOutputMaximum(1000)
            rescalFilt.SetOutputMinimum(0)
            image_sitk=rescalFilt.Execute(image_sitk)

            image_sitk=sitk.Cast(image_sitk, sitk.sitkInt64)
            slic_seg = slic.Execute(image_sitk)
            nda = sitk.GetArrayFromImage(slic_seg)
            nda=einops.rearrange(nda,'a b c -> 1 c b a')
            f.create_dataset(f"spleen/pat_{index}/slic",data= nda)
    #given we already have a dataset
    else:
        print("data loaded from hdf5")  

    pat_groups = list(f["spleen"].keys())
    spleenG= f["spleen"]
    # print(f"pat_groups {pat_groups}")
    grr=list(map( lambda groupp:spleenG[groupp] ,pat_groups))
    cached_subj=list(map( lambda groupp:(groupp["image"][:,:,:,:], groupp["label"][:,:,:,:],groupp["slic"][:,:,:,:]) ,grr))

    f.close()

    return cached_subj    
