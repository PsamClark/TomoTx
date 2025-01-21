#!/dls/ebic/data/staff-scratch/samclark/envs/orient/bin/python

import torch.utils.data as dta
import mrcfile as mrcf
import pandas as pd
import numpy as np
import skimage.transform as tm
import skimage.exposure as xpo
import torch 
import os
import pathlib 
from data import *

from torch.utils.data.distributed import DistributedSampler

class imgDataset(dta.Dataset):

    """ Dataset class
    """

    def __init__(self, fpath: str,
                dflen: int = None) -> None:
        super().__init__()

        #read in images from the mrcs files
        self._image_paths = np.array([i.absolute() for i in pathlib.Path(
            fpath).glob('**/*.mrc')]) 
        
        #set attributes
        self._params = {'df_length': dflen}

    
    def __len__(self):

        return 100

    
    def __getitem__(self, index):
    
            return self.prepare_set(index)      


    def prepare_set(self, index: int):

        """Prepare the data based on the use in cross transformer. 
        """
        
        img = mrcf.read(self._image_paths[index])
        
        img = img[None]

        #transform images
        img = self.transform_images(img)


        #mask image set
        img = mask_image(img)

        #generate sinograms
        img = sinogram_generator(img)

        #convert images to tensor    
        img = torch.from_numpy(img)

        return  img.to(torch.float32)

    def transform_images(self, images: np.ndarray):
    
        """ Generate circular mask for images setting everything outside 
        of the circles to zero.  
        """
        n=128
        #if set of images used 
        if images.ndim == 3:
    
            #create out image 
            images_out = np.zeros((images.shape[0],n,n))
    
            #iterate through and mask set
            for i in range(images_out.shape[0]):
                images_out[i] = self.transform_worker(images[i],n)                           
        else:
    
            #else just makes single image 
            images_out = self.transform_worker(images,n)
    
        return images_out

    def transform_worker(self, images: np.ndarray,n: int):

        """Perform image transformation 
        """

        
            
        images = (images-np.mean(images))/np.std(images)

        images_out = tm.resize(images, 
                                  output_shape =(n,n),
                                  anti_aliasing=False)

        #images_out = xpo.equalize_hist(images_out)
        images_out = images_out.astype(np.float32)

            
        return images_out


def prepare_training_data(fpath: str, 
                 tvsplit: np.array = np.array([0.6, 0.3, 0.1]),  
                          distrib = False):
    
    """Dataset preparation script. 
    """

    #access the dataset
    #dataset = imgDataset(fpath)
    dataset = SinogramDataset(fpath)
    
    #split dataset
    trn_set, tst_set, vldt_set = dta.random_split(dataset, tvsplit)
    
    if distrib:
        trn_smplr = DistributedSampler(trn_set)
        tst_smplr = DistributedSampler(tst_set, shuffle=False)
        vldt_smplr = DistributedSampler(vldt_set, shuffle=False)

        #load datasets
        trn_dl = dta.DataLoader(trn_set,sampler=trn_smplr,
                                batch_size=128)
        tst_dl = dta.DataLoader(tst_set, sampler=tst_smplr,
                                batch_size=16)
        vldt_dl = dta.DataLoader(vldt_set,sampler=vldt_smplr,
                                    batch_size=32)
     
    else:
        trn_dl = dta.DataLoader(trn_set, batch_size=32)
        tst_dl = dta.DataLoader(tst_set,
                                batch_size=16, shuffle=False)
        vldt_dl = dta.DataLoader(vldt_set, batch_size=32, shuffle=False)
        trn_smplr,tst_smplr,vldt_smplr=[],[],[]
    trn_pack = {'dl':trn_dl, 'smplr':trn_smplr}
    tst_pack = {'dl':tst_dl, 'smplr':tst_smplr}
    vldt_pack = {'dl':vldt_dl, 'smplr':vldt_smplr}    

    return trn_pack, tst_pack, vldt_pack


def mask_image(images: np.ndarray):

    """ Generate circular mask for images setting everything outside 
    of the circles to zero.  
    """

    #if set of images used 
    if images.ndim == 3:

        #create out image 
        images_out = np.zeros(images.shape)

        #iterate through and mask set
        for i in range(images_out.shape[0]):
            images_out[i] = mask_worker(images[i])                           
    else:

        #else just makes single image 
        images_out = mask_worker(images)

    return images_out


def mask_worker(images: np.ndarray):

    """ worker function for the mask
    """
    
    #get dimensions of the image
    nrows,ncols = images.shape

    #get the diameter of the circle (min dimension)
    diam = np.min([nrows,ncols])

    #generate rows and columns for mask 
    row,col = np.ogrid[:nrows,:ncols]

    #get centre of image
    cnt_row,cnt_col = nrows/2,ncols/2

    #generate outer disk mask 
    outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >
                   (diam / 2)**2)
    
    #copy image to mask out
    images_out = images.copy()
    
    #set all values outside of circular mask to zero
    images_out[outer_disk_mask]=0

    return images_out


def sinogram_generator(images: np.ndarray):

    """Generate sinogram of projections
    """
    #if set of images used 
    if images.ndim == 3:

        #create out image 
        images_out = np.zeros((images.shape[0], 360, np.min(images.shape[1:])))

        #iterate through and sinogram set
        for i in range(images_out.shape[0]):
            images_out[i] = sinogram_worker(images[i])                           
    else:

        #else just sinogram single image 
        images_out = sinogram_worker(images)

    return images_out


def sinogram_worker(images: np.ndarray):

    """Worker function for sinograms. 
    """
    
    #generate sinogram
    images_out = tm.radon(images, theta=np.arange(360))

    return images_out.T

