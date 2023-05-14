import os
import shutil
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def trainTestSplit(genderClassification: pd.DataFrame,
                   datasetDir: str,
                   test_size: float = .2
                   ) -> None:
  '''
  params
  ---------
  genderClassification: pd.DataFrame
      file contains gender classification, to classifiy the images is male or female
  test_size: float
      proportion of our test set
  datasetDir: str
      directory where all the data's located

  Returns
  ---------
  None
  Notes: it might be take sometimes, 'cause the code try to copy paste the images

  Example
  ---------
  MAIN = 'Datasets/'
  mapping_gender = pd.read_csv(MAIN + 'gender_classification.csv')
  imgs_ = os.listdir(MAIN + 'Images/')

  trainTestSplit(mapping_gender, images_ = imgs_, datasetDir= MAIN)
  '''

  MAIN = datasetDir

  X_train, X_test, y_train, y_test = train_test_split(genderClassification['Unnamed: 0'],
                                                      genderClassification['Male'],
                                                      test_size = test_size)

  train_key = dict(zip(X_train, y_train))
  test_key = dict(zip(X_test, y_test))

  for item in ["Train", "Test"]:
    if item == "Train":
      item_ = train_key
      os.makedirs(MAIN + "Processed_Images/Train/", exist_ok = True)
      os.makedirs(MAIN + "Processed_Images/Train/Male/", exist_ok = True)
      os.makedirs(MAIN + "Processed_Images/Train/Female/", exist_ok = True)
      Dir = MAIN + "Processed_Images/Train/"
    elif item == "Test":
      item_ = test_key
      os.makedirs(MAIN + "Processed_Images/Test/", exist_ok = True)
      os.makedirs(MAIN + "Processed_Images/Test/Male/", exist_ok = True)
      os.makedirs(MAIN + "Processed_Images/Test/Female/", exist_ok = True)
      Dir = MAIN + "Processed_Images/Test/"
    
    for key_, val_ in tqdm(item_.items(), desc = item):
        if val_ == 1:
          shutil.copy(MAIN + f'Images/{key_}', Dir + "Male/")
        else:
          shutil.copy(MAIN + f'Images/{key_}', Dir + "Female/")
    

  print("SPLITTING DONE!")


def dataLoader(trainDir: str,
               testDir: str,
               transformer: list,
               batch_size: int,
               shuffle: list = [True, False],
               num_workers: int = 2) :
  '''
  params
  ---------
  trainDir: str
      directory of your train set images
  testDir: str
      directory of your test set images
  transformer: list -> [transformerTrain, transformerTest]
      transformer we use to transform our train and test images
  batch_size: int
      # of batch
  crop_size: int
      -
  shuffle: bool
      -
  num_workers: int
      -

  Return
  ----------
  train_set, trainLoader, test_set, testLoader

  Example
  -----------
  BATCH = 128
  CROP_SIZE = 64

  train_transform = transforms.Compose([
      transforms.RandomRotation(15),
      transforms.RandomResizedCrop(CROP_SIZE, scale= (.8, 1)), ## zoom maximal 80% dari gambar semula
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
  ])

  test_transform = transforms.Compose([
      transforms.Resize(70),
      transforms.CenterCrop(CROP_SIZE),
      transforms.ToTensor()
  ])


  '''

  train_set = datasets.ImageFolder(trainDir, transform = transformer[0]) 
  trainLoader = DataLoader(train_set, batch_size = batch_size, shuffle = shuffle[0], num_workers = num_workers)

  test_set = datasets.ImageFolder(testDir, transform = transformer[1])
  testLoader = DataLoader(test_set, batch_size = batch_size, shuffle = shuffle[1], num_workers = num_workers)

  return train_set, trainLoader, test_set, testLoader