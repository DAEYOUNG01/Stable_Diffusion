import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2
from glob import glob
from collections import Counter
import numpy as np
from PIL import Image  # PIL 추가
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
import random
import os

# GPU 사용 여부 설정
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using Pytorch version : ', torch.__version__, ' Device : ', DEVICE)
