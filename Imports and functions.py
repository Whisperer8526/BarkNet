import numpy as np
import pandas as pd
import cv2
import sklearn
from PIL import Image

def count_files(dir):
    return len([1 for x in list(os.scandir(dir)) if x.is_file()])
  
 
