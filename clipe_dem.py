import numpy as np
import arcpy
import os
import math


arcpy.Clip_management(
    "E:/srtm_37_03/srtm_37_03.tif","4.276 48.031 4.285 48.037",
    "E:/srtm_37_03/srtm_37_03_mini.tif", "#", "#", "NONE", "NO_MAINTAIN_EXTENT")