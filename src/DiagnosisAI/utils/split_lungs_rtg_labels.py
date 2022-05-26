pneomia_classes = ["virus", "bacteria"]
import os 
import shutil
from pathlib import Path

srcpath = Path("./datasets/chest_xray/test/PNEUMONIA/")
destpath = Path("./datasets/chest_xray/test/")


for root, subFolders, files in os.walk(srcpath):
    for file in files:
        if pneomia_classes[0] in file:
            subFolder = os.path.join(destpath, 'PNEUMONIA_VIRUS')
            if not os.path.isdir(subFolder):
                os.makedirs(subFolder)
            shutil.move(os.path.join(root, file), subFolder)
        elif pneomia_classes[1] in file:
            subFolder = os.path.join(destpath, 'PNEUMONIA_BACTERIA')
            if not os.path.isdir(subFolder):
                os.makedirs(subFolder)
            shutil.move(os.path.join(root, file), subFolder)

