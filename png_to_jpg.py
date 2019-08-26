import glob
import numpy as np
import PIL.Image
import os

label_files = glob.glob(os.path.join("../../datasets/lines/val", '*.png'))
for file in label_files:
    print(file)
    img = np.asarray(PIL.Image.open(file))
    PIL.Image.fromarray(img).save(file.replace("png","jpg"))

