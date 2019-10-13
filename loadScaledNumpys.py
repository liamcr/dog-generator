import os
from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="float32" )
    return data

baseDir = ".\\6464picscolor"

trainDogsNumpy = [os.path.join(baseDir, fileDir) for fileDir in os.listdir(baseDir)]

trainArr = np.array([load_image(imgPath) for imgPath in trainDogsNumpy])
# print(trainArr)
# #trainArrMoreDims = np.expand_dims(trainArr, axis=3)

# print("done init loading train")

for pic in trainArr:
    for row in pic:
        for col in row:
            col[0] = (col[0] - 127.5) / 127.5
            col[1] = (col[1] - 127.5) / 127.5
            col[2] = (col[2] - 127.5) / 127.5

print(trainArr)
trainArr.dump("picsScaled.npy")
print("Done with training")