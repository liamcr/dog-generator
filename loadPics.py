import os
from PIL import Image

inputDir = input("Input Directory? ")
outputDir = input("Output Directory? ")
imgHeight = int(input("Image height? "))
imgWidth = int(input("Image width? "))
rgb = True if input("RGB (y/n)? ") == "y" else False

def load_image(infilename):
    img = Image.open( infilename )
    if (not rgb):
        img = img.convert("L")
    img = img.resize((imgWidth,imgHeight),Image.ANTIALIAS)
    img.load()
    return img

baseDir = "."
inputDirPath = os.path.join(baseDir, inputDir)
outputDirPath = os.path.join(baseDir, outputDir)

if (not os.path.exists(outputDirPath)):
    os.mkdir(outputDirPath)

listOfImgs = os.listdir(inputDirPath)

[load_image(os.path.join(inputDirPath, image)).save(os.path.join(outputDirPath, image)) for image in listOfImgs]
