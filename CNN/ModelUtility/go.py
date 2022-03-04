from PIL import Image
import cv2 as cv
import os


def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    output = (PngPath[:-3] + "jpg")
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            # os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            # os.remove(PngPath)
        return output
    except Exception as e:
        print("PNG转换JPG 错误", e)

