import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from time import sleep

scale = 50
unscaledPoints = np.array(pd.read_csv('Borders/Each Point Color.csv'))
points = unscaledPoints*scale
numberOfVisitors = int(max(unscaledPoints[:, 2]))

colors = np.random.randint(0, 255, size=(numberOfVisitors+1, 3))

bottomLeftCorner = np.min(np.array(points), axis=0)
bottomLeftCorner[2] = 1*scale # otherwise, bottomLeftCorner[2] would be -1*scale
topRightCorner = np.max(np.array(points), axis=0) - bottomLeftCorner
points = points - bottomLeftCorner

imageMargin = scale//2
imagePoints = np.zeros((int(topRightCorner[0])+2*imageMargin, int(topRightCorner[1])+2*imageMargin, 3), 'uint8')
for i, p in enumerate(points):
    color = int(unscaledPoints[i,2])
    if color == -1:
        imagePoints[-(int(p[0])+imageMargin), int(p[1])+imageMargin] = (255,255,255)
    else:
        imagePoints[-(int(p[0])+imageMargin), int(p[1])+imageMargin] = colors[color]

imagePoints = Image.fromarray(imagePoints)
draw = ImageDraw.Draw(imagePoints)
font = ImageFont.truetype('MONACO.TTF', size=16)

r1 = 17
r2 = 14
for visitorNumber in range(numberOfVisitors):
    thisVisitorPoints = points[points[:, 2] == scale*visitorNumber]
    sumX, sumY = 0, 0
    if len(thisVisitorPoints) != 0:
        for thisVisitorLoc in thisVisitorPoints:
            x = topRightCorner[0] - int(thisVisitorLoc[0])
            y = int(thisVisitorLoc[1])
            sumX += x+imageMargin
            sumY += y+imageMargin
        centerLoc = (sumY / len(thisVisitorPoints), sumX / len(thisVisitorPoints))
        textW, textH = font.getsize(f'{visitorNumber+1}')
        draw.ellipse(((centerLoc[0] - r1, centerLoc[1] - r1), (centerLoc[0] + r1, centerLoc[1] + r1)), fill=tuple(255-colors[visitorNumber+1]-50))
        draw.ellipse(((centerLoc[0] - r2, centerLoc[1] - r2), (centerLoc[0] + r2, centerLoc[1] + r2)), fill=tuple(255-colors[visitorNumber+1]))
        draw.text((centerLoc[0]-textW//2, centerLoc[1]-textH//2), f'{visitorNumber+1}', fill=tuple(colors[visitorNumber+1]), font=font)

imagePoints.show()
imagePoints.save('MapImage.png')
    

