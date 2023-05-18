import numpy as np
import pandas as pd
import folium
import os
import json

# _______________________________________
def polygonCenter(polygon):
    x = 0
    y = 0
    for point in polygon:
        x += point[0]
        y += point[1]
    x /= len(polygon)
    y /= len(polygon)

    return (x, y)

# _______________________________________
def generateRandomLocations(latRange, lonRange, size):
    lats = np.random.rand(size)*(latRange[1] - latRange[0]) + latRange[0]
    lons = np.random.rand(size)*(lonRange[1] - lonRange[0]) + lonRange[0]
    locations = [(lats[i], lons[i]) for i in range(size)]
    return locations

# _______________________________________
def borderEquation(polygon):
    ms = []
    bs = []
    for pointNumber in range(len(polygon)-1):
        y1 = polygon[pointNumber][0]
        y2 = polygon[pointNumber+1][0]
        x1 = polygon[pointNumber][1]
        x2 = polygon[pointNumber+1][1]
        ms.append((y2 - y1) / (x2 - x1))
        bs.append(y1 - ms[pointNumber]*x1)
    return ms, bs

# _____________________________________________________________________________________________________________________

with open('Maps/Iran.geojson', 'r') as fHandler:
    polygon = json.load(fHandler)
polygon = polygon['features'][0]['geometry']['coordinates'][0]
polygon = [(point[1], point[0]) for point in polygon]

ms, bs = borderEquation(polygon)

polygonCenter = polygonCenter(polygon)
regionMap = folium.Map(polygonCenter, zoom_start=5)

folium.PolyLine(polygon, popup='<b>Selected Region (Iran)</b>', weight=9, color='green', opacity=0.6).add_to(regionMap)

bottomLeftCorner = np.min(np.array(polygon), axis=0)
topRightCorner = np.max(np.array(polygon), axis=0)
locations = generateRandomLocations(bottomLeftCorner, topRightCorner, 200000)

validLocations = []
for location in locations:
    collisionPoint = 0
    for lineNumber in range(len(bs)):
        X = (location[0] - bs[lineNumber])/ms[lineNumber]
        xmin = min([polygon[lineNumber][1], polygon[lineNumber+1][1]])
        xmax = max([polygon[lineNumber][1], polygon[lineNumber+1][1]])
        if X < xmax and X > xmin and location[1] < X:
            collisionPoint += 1

    if collisionPoint%2 == 1:   # Point is in the polygon, if number of collsions with the right side borders of point is odd.
        validLocations.append(location)
        # folium.Marker(location=location, icon=folium.Icon(color='red'), popup=location).add_to(regionMap)
    else:
        pass
        # folium.Marker(location=location, icon=folium.Icon(icon='cloud'), popup=location).add_to(regionMap)

    if len(validLocations) == 9151:
        break
validLocations = pd.DataFrame(validLocations)
validLocations.to_csv('locations.csv')
regionMap.save('Region Map.html')
os.system('google-chrome "Region Map.html"')



