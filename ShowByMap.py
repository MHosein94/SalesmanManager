import numpy as np
import pandas as pd
import folium
from folium.plugins import MarkerCluster, MiniMap, Draw, BeautifyIcon, MousePosition, Geocoder
from os import system
import json

# _______________________________________
def findCenterPoint(allLocations):
    lat = 0
    lon = 0
    for point in allLocations:
        lat += point[0]
        lon += point[1]
    lat /= len(allLocations)
    lon /= len(allLocations)

    return (lat, lon)

# _______________________________________
def showOnMap(polygonPath, polygonName=None):
    with open(polygonPath, 'r') as fHandler:
        polygon = json.load(fHandler)
    polygon = polygon['features'][0]['geometry']['coordinates'][0]
    polygon = [(point[1], point[0]) for point in polygon]

    polygonCenter = findCenterPoint(polygon)
    bottomLeftCorner = np.min(np.array(polygon), axis=0)
    topRightCorner = np.max(np.array(polygon), axis=0)
    diameterCoord = topRightCorner - bottomLeftCorner
    diameter = np.sqrt(diameterCoord[0]**2 + diameterCoord[1]**2)
    zoom = round(-diameter*(6/23) + 6/23 + 11)
    
    Tiles = ['CartoDB positron', 'Stamen Terrain', 'Stamen Toner', 
    'OpenStreetMap', 'Stamen Watercolor', 'CartoDB Dark_matter']

    regionMap = folium.Map(polygonCenter, zoom_start=zoom, tiles=Tiles[3], control_scale=True)

    markedCustomers = pd.read_csv('Marked Data/Marked Customers.csv', header=None)
    customersInfo = pd.read_csv('Tables/VisitorsInfo.csv')

    Y = markedCustomers.iloc[:,0]
    X = markedCustomers.iloc[:,1]
    customerNumbers = markedCustomers.iloc[:,3]
    visitorNumbers = np.array(markedCustomers.iloc[:,2], dtype='int')
    numberOfVisitors = max(visitorNumbers)
    locations = [(Y[i], X[i]) for i in range(len(X))]

    nameMoshtary = customersInfo.NameMoshtary
    customerID = customersInfo.ID

    colorsArray = np.random.randint(0, 0xFFFFFF, size=numberOfVisitors)
    allColors = []
    allColorsComplementary = []
    for colorArray in colorsArray:
        allColors.append("#%06x" % colorArray)
        allColorsComplementary.append("#%06x" % (0xFFFFFF - colorArray))
        
    featureGroups = []
    markerCluster = []
    for fgCounter in range(1, numberOfVisitors+1):
        featureGroups.append(folium.FeatureGroup(name=f'Visitor Number: {fgCounter}'))
        markerCluster.append(MarkerCluster().add_to(featureGroups[fgCounter-1]))
    
    # Border
    folium.GeoJson(
        polygonPath, tooltip=polygonName,
        style_function=lambda fillColor='#000000', color='#00FFFF':{
                "fillColor": fillColor,
                "color": color,
            }, 
        ).add_to(regionMap)  

    # Customers
    for i, location in enumerate(locations):
        folium.Marker(location=location, 
        icon=BeautifyIcon(
            icon_shape='marker',
            border_color=allColorsComplementary[visitorNumbers[i]-1],
            text_color=allColorsComplementary[visitorNumbers[i]-1],
            icon_size=(36, 36), 
            background_color=allColors[visitorNumbers[i]-1],
            number=f'{int(customerNumbers[i])}'
            ), 
        popup=np.array(nameMoshtary[customerID == customerNumbers[i]])[0], tooltip=visitorNumbers[i]
        ).add_to(markerCluster[visitorNumbers[i]-1])

    # Coloring Map
    coloredPoints = np.array(pd.read_csv('Marked Data/Each Point Color.csv', header=None))
    precision = round(abs(coloredPoints[1,1] - coloredPoints[0,1]), 3)
    rRect = precision / 2

    for i, location in enumerate(coloredPoints):
        visitorNumber = int(location[2])-1
        folium.Rectangle(bounds=[(location[0]-rRect, location[1]-rRect), (location[0]+rRect, location[1]+rRect)], 
        fill=True, color=allColors[visitorNumber], fill_color=allColors[visitorNumber], fill_opacity=0.3,
        popup=i, tooltip=visitorNumber+1
        ).add_to(featureGroups[visitorNumber])

    # Finalizing
    for fg in featureGroups:
        fg.add_to(regionMap)

    minimap = MiniMap(toggle_display=True)
    regionMap.add_child(minimap)
    draw = Draw(export=True)
    draw.add_to(regionMap)
    MousePosition().add_to(regionMap)
    folium.LayerControl().add_to(regionMap)
    Geocoder(position='bottomright').add_to(regionMap)

    regionMap.save('Maps/Regioned Map.html')
    system('google-chrome "Maps/Regioned Map.html"')

if __name__ == '__main__':
    polygonPath = 'Maps/Tehran.geojson'
    polygonName = polygonPath.replace('Maps/', '').replace('.geojson', '')
    showOnMap(polygonPath=polygonPath, polygonName=polygonName)