import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from ShowOnMap import showOnMap
import json

# Latitiude = Y, Longtitude = X. Coordinates: (Lat, Lon)
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

# _________________________________________________________
def findDistanceOnMap(firstLoc, secondLoc):
    distanceLat = np.math.radians(secondLoc[0]) - np.math.radians(firstLoc[0])
    distanceLon = np.math.radians(secondLoc[1]) - np.math.radians(firstLoc[1])

    a = np.math.sin(distanceLat / 2)**2 + np.math.cos(np.math.radians(firstLoc[0])) * np.math.cos(np.math.radians(secondLoc[0])) * np.math.sin(distanceLon / 2)**2
    c = 2 * np.math.atan2(np.math.sqrt(a), np.math.sqrt(1 - a))
    R = 6371.0088 # Approximate radius of earth in km
    distance = R * c
    return distance

# _________________________________________________________
def calculateTime(firstLoc, secondLoc, vehicle, distanceType=None):
    if vehicle == None:
        vehicle = 'Car'
    # Speeds are in km/h, and return time in hour
    speeds = {'Bike': 14, 'Car': 60, 'Walk': 1}
    speed = speeds[vehicle]

    if distanceType == 'Manhattan':
        distance = abs(firstLoc[0] - secondLoc[0]) + abs(firstLoc[1] - secondLoc[1])
    elif distanceType == 'Euclidan':
        distance = np.sqrt((firstLoc[0] - secondLoc[0])**2 + (firstLoc[1] - secondLoc[1])**2)
    elif distanceType == 'Mercator' or distanceType == None:
        distance = findDistanceOnMap(firstLoc, secondLoc)
    elif distanceType == 'Chebyshev':
        distance = max([abs(firstLoc[0] - secondLoc[0]), abs(firstLoc[1] - secondLoc[1])])

    t = distance/speed
    return t

# _________________________________________________________
def findNearestPoint(location, allLocations, vehicle=None, distanceType=None):
    minT = float('inf')
    nearestPoint = 0
    for pointNumber, point in enumerate(allLocations):
        t = calculateTime(location, point, vehicle, distanceType)
        if t < minT:
            nearestPoint = pointNumber
            minT = t

    if minT == float('inf'):
        minT = 0

    return (minT, nearestPoint)

# _________________________________________________________
def findFurthestPoint(allLocations, vehicle=None, origin=None):
    if origin == None:
        origin = findCenterPoint(allLocations)
    
    maxT = float('-inf')
    furthestPoint = 0
    for pointNumber, point in enumerate(allLocations):
        t = calculateTime(origin, point, vehicle, distanceType='Euclidan')
        if t > maxT:
            furthestPoint = pointNumber
            maxT = t

    return furthestPoint

# _________________________________________________________
def findNearestLabel(point, markedCustomers, distanceType):
    minDistance = float('inf')

    for customerInfo in markedCustomers:
        customerLoc = (customerInfo[0], customerInfo[1])
        customerLabel = customerInfo[2]

        if distanceType == 'Manhattan':
            distance = abs(point[0] - customerLoc[0]) + abs(point[1] - customerLoc[1])
        elif distanceType == 'Euclidan':
            distance = np.sqrt((point[0] - customerLoc[0])**2 + (point[1] - customerLoc[1])**2)
        elif distanceType == 'Mercator':
            distance = findDistanceOnMap(point, customerLoc)
        elif distanceType == 'Chebyshev':
            distance = max([abs(point[0] - customerLoc[0]), abs(point[1] - customerLoc[1])])
        
        if distance < minDistance:
            minDistance = distance
            nearestLabel = customerLabel

    return nearestLabel

# _______________________________________
def findOrigin(allLocations):
    bottomLeftCorner = np.min(np.array(allLocations), axis=0)
    topRightCorner = np.max(np.array(allLocations), axis=0)
    origin = (0, (bottomLeftCorner[1] + topRightCorner[1])/2)
    return origin

# _______________________________________
def borderEquation(polygon):
    ms = []
    bs = []
    for pointNumber in range(1, len(polygon)):
        y1 = polygon[pointNumber-1][0]
        y2 = polygon[pointNumber][0]
        x1 = polygon[pointNumber-1][1]
        x2 = polygon[pointNumber][1]
        if (x2 == x1):
            m = float('inf')
            b = x1
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m*x1
        ms.append(m)
        bs.append(b)
    return ms, bs

# _______________________________________
def isPointInPolygon(point, polygon, polygonBs, polygonMs):
    collisionPoint = 0
    xPoint = point[1]
    yPoint = point[0]

    for borderNumber in range(len(polygonBs)):
        borderM = polygonMs[borderNumber]
        if borderM == 0: # to prevent horizontal lines
            borderM = 0.001

        borderB = polygonBs[borderNumber]
        if borderM != float('inf'): # line is not vertical
            xCollisision = (yPoint - borderB)/borderM
            xThisBorder = polygon[borderNumber][1]
            xNextBorder = polygon[borderNumber+1][1]
            xMin = min([xThisBorder, xNextBorder])
            xMax = max([xThisBorder, xNextBorder])
            if xMin < xCollisision < xMax and xPoint <= xCollisision:
                collisionPoint += 1
            if xCollisision == xMin or xCollisision == xMax: # Collision point is on the start or end of one borderline
                collisionPoint += 1/2
        
        else: # line is vertical
            # borderB is x-intersect
            xCollisision = borderB
            yCollisision = yPoint
            yThisBorder = polygon[borderNumber][0]
            yNextBorder = polygon[borderNumber+1][0]
            yMin = min([yThisBorder, yNextBorder])
            yMax = max([yThisBorder, yNextBorder])
            if yMin < yCollisision < yMax and point[1] <= xCollisision:
                collisionPoint += 1
            if yCollisision == yMin or yCollisision == yMax: # Collision point is on the start or end of one borderline
                collisionPoint += 1/2

    if collisionPoint%2 == 1:   # Point is in the polygon, if number of collsions with the right side borders of point is odd.
        return True
    else:
        return False

# _________________________________________________________
# ___________________ Main Functions ______________________
# _________________________________________________________
def calcTimeForBunchOfCustomers(customersLoc, vehicle, workHourPerDay):
    # Calculate total time of travelling between bunch of customers for one visitor
    totalTimeInDay = 0
    elapsedVisitTour = 0
    endFlag = False

    remainedCustomers = customersLoc.copy()
    markedCustomers = []

    startPointNumber = findFurthestPoint(customersLoc, origin=findOrigin(customersLoc))
    markedCustomers.append([remainedCustomers[startPointNumber]])
    remainedCustomers.pop(startPointNumber)
    startLoc = markedCustomers[-1][0]
    while(not(endFlag)):
        t, nextPoint = findNearestPoint(startLoc, remainedCustomers, vehicle, distanceType='Mercator')
        if t == 0: # there is only one customer in customersLoc
            endFlag = True
        totalTimeInDay += t
        if t > workHourPerDay:
            endFlag = True

        if 0 < totalTimeInDay <= workHourPerDay:
            markedCustomers.append([remainedCustomers[nextPoint]])
            remainedCustomers.pop(nextPoint)
            startLoc = markedCustomers[-1][0]
        
        elif totalTimeInDay > workHourPerDay:
            elapsedVisitTour += 1
            totalTimeInDay = 0
            elapsedCustomers = [markedCustomers[i][0] for i in range(len(markedCustomers))]
            startLoc = findCenterPoint(elapsedCustomers)

        if (len(remainedCustomers) == 0):
            endFlag = True
            
    timeNeeded = elapsedVisitTour + totalTimeInDay/workHourPerDay
    return timeNeeded

# _______________________________________
# _______________________________________
def markCustomersTour(customersLoc, vehicle, workHourPerDay, visitTour, customersID):
    visitorNumber = 1

    endFlag = False
    totalTimeInDay = 0
    elapsedVisitTour = 0

    remainedCustomers = customersLoc.copy()
    markedCustomers = []

    startPointNumber = findFurthestPoint(customersLoc, origin=findOrigin(customersLoc))

    markedCustomers.append([remainedCustomers[startPointNumber], visitorNumber, customersID[startPointNumber]])
    remainedCustomers.pop(startPointNumber)
    customersID.pop(startPointNumber)

    startLoc = markedCustomers[-1][0]

    while(not(endFlag)):
        t, nextPoint = findNearestPoint(startLoc, remainedCustomers, vehicle, distanceType='Mercator')
        totalTimeInDay += t
        if t > workHourPerDay:
            elapsedVisitTour = 0
            totalTimeInDay = 0
            visitorNumber += 1
            startLoc = remainedCustomers[findFurthestPoint(remainedCustomers, origin=findOrigin(remainedCustomers))]

        if totalTimeInDay <= workHourPerDay:
            markedCustomers.append([remainedCustomers[nextPoint], visitorNumber, customersID[nextPoint]])
            remainedCustomers.pop(nextPoint)
            customersID.pop(nextPoint)
            startLoc = markedCustomers[-1][0]
            
        elif elapsedVisitTour < visitTour-1:
            elapsedVisitTour += 1
            totalTimeInDay = 0
            thisVisitorsLocations = []
            for i in range(len(markedCustomers)):
                if markedCustomers[i][1] == visitorNumber:
                    thisVisitorsLocations.append(markedCustomers[i][0])
            
            if len(thisVisitorsLocations) != 0:
                startLoc = findCenterPoint(thisVisitorsLocations)
            else:
                endFlag = True
        
        elif elapsedVisitTour == visitTour-1:
            elapsedVisitTour = 0
            totalTimeInDay = 0
            visitorNumber += 1
            startLoc = remainedCustomers[findFurthestPoint(remainedCustomers, origin=findOrigin(remainedCustomers))]

        if (len(remainedCustomers) == 0):
            endFlag = True

    markedCustomers = [[markedCustomers[i][0][0], markedCustomers[i][0][1], markedCustomers[i][1], markedCustomers[i][2]] for i in range(len(markedCustomers))]
    markedCustomersDF = pd.DataFrame(markedCustomers)
    markedCustomersDF.to_csv('Marked Data/Marked Customers.csv', index=False, header=False)
    
    return markedCustomers

# _______________________________________
# _______________________________________
def markCustomersVisitor(customersLoc, vehicle, workHourPerDay, visitorsNumber, customersID, maxVisitTour):
    visitTour = 1
    visitorsNumberCalced = float('inf')
    while (visitorsNumberCalced > visitorsNumber):
        markedCustomers = markCustomersTour(customersLoc, vehicle, workHourPerDay, visitTour, customersID)
        visitTour += 1
        visitorsNumberCalced = int(max(np.array(markedCustomers, dtype='int')[:,2]))
        if visitTour == maxVisitTour:
            print(f'This region cannot be divided with {visitorsNumber} visitor numbers!')
            exit()
    return markedCustomers, visitTour-1

# _______________________________________
# _______________________________________
def makeVoronoi(markedCustomers, polygonPath, distanceType, pointNumber):
    with open(polygonPath, 'r') as fHandler:
        polygon = json.load(fHandler)
    polygon = polygon['features'][0]['geometry']['coordinates'][0]
    polygon = [(point[1], point[0]) for point in polygon]

    polygonMs, polygonBs = borderEquation(polygon)    

    bottomLeftCorner = np.min(np.array(polygon), axis=0)
    topRightCorner = np.max(np.array(polygon), axis=0)
    
    precision = round((topRightCorner[0]-bottomLeftCorner[0])/pointNumber, 3)
    pointsToCoverLat = np.arange(bottomLeftCorner[0]-precision/2, topRightCorner[0]+precision/2, precision)
    pointsToCoverLon = np.arange(bottomLeftCorner[1]-precision/2, topRightCorner[1]+precision/2, precision)
    
    eachPointColor = [] # Columns = Lat, Lon, Label

    for lat in tqdm(pointsToCoverLat):
        for lon in pointsToCoverLon:
            point = (lat, lon)
            lPoint = (lat, lon-precision)
            rPoint = (lat, lon+precision)
            uPoint = (lat+precision, lon)
            dPoint = (lat-precision, lon)
            if isPointInPolygon(point, polygon, polygonBs, polygonMs):
                pointLabel = findNearestLabel(point, markedCustomers, distanceType)
                eachPointColor.append([point[0], point[1], pointLabel])
            
            # Border points
            elif isPointInPolygon(lPoint, polygon, polygonBs, polygonMs): 
                lPointLabel = findNearestLabel(lPoint, markedCustomers, distanceType)
                eachPointColor.append([point[0], point[1], lPointLabel])
            elif isPointInPolygon(rPoint, polygon, polygonBs, polygonMs): 
                rPointLabel = findNearestLabel(rPoint, markedCustomers, distanceType)
                eachPointColor.append([point[0], point[1], rPointLabel])
            elif isPointInPolygon(uPoint, polygon, polygonBs, polygonMs): 
                uPointLabel = findNearestLabel(uPoint, markedCustomers, distanceType)
                eachPointColor.append([point[0], point[1], uPointLabel])
            elif isPointInPolygon(dPoint, polygon, polygonBs, polygonMs): 
                dPointLabel = findNearestLabel(dPoint, markedCustomers, distanceType)
                eachPointColor.append([point[0], point[1], dPointLabel])

    eachPointColorDF = pd.DataFrame(eachPointColor)
    eachPointColorDF.to_csv('Marked Data/Each Point Color.csv', index=False, header=False)
    return eachPointColor

# _______________________________________
# _______________________________________
class Graph:
    def __init__(self, numberOfVertices):
        self.numberOfVertices = numberOfVertices
        self.graph = {i: [] for i in range(self.numberOfVertices)} # Dictionary to store graph
    def __str__(self):
        return str(self.graph)
        
    def addEdge(self, u, v):
        self.graph[u].append(v)
      
     # Use BFS to check path between source and destination
    def isReachable(self, source, destination):
        visited =[False]*(self.numberOfVertices)
        queue=[]

        queue.append(source)
        visited[source] = True
  
        while queue:
            neighbor = queue.pop(0)
            if neighbor == destination:
                return True

            for i in self.graph[neighbor]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
        return False
    
# _______________________________________
# _______________________________________  
def findNeighborsOfEachPoint(eachPointColor):
    numberOfVertices = len(eachPointColor)
    graph = Graph(numberOfVertices)

    locations = [(round(eachPointColor[i][0], 6), round(eachPointColor[i][1],6)) for i in range(numberOfVertices)]
    labels = [eachPointColor[i][2] for i in range(len(eachPointColor))]
    precision = round(abs(eachPointColor[1][1] - eachPointColor[0][1]), 3)
    for pointIndex, thisPoint in enumerate(tqdm(locations)):
        neighborPoints = {
            'r': (thisPoint[0], thisPoint[1]+precision),
            'l': (thisPoint[0], thisPoint[1]-precision),
            'u': (thisPoint[0]+precision, thisPoint[1]),
            'd': (thisPoint[0]-precision, thisPoint[1]),

            'ru': (thisPoint[0]+precision, thisPoint[1]+precision),
            'rd': (thisPoint[0]-precision, thisPoint[1]+precision),
            'lu': (thisPoint[0]+precision, thisPoint[1]-precision),
            'ld': (thisPoint[0]-precision, thisPoint[1]-precision)
            }
        
        thisPointLabel = labels[pointIndex]
        for direction in neighborPoints:
            try:
                possibleNeighbor = (round(neighborPoints[direction][0], 6), round(neighborPoints[direction][1], 6))
                possibleNeighborIndex = locations.index(possibleNeighbor)
                possibleNeighborLabel = labels[possibleNeighborIndex]
                
                if possibleNeighborLabel == thisPointLabel:
                    graph.addEdge(pointIndex, possibleNeighborIndex)
            except ValueError:
                pass
    
    return graph

# _______________________________________
# _______________________________________
def findCustomerVertex(eachPointColor, customerLoc, customerLabel):
    precision = round(abs(eachPointColor[1][1] - eachPointColor[0][1]), 3)/2
    numberOfPoints = len(eachPointColor)
    locationsList = [(round(eachPointColor[i][0], 6), round(eachPointColor[i][1],6)) for i in range(numberOfPoints)]
    locations = np.array(locationsList)
    
    tmpLoc = locations[locations[:, 0] < customerLoc[0]+precision]
    tmpLoc = tmpLoc[customerLoc[0]-precision < tmpLoc[:, 0]]
    tmpLoc = tmpLoc[tmpLoc[:, 1] < customerLoc[1]+precision]
    tmpLoc = tmpLoc[customerLoc[1]-precision < tmpLoc[:, 1]]
    vertexLoc = (round(tmpLoc[0,0], 6), round(tmpLoc[0,1], 6))

    vertexNumber = locationsList.index(vertexLoc)
    
    tuners = [
        (-precision, 0), (precision, 0), (0, -precision), (0, precision),
        (-precision, -precision), (precision, -precision), (-precision, precision), (precision, precision)
    ]
    for tuner in tuners:
        if customerLabel == eachPointColor[vertexNumber][2]:
            break
        tmpLoc = locations[locations[:, 0] < customerLoc[0]+tuner[0]+precision]
        tmpLoc = tmpLoc[customerLoc[0]+tuner[0]-precision < tmpLoc[:, 0]]
        tmpLoc = tmpLoc[tmpLoc[:, 1] < customerLoc[1]+tuner[1]+precision]
        tmpLoc = tmpLoc[customerLoc[1]+tuner[1]-precision < tmpLoc[:, 1]]
        if len(tmpLoc) != 0:
            vertexLoc = (round(tmpLoc[0,0], 6), round(tmpLoc[0,1], 6))
        vertexNumber = locationsList.index(vertexLoc)

    return vertexNumber

# _______________________________________
# _______________________________________
def findThreeNeighbors(markedCustomers, sourceCustomer):
    distances = []
    for customerNumber, customer in enumerate(markedCustomers):
        if sourceCustomer[2] != markedCustomers[customerNumber][2]:
            distances.append((calculateTime(sourceCustomer, customer, vehicle=None, distanceType='Euclidan'), customerNumber))
    
    distances.sort()
    threeNeighbors = [distances[i][1] for i in range(3)]
    return threeNeighbors

# _______________________________________
# _______________________________________
def modifiyCastAwayCustomers(eachPointColor, markedCustomers, vehicle, workHourPerDay, visitTour, tolerances):
    print('Step1...')
    graph = findNeighborsOfEachPoint(eachPointColor)
    markedCustomers = np.array(markedCustomers)
    visitorsNumber = int(max(markedCustomers[:,2]))

    prevCustomersLen = 0
    thisCustomerNeighbors = []
    
    
    fullyFinishedCustomers = {} # Customers which time needed is equalt to visit tour
    almostFinishedCustomers = {} # Customers which time needed is greater than tolerance1*visit tour
    hardlyFinishedCustomers = {} # Customers which time needed is greater than tolerance2*visit tour
    notFinishedCustomers = {} # Customers which time needed is lesser than tolerance2*visit tour (Cast Away Customers)
    
    print('Step2...')
    for visitorNumber in tqdm(range(1, visitorsNumber+1)):
        thisVisitorCustomers = markedCustomers[markedCustomers[:, 2] == visitorNumber]
        setNumberFully = 0
        setNumberAlmost = 0
        setNumberHardly = 0
        setNumberCastAway = 0
        thisVisitorNeighboredCustomers = [] # All neighbored customers of this visitor

        for customerNumber in range(len(thisVisitorCustomers)):
            if customerNumber in thisVisitorNeighboredCustomers:
                continue
            #Else:
            customerLoc = (thisVisitorCustomers[customerNumber, 0], thisVisitorCustomers[customerNumber, 1])
            sourceVertex = findCustomerVertex(eachPointColor, customerLoc, visitorNumber)
            
            thisCustomerNeighbors = [customerNumber+prevCustomersLen]
            thisVisitorNeighboredCustomers.append(customerNumber)
            thisNeighborCustomersLoc = [customerLoc]

            for possibleNeighborCustomerNumber in range(len(thisVisitorCustomers)):
                if possibleNeighborCustomerNumber == customerNumber or possibleNeighborCustomerNumber in thisVisitorNeighboredCustomers:
                    continue
                #Else:
                possibleNeighborLoc = (
                    thisVisitorCustomers[possibleNeighborCustomerNumber, 0], 
                    thisVisitorCustomers[possibleNeighborCustomerNumber, 1])
                
                destinationVertex = findCustomerVertex(eachPointColor, possibleNeighborLoc, visitorNumber)
                if graph.isReachable(sourceVertex, destinationVertex):
                    thisCustomerNeighbors.append(possibleNeighborCustomerNumber+prevCustomersLen)
                    thisVisitorNeighboredCustomers.append(possibleNeighborCustomerNumber)
                    thisNeighborCustomersLoc.append(possibleNeighborLoc)

            neighborsTime = calcTimeForBunchOfCustomers(thisNeighborCustomersLoc, vehicle, workHourPerDay)

            if neighborsTime >= tolerances[0]*visitTour:
                fullyFinishedCustomers[(visitorNumber, setNumberFully)] = thisCustomerNeighbors
                setNumberFully += 1
            
            elif neighborsTime >= tolerances[1]*visitTour:
                almostFinishedCustomers[(visitorNumber, setNumberAlmost)] = thisCustomerNeighbors
                setNumberAlmost += 1
            
            elif neighborsTime >= tolerances[2]*visitTour:
                hardlyFinishedCustomers[(visitorNumber, setNumberHardly)] = thisCustomerNeighbors
                setNumberHardly += 1

            else:
                notFinishedCustomers[(visitorNumber, setNumberCastAway)] = thisCustomerNeighbors
                setNumberCastAway += 1
            
        prevCustomersLen += len(thisVisitorCustomers)
    
    # -----
    print('Modifying...')
    for notFinishedVisitor in tqdm(notFinishedCustomers):
        for castAwayCustomer in notFinishedCustomers[notFinishedVisitor]:
            threeNeighbors = findThreeNeighbors(markedCustomers, markedCustomers[castAwayCustomer])
            neighborAssigned = False
            
            # Change label in according to priorities
            # 1. Hardly finished
            if not(neighborAssigned):
                for hardlyFinishedVisitor in hardlyFinishedCustomers:
                    for neighbor in threeNeighbors:
                        if neighbor in hardlyFinishedCustomers[hardlyFinishedVisitor]:
                            markedCustomers[castAwayCustomer][2] = hardlyFinishedVisitor[0]
                            hardlyFinishedCustomers[hardlyFinishedVisitor].append(castAwayCustomer)
                            neighborAssigned = True
                            break
                    if neighborAssigned:
                        break
            
            # 2. Almost finished
            if not(neighborAssigned):
                for almostFinishedVisitor in almostFinishedCustomers:
                    for neighbor in threeNeighbors:
                        if neighbor in almostFinishedCustomers[almostFinishedVisitor]:
                            markedCustomers[castAwayCustomer][2] = almostFinishedVisitor[0]
                            almostFinishedCustomers[almostFinishedVisitor].append(castAwayCustomer)
                            neighborAssigned = True
                            break
                    if neighborAssigned:
                        break
            
            # 3. Fully finished
            if not(neighborAssigned):
                for fullyFinishedVisitor in fullyFinishedCustomers:
                    for neighbor in threeNeighbors:
                        if neighbor in fullyFinishedCustomers[fullyFinishedVisitor]:
                            markedCustomers[castAwayCustomer][2] = fullyFinishedVisitor[0]
                            fullyFinishedCustomers[fullyFinishedVisitor].append(castAwayCustomer)
                            neighborAssigned = True
                            break
                    if neighborAssigned:
                        break
            
            # 4. Cast away customers which its label not corrected yet
            if not(neighborAssigned):
                for notFinishedVisitor in notFinishedCustomers:
                    for neighbor in threeNeighbors:
                        if neighbor in notFinishedCustomers[notFinishedVisitor]:
                            markedCustomers[castAwayCustomer][2] = notFinishedVisitor[0]
                            if notFinishedCustomers[notFinishedVisitor][-1] != castAwayCustomer:
                                notFinishedCustomers[notFinishedVisitor].append(castAwayCustomer)
                            neighborAssigned = True
                            break
                    if neighborAssigned:
                        break

    markedCustomersDF = pd.DataFrame(markedCustomers)
    markedCustomersDF.to_csv('Marked Data/Marked Customers.csv', index=False, header=False)
    return markedCustomers

# _______________________________________
# _______________________________________
# _______________________________________
def inRegionCustomers(polygonPath):
    customersInfo = pd.read_csv('Tables/VisitorsInfo.csv', low_memory=False)
    customersLat = customersInfo.Y
    customersLon = customersInfo.X
    ccCustomer = customersInfo.ID

    with open(polygonPath, 'r') as fHandler:
        polygon = json.load(fHandler)
    polygon = polygon['features'][0]['geometry']['coordinates'][0]
    polygon = [(point[1], point[0]) for point in polygon]
    polygonMs, polygonBs = borderEquation(polygon)

    customersLoc = []
    customersID = []
    for customerNumber in range(len(customersLat)):
        customerLoc = (customersLat[customerNumber], customersLon[customerNumber])
        if isPointInPolygon(customerLoc, polygon, polygonBs, polygonMs):
            customersLoc.append(customerLoc)
            customersID.append(ccCustomer[customerNumber])
    
    return customersLoc, customersID

# _______________________________________
# _______________________________________
def randomCustomers(pointsNumber):
    customersInfo = pd.read_csv('Tables/VisitorsInfo.csv', low_memory=False)
    customersLat = customersInfo.Y
    customersLon = customersInfo.X
    customersID = customersInfo.ID

    selectedPoints = np.random.choice(range(len(customersLon)), size=pointsNumber, replace=False).ravel()
    customersLoc = [(customersLat[i], customersLon[i]) for i in selectedPoints]
    customersID = [customersID[i] for i in selectedPoints]
    return customersLoc, customersID

# _______________________________________
# _______________________________________
def editVisitorsInfo():
    visitorsInfo = pd.read_csv('Tables/VisitorsInfo.csv')
    markedCustomers = np.array(pd.read_csv('Marked Data/Marked Customers.csv', header=None))

    customersID = visitorsInfo.ID
    nameMoshtary = visitorsInfo.NameMoshtary
    X = visitorsInfo.X
    Y = visitorsInfo.Y
    shomareVisitor = np.array(visitorsInfo.ShomareVisitor)

    for customer in markedCustomers:
        customerID = customer[3]
        shomareVisitor[customersID == customerID] = customer[2]

    visitorsInfoFile = {}
    visitorsInfoFile['ID'] = customersID
    visitorsInfoFile['NameMoshtary'] = nameMoshtary
    visitorsInfoFile['X'] = X
    visitorsInfoFile['Y'] = Y
    visitorsInfoFile['ShomareVisitor'] = shomareVisitor

    pd.DataFrame(visitorsInfoFile).to_csv('Tables/VisitorsInfo.csv', index=True, header=True)

# _______________________________________
# _______________________________________
def getParameters(visitorOrTour):
    workHourPerDay = 8
    vehicle = 'Walk'
    pointNumber = 40
    if visitorOrTour == 'Tour':
        visitTour = 4
        visitorsNumber = -1
    else:
        visitTour = -1
        visitorsNumber = 21
    return workHourPerDay, vehicle, visitTour, visitorsNumber, pointNumber
            
# ___________________________________________________________________________________________________________
# ____________________________________________ Main Routine _________________________________________________
# ___________________________________________________________________________________________________________
if __name__ == '__main__':
    visitorOrTours = ['Visitor', 'Tour'] # Which one is constant? Visit Tour ('Tour') or Visitor Number ('Visitor')
    visitorOrTour = visitorOrTours[1]
    workHourPerDay, vehicle, visitTour, visitorsNumber, pointNumber = getParameters(visitorOrTour)

    distanceTypes = ['Manhattan', 'Euclidan', 'Mercator', 'Chebyshev']
    polygonPath = 'Maps/Tehran.geojson'
    polygonName = polygonPath.replace('Maps/', '').replace('.geojson', '')
    # customersLoc, customersID = randomCustomers(1000)
    customersLoc, customersID = inRegionCustomers(polygonPath)

    print('Labeling Customers...')
    if visitorOrTour == 'Tour':
        markedCustomers = markCustomersTour(customersLoc, vehicle, workHourPerDay, visitTour, customersID)
    
    elif visitorOrTour == 'Visitor':
        markedCustomers, visitTour = markCustomersVisitor(customersLoc, vehicle, workHourPerDay, visitorsNumber, customersID)
        print('Visit Tour:', visitTour, 'days')

    print('Assigning Regions...')
    # distanceType: for find nearestLabel
    eachPointColor = makeVoronoi(markedCustomers, polygonPath, distanceType=distanceTypes[1], pointNumber=pointNumber) 
    
    visitorsNumber = max(np.array(markedCustomers, dtype='int')[:,2])
    if visitorsNumber != 1:
        print('Checking...')
        modifiedCustomers = modifiyCastAwayCustomers(
            eachPointColor, markedCustomers, vehicle, 
            workHourPerDay, visitTour, tolerances=(0.95, 0.8, 0.7)
            )
        print('Assigning Modified Regions...')
        eachPointColor = makeVoronoi(modifiedCustomers, polygonPath, distanceType=distanceTypes[1], pointNumber=pointNumber)

    print('Editing Database...')    
    editVisitorsInfo()

    showOnMap(polygonPath=polygonPath, polygonName=polygonName)

