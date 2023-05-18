from time import sleep
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import folium
from folium.plugins import MiniMap, Draw, MousePosition, Geocoder, BeautifyIcon
from os import system
from datetime import datetime, time, timedelta

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
def calculateTime(firstLoc, secondLoc, vehicle, visitingTime):
    # Visiting Time: Time needed that visitor remain in customer store in hour
    if vehicle == None:
        vehicle = 'Car'
    # Speeds are in km/h, and return time in hour
    speeds = {'Bike': 14, 'Car': 60, 'Walk': 3}
    speed = speeds[vehicle]

    distance = findDistanceOnMap(firstLoc, secondLoc)

    t = distance/speed + visitingTime
    return t

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
def getCurrentLoc():
    return (38.9893, 45.07141)

# _________________________________________________________
def getVisitorRestTime():
    return (time(13,0,0), time(14,0,0))

# _________________________________________________________
def scaleVector(vector):
    denominator = max(vector) - min(vector)
    if (denominator) != 0:
        return (vector - min(vector))/denominator
    else:
        return [1]*len(vector)

# _________________________________________________________
def getParameters():
    workHourPerDay = 8
    visitTour = 4
    vehicle = 'Car'
    visitorNumber = 1
    visitingTime = 0.2
    currentLocation = getCurrentLoc()
    return workHourPerDay, visitTour, vehicle, visitorNumber, currentLocation, visitingTime

# _________________________________________________________
def getPrevParameters(visitorNumber, visitTour):
    visitorsActivity = pd.read_csv('Tables/Visitors Activity.csv', index_col=0)
    visitorRow = visitorsActivity[visitorsActivity['Visitor Number'] == visitorNumber]

    if len(visitorRow) != 0:
        elapsedDays = visitorRow['Elapsed Days'].values[0]
        visitedCustomers = visitorRow['Visited Customers'].values[0]
        if str(visitedCustomers) != 'nan':
            visitedCustomers = visitedCustomers.split(',')
        else:
            visitedCustomers = []
        visitedCustomers = [int(visitedCustomer) for visitedCustomer in visitedCustomers]
        
        if elapsedDays == visitTour:
            elapsedDays = 0
            visitedCustomers = []
            visitorsActivity.loc[visitorsActivity['Visitor Number'] == visitorNumber, 'Visited Customers'] = ''
            visitorsActivity.loc[visitorsActivity['Visitor Number'] == visitorNumber, 'Elapsed Days'] = elapsedDays
            visitorsActivity.to_csv('Tables/Visitors Activity.csv')

    else:
        elapsedDays = 0
        visitedCustomers = []
    
    return elapsedDays, visitedCustomers

# _________________________________________________________
def changeKPI(rate):
    maxRate = 10
    # rate = maxRate + 10
    # while (rate<0 or rate>maxRate):
    #     rate = int(input(f'Please rate the ranking with number 0 (worst) to {maxRate} (best): '))
    #     if rate<0 or rate>maxRate:
    #         print('Invalid!')
    rate = maxRate - rate
    kpis = pd.read_csv('Tables/KPI.csv', index_col=[0])
    for kpiName in kpis.iloc[:, 1:]:
        kpiChanger = np.random.random(size=100)
        q = int(10*rate*np.random.random())
        alpha = round(np.percentile(kpiChanger, q=q, interpolation='midpoint'), 3)
        mOrP = np.random.random() < 0.5
        
        originalSign = np.sign(kpis[kpiName].values[0])
        possibleNewKPIs = [kpis[kpiName].values[0] - alpha*rate, kpis[kpiName].values[0] + alpha*rate]
        
        if np.sign(possibleNewKPIs[0]) == originalSign and np.sign(possibleNewKPIs[1]) != originalSign:
            kpis[kpiName] = possibleNewKPIs[0]

        elif np.sign(possibleNewKPIs[0]) != originalSign and np.sign(possibleNewKPIs[1]) == originalSign:
            kpis[kpiName] = possibleNewKPIs[1]

        elif np.sign(possibleNewKPIs[0]) == originalSign and np.sign(possibleNewKPIs[1]) == originalSign:
            if mOrP:
                kpis[kpiName] = possibleNewKPIs[0]
            else:
                kpis[kpiName] = possibleNewKPIs[1]

    kpis.to_csv('Tables/KPI.csv')

# _________________________________________________________
def editVisitorsActivity(visitorNumber, elapsedDays, visitedCustomers):
    visitorsActivity = pd.read_csv('Tables/Visitors Activity.csv', index_col=0)

    if len(visitorsActivity[visitorsActivity['Visitor Number'] == visitorNumber]) != 0:
        prevVisitedCustomers = visitorsActivity.loc[visitorsActivity['Visitor Number'] == visitorNumber, 'Visited Customers'].values[0]
        if str(prevVisitedCustomers) == 'nan': # New visit tour
            newVisitedCustomersStr = str(visitedCustomers)[1:-1]
        else:
            newVisitedCustomersStr = prevVisitedCustomers+', '+str(visitedCustomers)[1:-1]
        visitorsActivity.loc[visitorsActivity['Visitor Number'] == visitorNumber, 'Visited Customers'] = newVisitedCustomersStr
        visitorsActivity.loc[visitorsActivity['Visitor Number'] == visitorNumber, 'Elapsed Days'] = elapsedDays

    else:
        newRow = pd.DataFrame(
            {'Visitor Number': visitorNumber, 
            'Elapsed Days': elapsedDays, 
            'Visited Customers': str(visitedCustomers)[1:-1]},
            index=[0]
        )
        visitorsActivity = visitorsActivity.append(newRow, ignore_index=True)
    
    visitorsActivity.to_csv('Tables/Visitors Activity.csv')

# _________________________________________________________
# _________________________________________________________
def showOnMap(customerLocs, customersID, visitorNumber):
    locsCenter = findCenterPoint(customerLocs)
    
    Tiles = ['CartoDB positron', 'Stamen Terrain', 'Stamen Toner', 
    'OpenStreetMap', 'Stamen Watercolor', 'CartoDB Dark_matter']
    visitorMap = folium.Map(locsCenter, zoom_start=7, tiles=Tiles[3], control_scale=True)

    # Visitor
    folium.Marker(location=customerLocs[0], 
        icon=folium.Icon(color='orange', icon_color='black', icon='male', prefix='fa'), 
        popup='Visitor'
        ).add_to(visitorMap)
        
    # Connect customers in order
    folium.PolyLine(locations=customerLocs, color='purple', opacity=0.7).add_to(visitorMap)
    # Customers Marker
    for customerNumber in range(1, len(customerLocs)):
        folium.Marker(location=customerLocs[customerNumber], 
        icon=BeautifyIcon(
            icon_shape='marker',
            border_color='black',
            text_color='white',
            icon_size=(36, 36), 
            background_color='red',
            number=f'{int(customerNumber)}'
            ),
        tooltip=customerLocs[customerNumber], popup=customersID[customerNumber-1]
        ).add_to(visitorMap)
    
    minimap = MiniMap(toggle_display=True)
    visitorMap.add_child(minimap)
    draw = Draw(export=True)
    draw.add_to(visitorMap)
    MousePosition().add_to(visitorMap)
    folium.LayerControl().add_to(visitorMap)
    Geocoder(position='bottomright').add_to(visitorMap)

    visitorMap.save(f'Maps/Visitor Map{visitorNumber}.html')
    system(f'google-chrome "Maps/Visitor Map{visitorNumber}.html"')

# ___________________________________________________________________________________________________________
# ____________________________________________ Main Routine _________________________________________________
# ___________________________________________________________________________________________________________
if __name__ == '__main__':
    workHourPerDay, visitTour, vehicle, visitorNumber, currentLocation, visitingTime = getParameters()
    elapsedDays, visitedCustomers = getPrevParameters(visitorNumber, visitTour)
    visitorRestTime = getVisitorRestTime()
    visitorRestHour = visitorRestTime[1].hour - visitorRestTime[0].hour
    visitorRestMinute = visitorRestTime[1].minute - visitorRestTime[0].minute
    visitorRestTime = visitorRestHour + (visitorRestMinute/60)*100
    # Read from tables
    customersKPI = pd.read_csv('Tables/CustomerKPI.csv', index_col=[0])
    visitorsInfo = pd.read_csv('Tables/VisitorsInfo.csv', index_col=[0])
    coeffs = pd.read_csv('Tables/KPI.csv', index_col=[0])

    # Clean datasets
    customersKPI = customersKPI.fillna(0)

    # Edit and assimilate column names
    customersKPI.columns = customersKPI.columns.str.capitalize()
    coeffs.columns = coeffs.columns.str.capitalize()
    visitorsInfo.columns = visitorsInfo.columns.str.capitalize()
    
    # Scale the data
    customerKPIColumns = customersKPI.columns
    customersKPIID = pd.DataFrame(customersKPI.Id, columns=[customerKPIColumns[0]])
    scaler = MinMaxScaler()
    customersKPI = pd.DataFrame(scaler.fit_transform(customersKPI.drop('Id', axis=1)), columns=customerKPIColumns[1:])
    customersKPI = pd.concat((customersKPIID, customersKPI), axis=1)

    # Sort customers of determined visitor
    customersOfThisVisitor = visitorsInfo[visitorsInfo.Shomarevisitor == visitorNumber]
    customersScores = []
    customersSorted = []
    locations = [currentLocation]

    for customerID in customersOfThisVisitor.Id:
        if customerID in visitedCustomers:
            continue
        customerInfo = customersKPI[customersKPI.Id == customerID]
        customerInfo = customerInfo.drop('Id', axis=1)

        thisCustomerScore = 0
        for kpi in customerInfo:
            thisCustomerScore += customerInfo[kpi].values[0]*coeffs.loc[0, kpi]
        customersScores.append([thisCustomerScore, customerID])
    
    originalCustomersScores = customersScores.copy()
    # Calculating Times:
    totalTime = 0
    while (len(originalCustomersScores) != 0):
        customersScores = originalCustomersScores.copy()
        elapsedTimes = np.array([])
        for customerScore in customersScores:
            thisCustomerLoc = (
                visitorsInfo[visitorsInfo.Id == customerScore[1]].Y.values[0],
                visitorsInfo[visitorsInfo.Id == customerScore[1]].X.values[0]
                )
            elapsedTimes = np.append(elapsedTimes, calculateTime(currentLocation, thisCustomerLoc, vehicle, visitingTime))
        
        # Scale Times
        elapsedTimesScaled = scaleVector(elapsedTimes)
        for i, customerScore in enumerate(customersScores):
            customerScore[0] += coeffs.loc[0, 'Faselemoshtarybadi']*elapsedTimesScaled[i]
        
        customersScores.sort()
        bestCustomer = customersScores[-1]
        totalTime += elapsedTimes[originalCustomersScores.index(bestCustomer)]

        now = datetime.now()
        hour = int(totalTime)
        minuteAndSecond = (totalTime - int(totalTime))*60
        minute = int(minuteAndSecond)
        second = int((minuteAndSecond - minute)*60)
        reachingDestinationTime = now + timedelta(hours=hour, minutes=minute, seconds=second)
        
        if totalTime > workHourPerDay - visitorRestTime:
            break
        # Else:
        customersSorted.append(bestCustomer[1])
        currentLocation = (
                visitorsInfo[visitorsInfo.Id == bestCustomer[1]].Y.values[0],
                visitorsInfo[visitorsInfo.Id == bestCustomer[1]].X.values[0]
                )

        originalCustomersScores.remove(bestCustomer)
        locations.append(currentLocation)
    
    elapsedDays += 1
    editVisitorsActivity(visitorNumber, elapsedDays, customersSorted)
    showOnMap(locations, customersSorted, visitorNumber)
    changeKPI()
    
