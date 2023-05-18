import numpy as np
import pandas as pd

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
def findNearestLabel(point):
    markedCustomers = np.array(pd.read_csv('Borders/Marked Customers.csv', header=None))
    minDistance = float('inf')

    for customerInfo in markedCustomers:
        customerLoc = (customerInfo[0], customerInfo[1])
        customerLabel = customerInfo[2]

        distance = abs(point[0] - customerLoc[0]) + abs(point[1] - customerLoc[1])
        if distance < minDistance:
            minDistance = distance
            nearestLabel = customerLabel

    return int(nearestLabel)

# _________________________________________________________
def getNewCustomerLoc():
    return (35.61126, 51.32298)
# _________________________________________________________
# _________________________________________________________
if __name__ == '__main__':
    newCustomerLoc = getNewCustomerLoc()
    newCustomerVisitor = findNearestLabel(newCustomerLoc)
    print(newCustomerVisitor)
