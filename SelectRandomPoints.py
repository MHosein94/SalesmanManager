import numpy as np
import pandas as pd
customersInfo = pd.read_csv('Tables/Customer.csv')
customersLat = customersInfo.X
customersLon = customersInfo.Y

selectedPoints = pd.DataFrame(np.random.choice(range(len(customersLon)), size=1000, replace=False))
selectedPoints.to_csv('Random Points/Selected Points4.csv', index=False)
