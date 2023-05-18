import pandas as pd

customersInfo = pd.read_csv('Tables/tblFO_Moshtary.csv', low_memory=False)

ccMoshtary = customersInfo.ccMoshtary
nameMoshtary = customersInfo.NameMoshtary
X = customersInfo.X
Y = customersInfo.Y
shomareVisitor = [-1]*len(ccMoshtary)

shomareVisitorFile = {}
shomareVisitorFile['ID'] = ccMoshtary
shomareVisitorFile['NameMoshtary'] = nameMoshtary
shomareVisitorFile['X'] = X
shomareVisitorFile['Y'] = Y
shomareVisitorFile['ShomareVisitor'] = shomareVisitor

pd.DataFrame(shomareVisitorFile).to_csv('Tables/VisitorsInfo.csv', index=True, header=True)