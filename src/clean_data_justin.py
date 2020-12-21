import zipfile
import pandas as pd

# with zipfile.ZipFile('../data/Test.zip', 'r') as testZip:
#     testZip.extractall('../data')

trainDf = pd.read_csv('../data/Train.csv')

print(trainDf.info())
trainDf['saledate'] = pd.to_datetime(trainDf['saledate'])

dateSort = trainDf.sort_values('saledate').copy()
print(dateSort['saledate'].head())

# uniqueDic = {}



# for idx in trainDf.columns:
#     uniqueDic[idx] = trainDf[idx].unique()

# print(uniqueDic)    