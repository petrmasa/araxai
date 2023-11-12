import os

from sklearn.impute import SimpleImputer
import pandas as pd
from araxai import ara

print(os.getcwd())

df = pd.read_csv (os.path.join(os.getcwd(),'accidents.zip'), encoding='cp1250', sep='\t')

df=df[['Driver_Age_Band','Driver_IMD','Sex','Journey','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

a = ara.arap(df,'Severity','Fatal',options={"min_base":1})

print(a)

a=a["results"]["rules"]

ara.print_result(res=a)