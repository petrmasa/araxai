from sklearn.impute import SimpleImputer
import pandas as pd
from ara import ara
import matplotlib.pyplot as plt
import os



dir = os.getcwd()

os.makedirs("fig",exist_ok=True)

dir = os.path.join(dir,"fig")

df = pd.read_csv (os.path.join(os.getcwd(),'accidents.zip'), encoding='cp1250', sep='\t')

df=df[['Driver_Age_Band','Driver_IMD','Sex','Journey','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

a = ara.ara(df=df,target='Severity',target_class='Fatal',options={"min_base":1})

#print text results
a.print_result()

#print task summary
a.print_task_info()

#print run statistics
a.print_statisics()

#export charts/results
a.draw_result()

print("")
print("Demonstration how to get machine readable inputs:")
print(f"   ...full results : {a.get_task_info()}")
print(f"   ...only results : {a.get_results()}")
print(f"   ...only rules   : {a.get_rules()}")





