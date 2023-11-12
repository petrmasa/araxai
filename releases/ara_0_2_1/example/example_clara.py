import os

from sklearn.impute import SimpleImputer
import pandas as pd
from ara import ara




def getlabels(s):
    lst = []
    for i in range(len(s)-1):
        j=i+1
        if (j==1):
            item = '<' + str(s[i]) + ','+ str(s[i+1]) + '>'
        else:
            item = '(' + str(s[i]) + ','+ str(s[i+1]) + '>'
        lst.append(item)

    print(s)
    print(lst)
    return lst

cwd = os.getcwd()

original=pd.read_csv(os.path.join(cwd,'pistachio.zip'))

print(original.columns)
print(original['Class'].unique())


to_qcut=['AREA', 'PERIMETER', 'MAJOR_AXIS', 'MINOR_AXIS', 'ECCENTRICITY',
       'EQDIASQ', 'SOLIDITY', 'CONVEX_AREA', 'EXTENT', 'ASPECT_RATIO',
       'ROUNDNESS', 'COMPACTNESS', 'SHAPEFACTOR_1', 'SHAPEFACTOR_2',
       'SHAPEFACTOR_3', 'SHAPEFACTOR_4']


for varname in to_qcut:
    original[varname] = pd.qcut(original[varname], q=5)


df=original

a = ara.ara(df=df,target='Class',target_class='Siit_Pistachio',CL=['AREA', 'PERIMETER', 'MAJOR_AXIS'])




print("")
print("Demonstration how to get machine readable inputs:")
print(f"   ...full results : {a.get_task_info()}")
print(f"   ...only results : {a.get_results()}")
print(f"   ...only rules   : {a.get_rules()}")

a.print_result()

a.print_task_info()

a.print_statisics()



