# ARAxai

<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/araxai">
<img alt="PyPI - Wheel" src="https://img.shields.io/pypi/wheel/araxai">
<img alt="PyPI - Status" src="https://img.shields.io/pypi/status/araxai">
<img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/araxai">

## What is ARA (ARAxai package)

ARAxai (association rule analysis) is a profiling tool that discovers main influencers in data. It can be also used to explain the model by simplification to outline most significant influencers.

It uses categorical data sets and search for most significal influencers. For them, deep dive is made by calling ara for this subset.

## Installing ARAxai

Installation is simple. Simply run 

pip install araxai

Alternative way is to install manually from [PyPI](https://pypi.org/project/araxai/)

## Running ARAxai

The key command to run data/model profiling is the

```
a = ara(df=df,target='Severity',target_class='Fatal',options={"min_base":1})
a.print_result()
```

Parameters are

* **df** - dataframe with categorical data
* **target** - target variable
* **target_class** - class of target variable to find influencers for
* **options** - options( see below)


The complex example how to use the data is in following box. Please just copy accidents.zip file from the Github repository to folder with your code.

```
from sklearn.impute import SimpleImputer
import pandas as pd
from araxai import ara
import matplotlib.pyplot as plt
import os



dir = os.getcwd()

os.makedirs("fig",exist_ok=True)

dir = os.path.join(dir,"fig")

df = pd.read_csv (os.path.join(os.getcwd(),'accidents.zip'), encoding='cp1250', sep='\t')

df=df[['Driver_Age_Band','Driver_IMD','Sex','Journey','Hit_Objects_in','Hit_Objects_off','Casualties','Severity']]

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

a = ara(df=df,target='Severity',target_class='Fatal',options={"min_base":1})

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

```

## Interpreting text output

```
-.. Journey(4,Pupil riding to/from school) /4.3 (=/4.3)
    ++. Driver_Age_Band(21 - 25) x6.0 (=x1.4)
    +.. Driver_IMD(3) x2.6 (=/1.7)
    +.. Driver_IMD(6) x2.2 (=/1.9)
    +.. Driver_IMD(7) x2.0 (=/2.1)
    +.. Driver_IMD(9) x2.2 (=/1.9)
    +++ Hit_Objects_in(7) x27.9 (=x6.5)
    ++. Hit_Objects_off(2) x8.3 (=x1.9)
...
```
First line is the top-level. We can see that for *Journey(4,Pupil riding to/from school)* there is 4.3 lower probablity of Fatal accident compared to the entire base.
Second (and following lines until indent - that are all due to showing only short part of output here) are strong influencers for subset *Journey(4,Pupil riding to/from school)*. We can see that in this subset, for *Driver_Age_Band(21 - 25)*, there is 6 times higher probablity of Fatal accident than in the entire subset. Moreover, it is 1.4 higher probability than on the entire set (information in parenthnesses). And two plus signs means that it is stronger influencers (default boundaries for number of pluses are [2,5,10] and lift is compared to these values). 

## ARA options

Currently there are several options available

* **min_base** - set minimum base for the rule
* **max_depth** - maximum depth of the deep dive
* **boundaries** - list of boundaries to consider influencer as a strong (measured by lift). Default [2,5,10]. First is the minimal lift for including, next are boundaries to be considered as stronger (number of +/- signs in output)

## CLARA method

There is also local method called CLARA included. It finds combination of conditions and ARA rules and shows the strongest influencers.

Example is here (please don't forget to copy pistachio.zip datafile from the GITHUB repository.

```
import os

from sklearn.impute import SimpleImputer
import pandas as pd
from araxai import ara




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

a = ara(df=df,target='Class',target_class='Siit_Pistachio',CL=['AREA', 'PERIMETER', 'MAJOR_AXIS'])




print("")
print("Demonstration how to get machine readable inputs:")
print(f"   ...full results : {a.get_task_info()}")
print(f"   ...only results : {a.get_results()}")
print(f"   ...only rules   : {a.get_rules()}")

a.print_result()

a.print_task_info()

a.print_statisics()

```
## AUTO BOUNDARY option

For some cases, there are no strong influencers or there is too much strong influencers. Analyst may fine-tune boundary parameter or use *auto_boundary* option to get 2-10 strongest influencers.

Example is below:

```
from sklearn.impute import SimpleImputer
import pandas as pd
from araxai import ara
import matplotlib.pyplot as plt
import os
from pandas.api.types import CategoricalDtype


#This is a demo of ARA on multiple datasets downloaded from Kaggle and automatically binned
#This demonstration shows how many rules are returned and allows to switch on auto_boundary function to find 2-10 rules (see switch below)
#at the end, you will see how many rules and subrules as been found for individual datasets and number of rules in each individual iterarion when looking for 2-10 rules


#Please copy the sample pickle data (data_pkl folder) as a subfolder of folder with your sample code

cwd = os.getcwd()

dir = os.path.join(cwd,"data_pkl")

dirfig = os.path.join(dir,"fig")


ds = pd.read_csv(os.path.join(dir,'datasets.csv'),sep=';')

ds['rules']=None
ds['records']=None
ds['boundaries']=None
ds['changes']=None
ds['rules_iter']=None

print(ds)

for ind in ds.index:
    print(f"...will go for dataset {ds['filename'][ind]}, target {ds['target'][ind]}, target_class {ds['target_class'][ind]}")
    fname=ds['filename'][ind]
    df = pd.read_pickle(os.path.join(dir,fname))
    print(f"......will go for column {ds['target'][ind]} out of {df.columns}")
    tgt=ds['target'][ind]
    df2 = df[tgt]
    print(df2.unique())
    cls = ds['target_class'][ind]

    if cls.isnumeric():
        cls=int(cls)


# --------->  Change switch to False (only influencers with lift>=2) or True (get top 2-10 influencers no matter how stong they are <---------------------
    auto_boundaries = True

    a = ara(df=df,target=tgt,target_class=cls,options={"min_base":2,"auto_boundaries":auto_boundaries})

    print(f"res: {a.res}")

    print(f"results: {a.res['results']}")

    print(a.res['results']['rules'])


    ds['rules'][ind] = a.res['results']['rules']
    if auto_boundaries:
        ds['rules_iter'][ind] = a.res['task_info']['auto_boundaries']['rules_iter']
    else:
        ds['rules_iter'][ind] = 1

    newdir=os.path.join(dirfig,os.path.splitext(ds['filename'][ind])[0])
    if not(os.path.exists(newdir)):
        os.makedirs(newdir)

    a.draw_result(dir=newdir)


print("")
print("RESULTS:")
print("")

for ind in ds.index:
    rules = ds['rules'][ind]
    cnt_rules = len(rules)
    # ONLY FOR LEVEL 2!!!
    cnt_subrules = 0
    for ii in range(cnt_rules):
        rule = rules[ii]
        cnt_subrules += len(rule['sub'])

    print(f"...rules: {cnt_rules}, subrules: {cnt_subrules}, iterations {ds['rules_iter'][ind]} dataset {ds['filename'][ind]}")


```
