from sklearn.impute import SimpleImputer
import pandas as pd
from ara import ara
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

    a = ara.ara(df=df,target=tgt,target_class=cls,options={"min_base":2,"auto_boundaries":auto_boundaries})

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

