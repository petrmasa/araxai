
from cleverminer import cleverminer

import pandas
import time

class ara:
    """
    Categorical Lift-based Association Rules Analysis.
    """

    version_string = "0.1.0"
    max_depth = 2
    min_base = 20
    print_rules_on_the_fly = 0
    boundaries = [2,5,10]

    stats={}

    start_time = None
    end_time = None

    clms=[]

    def __init(self):
        """
        Initializes a class.
        """

    @staticmethod
    def arap(df:pandas.DataFrame=None,target=None,target_class=None,cond=None,clm=None,prepend_str='',depth=1,options=None,lift_pre=1):
        cumlift=lift_pre
        if cond==None:
            ara.start_time=time.time()
            ara.stats={}
            ara.stats["clm_runs"]=0
            ara.stats["resulting_rules"]=0
            print(f"ARA version {ara.version_string}")
            if not(options is None):
                if type(options) is dict:
                    if "max_depth" in options:
                        ara.max_depth=options.get("max_depth")
                    if "min_base" in options:
                        ara.min_base = options.get("min_base")
                    if "print_rules_on_the_fly" in options:
                        ara.print_rules_on_the_fly = options.get("print_rules_on_the_fly")
                        print("WARNING: this option is marked as experimental. Will be replaced.")
                else:
                    print("ERROR: options must be a dictionary")
                    return
            ara.clms = []
            for i in range(ara.max_depth):
                print(f"..will initialize CLM#{i+1}")
                clm_l=cleverminer(df=df)
                ara.clms.append(clm_l)
        res_ara = []
        if (df is None):
            print("Dataframe is missing")
            return
        if (target is None):
            print("Target is missing")
            return
        if not(target in df.columns):
            print(f"{target} is not present in the dataframe.")
            return
        def var_str_to_literal(name):
            d = {}
            d['name']=name
            d['type']='subset'
            d['minlen']=1
            d['maxlen']=1
            return d
        an=[]
        def cond_str_lst(cond):
            res=[]
            if cond is None:
                return res
            attr= cond['attributes']
            for i in attr:
                res.append(i['name'])
            return res
        for nm in df.columns:
            if not(nm==target):
                if not(target in cond_str_lst(cond)):
                    an.append(var_str_to_literal(nm))
        su=[]
        if target_class is None:
            su.append(var_str_to_literal(target))
        else:
            d = {}
            d['name']=target
            d['type']='one'
            d['value']=target_class
            su.append(d)
        clm = ara.clms[depth-1]
        if cond is None:
            clm.mine(proc='4ftMiner',quantifiers={'Base':ara.min_base}, ante ={'attributes':an, 'minlen':1, 'maxlen':1, 'type':'con'},
                   succ ={'attributes':su, 'minlen':1, 'maxlen':1 , 'type':'con'}
                              )
        else:
            clm.mine(proc='4ftMiner', quantifiers={'Base': ara.min_base},
                              ante={'attributes': an, 'minlen': 1, 'maxlen': 1, 'type': 'con'},
                              succ={'attributes': su, 'minlen': 1, 'maxlen': 1, 'type': 'con'},
                              cond = cond
                              )


        ara.stats["clm_runs"] += 1
        for i in range(clm.get_rulecount()):
            rule_id = i+1
            fft = clm.get_fourfold(rule_id)
            lift = clm.get_quantifiers(rule_id)['aad']+1
            cumlift=lift_pre*lift
            ante_str = ''
            valid = 0
            disp_str=""
            for i2 in range(len(ara.boundaries)):
                if lift > 1:
                    if lift>=ara.boundaries[i2]:
                        valid = i2 + 1
                        disp_str += "+"
                    else:
                        disp_str += "."
                elif lift < 1:
                    if lift <= 1/ara.boundaries[i2]:
                        valid = i2 - 1
                        disp_str += "-"
                    else:
                        disp_str += "."
                else:
                    disp_str="." * len(ara.boundaries)
            if not(valid==0):
                ara.stats["resulting_rules"] += 1
                ante_str = clm.result['rules'][i]['cedents_str']['ante']
                if ara.print_rules_on_the_fly==1:
                    if valid>0:
                        print(f"{prepend_str}{disp_str} {str(ante_str)} x{lift:.1f}")
                    else:
                        print(f"{prepend_str}{disp_str} {str(ante_str)} /{1/lift:.1f}")
                cs = clm.result['rules'][i]['cedents_struct']
                cl = cs['ante']
                cl.update(cs['cond'])

                cs2 = clm.result['rules'][i]['trace_cedent_dataorder']
                cl2 = cs2['ante'] + cs2['cond']
                cs2b = clm.result['rules'][i]['traces']
                vals2 = cs2b['ante'] + cs2b['cond']
                newcond=[]
                for i in range(len(cl2)):
                    ca = {}
                    ca['name'] = clm.result['datalabels']['varname'][cl2[i]]
                    ca['type'] = 'one'
                    ca['value'] = clm.result['datalabels']['catnames'][cl2[i]][vals2[i][0]]
                    newcond.append(ca)
                cond_d = {}
                cond_d['attributes'] = newcond
                cond_d['minlen'] = len(cl.items())
                cond_d['maxlen'] = len(cl.items())
                cond_d['type'] = 'con'
                subres=[]
                if depth<ara.max_depth:
                    res_s=ara.arap(df,target=target,target_class=target_class,cond=cond_d,clm=clm,prepend_str='   ',depth=depth+1,lift_pre=cumlift)
                    subres = res_s
                res_l={}
                vars=[]
                for i in range(len(newcond)):
                    vr={}
                    vr['varname']=newcond[i]['name']
                    vr['value']=newcond[i]['value']
                    vars.append(vr)
                res_l['vars'] = vars
                res_l['fft'] = fft
                res_l['lift'] = lift
                res_l['cumlift'] = cumlift
                res_l["target_class_ratio"] = fft[0]/(fft[0]+fft[1])
                if lift > 1:
                    res_l['booster'] ='x' + "{:.1f}".format(lift)
                else:
                    res_l['booster'] ='/' + "{:.1f}".format(1/lift)
                res_l['valid_level'] = valid
                res_l['valid_level_disp_string'] = disp_str
                res_l['sub'] = subres
                res_ara.append(res_l)
        if cond is not None:
            return res_ara
        profile = df.groupby([target])[target].count().to_dict()
        summ=0
        for k,v in profile.items():
            summ+=v
        tgt_ratio=profile[target_class]/summ
        res_total={}
        task_info={}
        task_info["target"]=target
        task_info["target_class"]=target_class
        opts={}
        opts["min_base"]=ara.min_base
        opts["max_depth"]=ara.max_depth
        task_info["opts"]=opts
        res_total["task_info"]=task_info
        ara.end_time = time.time()
        ara.stats["time_sec"] = ara.end_time-ara.start_time
        res_total["stats"]=ara.stats
        res_aa={}
        res_aa["target_var_profile"] = profile
        res_aa["target_class_ratio"] = tgt_ratio
        res_aa["rules"] = res_ara
        res_total["results"]=res_aa
        return res_total

    @staticmethod
    def print_result(res=None,pre="",mult=1):
        for item in res:
            total_lift = mult*item['lift']
            if total_lift>=1:
                total_lift_str="x" + "{:.1f}".format(total_lift)
            else:
                total_lift_str="/"+ "{:.1f}".format(1/total_lift)
            print(f"{pre}{item['valid_level_disp_string']} {item['vars'][0]['varname']}({item['vars'][0]['value']}) {item['booster']} (={total_lift_str})")
            if len(item['sub'])>0:
                ara.print_result(res=item['sub'],pre=pre+"    ",mult=total_lift)



