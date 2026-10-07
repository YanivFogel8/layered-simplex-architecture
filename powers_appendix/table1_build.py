"""Extend exact main Table 1 trials with normalized powered-model predictions."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from protocol import TRIALS,ROWS,sha,normalized_loss
from table1_trials import build

def load_result(folder,name):
 p=Path(folder)/(name+'.pkl');m=p.with_suffix('.provenance.json')
 if not m.exists():raise ValueError(f'Unverified result: {p}')
 manifest=json.loads(m.read_text())
 if manifest['inputs'].get('trials')!=sha(Path(folder)/'trials.pkl'):raise ValueError('Result/trials mismatch')
 with p.open('rb') as f:return pickle.load(f)
def summary(v):
 v=np.asarray(v,float)
 if np.isnan(v).any():return {'mean':None,'SE':None,'status':'undefined'}
 if np.isinf(v).any():return {'mean':None,'SE':None,'status':'infinite'}
 return {'mean':float(v.mean()),'SE':float(v.std(ddof=1)/np.sqrt(len(v))),'status':'finite'}
def mixture(trial,parts):
 lq=np.array([x[0] for x in parts]);q=np.array([x[1] for x in parts])
 if not np.isfinite(lq).all():raise ValueError('Incomplete/nonfinite evidence')
 return normalized_loss(trial,np.exp(lq-logsumexp(lq))@q)
def build_table(folder):
 folder=Path(folder);tr=build()
 with (folder/'trials.pkl').open('rb') as f:actual=pickle.load(f)
 if set(actual)!=set(tr):raise ValueError('Expected all 220 main Table 1 trials')
 for k in tr:
  if not all(np.array_equal(x,y) for x,y in zip(tr[k],actual[k])):raise ValueError('Not main Table 1 trials')
 dep={tag:load_result(folder,tag) for tag in ['delta1','exp1','unif2','unif1']}
 pw=[load_result(folder,f'pow_w{k}') for k in range(1,81)]
 inv=[load_result(folder,f'pow_w1_{k}') for k in range(2,81)]
 results=[];unit_differences=[]
 with np.load(TRIALS,allow_pickle=False) as main:
  methods=list(main['methods']);baseline=['add-1','KT','Ristad','GT','Dir-tau','AD','L=22','LSA avg','oracle']
  for ti,(label,key) in enumerate(ROWS):
   vals={m:main[key+'__normalized'][:,methods.index(m)].tolist() for m in baseline}
   for name in ['powers 0-80','powers with reciprocals','Exp(1) 0-80','U[0,2] 0-80','U[0,1] 0-80']:vals[name]=[]
   for t in range(20):
    k=(ti,0,t);trial=tr[k];n,r,*_=trial;zero=(-n*np.log(10000),np.full(len(r)+1,1/10000))
    def parts(R):return [(R['logq'][k+(L,)],R['q'][k+(L,)]) for L in range(1,81)]
    unit_differences.append(mixture(trial,[zero]+parts(dep['delta1']))-vals['LSA avg'][t])
    positive=[(P['logq'][k+(1,)],P['q'][k+(1,)]) for P in pw];reciprocal=[(P['logq'][k+(1,)],P['q'][k+(1,)]) for P in inv]
    vals['powers 0-80'].append(mixture(trial,[zero]+positive))
    vals['powers with reciprocals'].append(mixture(trial,[zero]+positive+reciprocal))
    for tag,name in [('exp1','Exp(1) 0-80'),('unif2','U[0,2] 0-80'),('unif1','U[0,1] 0-80')]:vals[name].append(mixture(trial,[zero]+parts(dep[tag])))
   results.append({'target':label,'statistics':{k:summary(v) for k,v in vals.items()},'paired_vs_LSA':{k:summary(np.asarray(v)-vals['LSA avg']) for k,v in vals.items() if k!='LSA avg'},'trials':{k:[float(x) if np.isfinite(x) else ('infinite' if np.isinf(x) else 'undefined') for x in v] for k,v in vals.items()}})
 return {'protocol':'main-paper-L0-v1','n':1000,'d':10000,'trials':20,'main_table1_sha256':sha(TRIALS),'rows':results,'max_unit_engine_difference_bits':float(np.max(np.abs(unit_differences))),'note':'Main-paper columns are retained exactly from archived paired trials; added columns use the powers engine. Check unit-engine diagnostic before interpreting small differences.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('results_main/bench'));a=ap.parse_args();r=build_table(a.out);(a.out/'table.json').write_text(json.dumps(r,indent=2,allow_nan=False));print('Saved',a.out/'table.json','unit-engine discrepancy:',r['max_unit_engine_difference_bits'])
