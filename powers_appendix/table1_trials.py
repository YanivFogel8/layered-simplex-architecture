"""Convert the exact main Table 1 trials to the powers evaluator's profiles."""
import argparse,pickle
from pathlib import Path
import numpy as np
from profiles import profile_of
from protocol import TRIALS,ROWS as MAIN_ROWS,sha
ROWS=[(name,key,i) for i,(name,key) in enumerate(MAIN_ROWS)]
def build(archive=TRIALS):
 out={}
 with np.load(archive,allow_pickle=False) as a:
  for ti,(_,key) in enumerate(MAIN_ROWS):
   p=a[key+'__p'];counts=a[key+'__counts']
   if p.shape!=(10000,) or counts.shape!=(20,10000):raise ValueError('Expected main Table 1 shape')
   for t,c in enumerate(counts):
    if c.sum()!=1000:raise ValueError('Expected n=1000')
    r,cc=profile_of(c);S=np.array([p[c==0].sum()]+[p[c==v].sum() for v in r])
    out[(ti,0,t)]=(1000,r,cc,S,float(np.sum(p[p>0]*np.log(p[p>0]))))
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path,default=TRIALS);ap.add_argument('--out',type=Path,default=Path('results_main/bench/trials.pkl'));a=ap.parse_args()
 payload=pickle.dumps(build(a.archive),protocol=4);a.out.parent.mkdir(parents=True,exist_ok=True)
 if a.out.exists() and a.out.read_bytes()!=payload:raise SystemExit('Refusing to replace different trials; use a new output directory')
 a.out.write_bytes(payload);print('Prepared 220 main-paper trials; input SHA256',sha(a.archive))
