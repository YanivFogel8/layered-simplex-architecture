"""Extend canonical Table 2 with powered mixtures, using depths 0..54."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from protocol import DATA,TOKENS,sha,load_tokens

def build_table(folder):
 folder=Path(folder);ref=json.loads((DATA/'main_table2_reference.json').read_text());ns=np.array([r['n'] for r in ref]);H=np.array([r['H'] for r in ref]);dist=[r['distinct'] for r in ref]
 load_tokens() # verifies the exact canonical stream
 def load(name):
  p=folder/(name+'.pkl');m=p.with_suffix('.provenance.json')
  if not m.exists():raise ValueError(f'Unverified result {p}')
  manifest=json.loads(m.read_text())
  if manifest['inputs'].get('tokens')!=sha(TOKENS):raise ValueError('Corpus mismatch')
  with p.open('rb') as f:r=pickle.load(f)
  if list(r['ckpts'])!=list(ns) or list(r['distinct'])!=dist:raise ValueError('Checkpoint mismatch')
  np.testing.assert_allclose(r['H'],H,atol=1e-12,rtol=0)
  if not np.isfinite(r['logq']).all():raise ValueError('Incomplete/nonfinite results')
  return r
 def red(q):return -(logsumexp(q,axis=1)-np.log(q.shape[1]))/np.log(2)/ns-H
 zero=(-ns*np.log(100000))[:,None];cols={};unit=None
 for tag,label in [('delta1','unit engine 0-54'),('exp1','Exp(1) 0-54'),('unif2','U[0,2] 0-54'),('unif1','U[0,1] 0-54')]:
  r=load(tag)
  if r['ladder']!=list(range(1,55)):raise ValueError('Expected positive components 1..54')
  value=red(np.column_stack([zero,r['logq']]))
  if tag=='delta1':unit=value
  else:cols[label]=value
 labels=[f'w{k}' for k in range(1,28)]+[f'w1_{k}' for k in range(2,28)]
 power=np.array([load('bible_pow_'+label)['logq'] for label in labels]).T
 cols['powers 0-27']=red(np.column_stack([zero,power[:,:27]]));cols['powers with reciprocals']=red(np.column_stack([zero,power]))
 rows=[dict(r,**{k:float(v[i]) for k,v in cols.items()}) for i,r in enumerate(ref)]
 return {'protocol':'main-paper-L0-v1','token_archive_sha256':sha(TOKENS),'d':100000,'depths':[0,54],'rows':rows,'max_unit_engine_difference_bits_per_token':float(np.max(np.abs(unit-np.array([r['LSA 0-54'] for r in ref])))),'note':'Main Table 2 reference columns share the canonical stream; AD omitted because its unregularized sequential code is nonfinite. Unit-engine discrepancy is a numerical diagnostic, not a sampling SE.'}
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('results_main/bible'));a=ap.parse_args();r=build_table(a.out);(a.out/'table.json').write_text(json.dumps(r,indent=2,allow_nan=False));print('Saved',a.out/'table.json','unit-engine discrepancy:',r['max_unit_engine_difference_bits_per_token'])
