"""Current main-paper inputs and provenance guards (no numerical kernels)."""
from pathlib import Path
import hashlib,json,os
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
TRIALS=DATA/'main_table1_trials.npz'
TOKENS=DATA/'main_bible_tokens.npz'
ROWS=[('uniform','uniform'),('step','step'),('Zipf 1','zipf1'),('Zipf 1.5','zipf1.5'),('Zipf 2','zipf2'),('Zipf 3','zipf3'),('Zipf 4','zipf4'),('Zipf 5','zipf5'),('geometric','geom'),('Dirichlet-1','dir1'),('Dirichlet-1/2','dir0.5')]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_tokens(path=TOKENS):
 with np.load(path,allow_pickle=False) as a:t=a['tokens'].copy()
 if t.ndim!=1 or len(t)!=915860 or len(np.unique(t))!=13550:raise ValueError('Not the canonical Table 2 token stream')
 expected=json.loads((DATA/'provenance.json').read_text())['token_ids_sha256']
 if hashlib.sha256(t.astype('<i4').tobytes()).hexdigest()!=expected:raise ValueError('Table 2 token order/hash mismatch')
 return t

def guard(out,tag,inputs,settings):
 """Refuse stale checkpoints, even when array shapes happen to agree."""
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 manifest={'protocol':'main-paper-L0-v1','inputs':{str(k):sha(v) for k,v in inputs.items()},'settings':settings,
           'sources':{p.name:sha(p) for p in sorted(HERE.glob('*.py'))}}
 dest=out/(tag+'.provenance.json')
 if dest.exists():
  if json.loads(dest.read_text())!=manifest:raise ValueError(f'Incompatible cached run: {dest}; choose a new output directory')
 elif (out/(tag+'.pkl')).exists() or (out/(tag+'_state.npz')).exists():
  raise ValueError(f'Unverified historical cache for {tag}; choose a new output directory')
 tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,dest)
 return manifest

def normalized_loss(trial,q,d=10000):
 n,r,c,S,plp=trial;q=np.asarray(q,float)
 multiplicity=np.r_[d-int(c.sum()),c]
 total=multiplicity@q
 if not np.isfinite(q).all() or np.any(q<0) or not np.isfinite(total) or total<=0:raise ValueError('Invalid predictive vector')
 q=q/total
 if np.any(q[S>0]==0):return float('inf')
 return float((plp-S@np.log(np.where(S>0,q,1.)))/np.log(2))
