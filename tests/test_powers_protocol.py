"""Main-paper inputs, normalization, provenance, and independent w=1 check."""
import sys,json
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'powers_appendix'))
from protocol import load_tokens,TRIALS,ROWS,guard,normalized_loss
from table1_trials import build

def test_exact_main_profiles_and_targets():
 trials=build();assert len(trials)==220
 with np.load(TRIALS) as a:
  for ti,(_,key) in enumerate(ROWS):
   for t in range(20):
    n,r,c,S,plp=trials[ti,0,t];counts=a[key+'__counts'][t];p=a[key+'__p']
    assert n==counts.sum()==1000
    assert np.dot(r,c)==n
    np.testing.assert_allclose(S.sum(),1,atol=1e-14)
    # Add-one scored by class must match the saved per-symbol implementation.
    q=np.r_[1,r+1]/11000
    got=normalized_loss(trials[ti,0,t],q)
    want=a[key+'__normalized'][t,list(a['methods']).index('add-1')]
    assert abs(got-want)<1e-12

def test_canonical_bible_prefixes():
 t=load_tokens();ref=json.loads((ROOT/'powers_appendix/data/main_table2_reference.json').read_text())
 for r in ref:
  c=np.bincount(t[:r['n']]);p=c[c>0]/r['n']
  assert len(p)==r['distinct']
  assert abs(-(p*np.log2(p)).sum()-r['H'])<1e-12
 assert len(t)==915860

def test_normalization_and_nonfinite_loss():
 trial=(2,np.array([2]),np.array([1]),np.array([.5,.5]),-np.log(2))
 assert normalized_loss(trial,[.2,.8],d=2)==pytest.approx(normalized_loss(trial,[2.,8.],d=2))
 assert np.isinf(normalized_loss(trial,[0,1],d=2))
 with pytest.raises(ValueError):normalized_loss(trial,[np.nan,1],d=2)

def test_cache_rejects_changed_inputs_settings_and_legacy(tmp_path):
 p=tmp_path/'input';p.write_text('a');guard(tmp_path,'model',{'input':p},{'Lmax':54})
 guard(tmp_path,'model',{'input':p},{'Lmax':54})
 with pytest.raises(ValueError):guard(tmp_path,'model',{'input':p},{'Lmax':40})
 p.write_text('b')
 with pytest.raises(ValueError):guard(tmp_path,'model',{'input':p},{'Lmax':54})
 (tmp_path/'legacy.pkl').write_bytes(b'not loaded')
 with pytest.raises(ValueError):guard(tmp_path,'legacy',{'input':p},{})

def test_uniform_component_and_posterior_evidence_weights():
 from table1_build import mixture
 trial=(2,np.array([2]),np.array([1]),np.array([.5,.5]),-np.log(2))
 zero=(-2*np.log(10000),np.full(2,.0001));positive=(zero[0]+np.log(3),np.array([.00005,.50005]))
 want=normalized_loss(trial,.25*zero[1]+.75*positive[1])
 assert mixture(trial,[zero,positive])==pytest.approx(want)

def test_single_power_one_matches_independent_dirichlet_formula():
 from scipy.special import gammaln
 from power_table import ht_rows
 from modelb_lib import Grid,untilt,mixture_predictive
 grid=Grid(-30,40,.025);rows=np.arange(5)
 ht=ht_rows(rows,grid.u,1.)
 lq,preds,_=mixture_predictive(5,10,np.array([2,3]),np.array([1,1]),untilt(ht,rows,grid),{int(r):i for i,r in enumerate(rows)},grid,want_ratios=True,spline_pad=60)
 exact=gammaln(10)-gammaln(15)+gammaln(3)+gammaln(4)
 assert abs(lq-exact)<1e-7
 for r in [0,2,3]:assert abs(preds[r]-(r+1)/15)<1e-8

def test_bible_rejects_wrong_stream(tmp_path):
 p=tmp_path/'old.npz';np.savez_compressed(p,tokens=np.zeros(915849,dtype=np.int32))
 with pytest.raises(ValueError):load_tokens(p)
