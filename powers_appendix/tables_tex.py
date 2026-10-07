"""Render computed main-protocol tables; no transcribed baseline columns."""
import argparse,json
from pathlib import Path

def number(v):return f'{v:.5g}' if isinstance(v,float) else str(v)
def escape(s):return str(s).replace('_',r'\_').replace('%',r'\%')
def write(path,caption,headers,rows):
 s='\n'.join([r'\begin{table*}[t]',r'\centering\scriptsize',r'\caption{'+caption+'}',r'\resizebox{\linewidth}{!}{%',r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',' & '.join(map(escape,headers))+r' \\',r'\midrule',*[' & '.join(row)+r' \\' for row in rows],r'\bottomrule',r'\end{tabular}}',r'\end{table*}'])
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(s)
def render(root):
 root=Path(root)
 bench=json.loads((root/'bench/table.json').read_text());headers=['Target',*bench['rows'][0]['statistics']];rows=[]
 for r in bench['rows']:
  cells=[escape(r['target'])]
  for v in r['statistics'].values():cells.append('$'+number(v['mean'])+r'\pm '+number(v['SE'])+'$' if v['status']=='finite' else (r'$\infty$' if v['status']=='infinite' else 'n.a.'))
  rows.append(cells)
 write(root/'report/powers_benchmark.tex',r'Mean normalized KL loss $\pm$ one SE (bits), on the exact 20 paired trials of main Table~1. Dirichlet targets are fixed. Every added mixture includes the uniform component; depth mixtures use $L=0,\ldots,80$. Main-paper columns are retained from the saved trials. Individual SEs do not measure uncertainty in paired differences.',headers,rows)
 bible=json.loads((root/'bible/table.json').read_text());headers=list(bible['rows'][0]);rows=[[escape(number(r[k])) for k in headers] for r in bible['rows']]
 write(root/'report/powers_bible.tex',r'Redundancy in bits per token on the exact Table~2 stream (915,860 tokens, 13,550 types; $d=10^5$). Every depth mixture includes $L=0,\ldots,54$; fixed-power mixtures include power zero. Main-paper reference columns use the same stream. Numerical differences between engines must be checked before interpreting small gaps.',headers,rows)
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('results_main'));a=ap.parse_args();render(a.out)
