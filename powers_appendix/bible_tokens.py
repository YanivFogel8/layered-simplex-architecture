"""Export the canonical Table 2 stream; never re-tokenize a different raw Bible."""
import argparse
from pathlib import Path
import numpy as np
from protocol import load_tokens

def tokens():return load_tokens()
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=Path('results_main/bible_tokens.npy'));a=ap.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
 t=tokens();np.save(a.out,t);print(f'{len(t)} tokens, {len(np.unique(t))} types -> {a.out}')
