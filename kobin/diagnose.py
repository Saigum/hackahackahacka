import json,sys
from pathlib import Path
import numpy as np
from train import DT
W=Path('work');truth=json.loads((W/'truth.json').read_text());qs=[l.split('\t') for l in (W/'queries.tsv').read_text().splitlines()];tids=(W/'train_pairs.targets').read_text().splitlines()
found=[set() for q in qs]
for b in range(20):
 rows=np.fromfile(W/f'train_pairs.{b}.bin',dtype=DT)
 for q,t in zip(rows['q'],rows['t']):found[q].add(tids[t])
summary={}
missing={}
for i,q in enumerate(qs):
 c=q[3];actual=set(truth[q[0]]);m=actual-found[i]
 s=summary.setdefault(c,[0,0]);s[0]+=len(actual);s[1]+=len(m)
 if i<35000:
  for t in m:missing[t]=i
print(summary,flush=True)
selected={t:i for t,i in list(missing.items())[:150]}
for src in (2,3):
 with (W/f'train{src}.tsv').open() as f:
  for line in f:
   t=line.rstrip('\n').split('\t')
   if t[0] in selected:
    q=qs[selected[t[0]]]
    print(json.dumps({'q':q,'t':t},ensure_ascii=False),flush=True)
