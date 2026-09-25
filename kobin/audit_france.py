from pathlib import Path
import csv,json
W=Path('work'); D=Path('dataset/student_resource/dataset/test')
qs=[l.split('\t') for l in (W/'test_pairs.queries').read_text().splitlines()]
samples=[]
for b in range(10):
 with (W/f'part.{b}.matches').open() as f:
  for j,l in enumerate(f):
   q,ids=l.rstrip('\n').split('\t')
   if qs[b*2500+j][1]=='France' and ids:
    samples.append((q,ids.split(',')[0]))
    if len(samples)>=12:break
 if len(samples)>=12:break
wanted={x for p in samples for x in p}; rec={}
for src in (1,2,3):
 with (D/f'test_source{src}.tsv').open() as f:
  for r in csv.DictReader(f,delimiter='\t'):
   if r['entity_id'] in wanted:rec[r['entity_id']]=r
for q,t in samples:print(json.dumps([rec[q],rec[t]],ensure_ascii=False))
