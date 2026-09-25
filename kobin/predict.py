import time,json,os
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train import DT
ROOT=Path(__file__).resolve().parents[1]; W=ROOT/'work'; O=ROOT/'output'
if __name__=='__main__':
    while not (W/'test_pairs.targets').exists() or not (W/'test_pairs.queries').exists():time.sleep(2)
    # Files finish before the first atomic feature chunk appears.
    while not list(W.glob('test_pairs.*.bin')): time.sleep(2)
    targets=(W/'test_pairs.targets').read_text().splitlines()
    queries=[s.split('\t')[0] for s in (W/'test_pairs.queries').read_text().splitlines()]
    while not (W/'final_model.txt').exists():time.sleep(2)
    model=lgb.Booster(model_file=str(W/'final_model.txt'))
    threshold=json.loads((W/'metrics.json').read_text())['threshold']
    total=(len(queries)+2499)//2500
    done={b for b in range(total) if (W/f'part.{b}.matches').exists() and (W/f'part.{b}.candidates').exists() and (W/f'test_scores.{b}.bin').exists() and not (W/f'test_pairs.{b}.bin').exists()}
    start=time.time()
    print('Resuming with',len(done),'completed batches',flush=True)
    while len(done)<total:
        found=False
        for b in range(total):
            path=W/f'test_pairs.{b}.bin'
            if b in done or not path.exists():continue
            found=True
            rows=np.fromfile(path,dtype=DT)
            # The first 52 batches were scored before the validated speed optimization.
            p=model.predict(np.ascontiguousarray(rows['x']),num_threads=4,
                pred_early_stop=(b>=52),pred_early_stop_margin=12)
            # Retain small score arrays so a later threshold change need not rerun retrieval.
            scores=np.empty(len(rows),dtype=[('q','<u4'),('t','<u4'),('p','<f4')])
            scores['q']=rows['q'];scores['t']=rows['t'];scores['p']=p
            scores.tofile(W/f'test_scores.{b}.bin')
            lo=b*2500; hi=min(len(queries),lo+2500)
            starts=np.searchsorted(rows['q'],np.arange(lo,hi+1))
            with (W/f'part.{b}.matches').open('w') as fm,(W/f'part.{b}.candidates').open('w') as fc:
                for i,qi in enumerate(range(lo,hi)):
                    s,e=starts[i:i+2]; tids=rows['t'][s:e];probs=p[s:e]
                    fc.write(queries[qi]+'\t'+','.join(targets[t] for t in tids)+'\n')
                    fm.write(queries[qi]+'\t'+','.join(targets[t] for t in tids[probs>=threshold])+'\n')
            done.add(b)
            path.unlink() # Features are reproducible and would otherwise consume >15 GB.
            print(f'scored {len(done)}/{total} batches, {time.time()-start:.1f}s',flush=True)
        if not found:time.sleep(2)
    import shutil
    for suffix,name,header in [('matches','matching_results.tsv','source1_entity_id\tmatched_entity_ids\n'),('candidates','candidate_pairs.tsv','source1_entity_id\tcandidate_entity_ids\n')]:
        tmp=O/(name+'.tmp')
        with tmp.open('w') as f:
            f.write(header)
            for b in range(total):
                with (W/f'part.{b}.{suffix}').open() as g:shutil.copyfileobj(g,f)
        tmp.rename(O/name)
    print('OUTPUTS READY',flush=True)
