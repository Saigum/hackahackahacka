import json,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from train import DT,metric
W=Path('work');z=np.load(W/'validation.npz');q=z['q'];y=z['y'];counts=z['counts'];mask=q>=35000
rows=np.concatenate([np.fromfile(W/f'train_pairs.{i}.bin',dtype=DT) for i in range(14,20)])
x=np.ascontiguousarray(rows['x']);del rows
model=lgb.Booster(model_file=str(W/'final_model.txt'));start=time.time();exact=model.predict(x,num_threads=4);print('exact_seconds',time.time()-start,flush=True)
report={}
for margin in [5.,7.,10.]:
 start=time.time();p=model.predict(x,num_threads=4,pred_early_stop=True,pred_early_stop_freq=10,pred_early_stop_margin=margin)
 s,_,_=metric(q[mask],y[mask],p,counts,.72,35000,50000)
 report[str(margin)]={'seconds':time.time()-start,'decision_changes':int(((exact>=.72)!=(p>=.72)).sum()),'calib':float(s[:7500].mean()),'holdout':float(s[7500:].mean())}
 print(margin,report[str(margin)],flush=True)
(W/'fast_predict_check.json').write_text(json.dumps(report,indent=2))
