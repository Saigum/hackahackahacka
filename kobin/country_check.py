from pathlib import Path
import json
import numpy as np
import lightgbm as lgb
from train import DT,metric
W=Path('work')
z=np.load(W/'validation.npz');q=z['q'];y=z['y'];counts=z['counts'];
countries=np.array([l.split('\t')[1] for l in (W/'train_pairs.queries').read_text().splitlines()])
rows=np.concatenate([np.fromfile(W/f'train_pairs.{b}.bin',dtype=DT) for b in range(20)])
x=np.ascontiguousarray(rows['x']);del rows
out={}
for train_country,test_country in [('US','India'),('India','US')]:
    train=(q<35000)&(countries[q]==train_country)
    val=(q>=35000)&(q<42500)
    model=lgb.LGBMClassifier(n_estimators=400,learning_rate=.065,num_leaves=63,min_child_samples=60,
      colsample_bytree=.9,reg_lambda=3,n_jobs=4,verbosity=-1,random_state=815)
    model.fit(x[train],y[train]);p=model.predict_proba(x[val],num_threads=4)[:,1]
    s,_,_=metric(q[val],y[val],p,counts,.72,0,50000)
    m=(np.arange(50000)>=35000)&(np.arange(50000)<42500)&(countries==test_country)
    out[f'{train_country}_to_{test_country}']=float(s[m].mean())
    print(out,flush=True)
(W/'country_transfer.json').write_text(json.dumps(out,indent=2))
