import json, time
from pathlib import Path
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]; W=ROOT/'work'
DT=np.dtype([('q','<u4'),('t','<u4'),('x','<f4',(44,))])

def metric(q,y,p,counts,threshold,lo,hi):
    take=p>=threshold
    pred=np.bincount(q[take],minlength=len(counts))
    tp=np.bincount(q[take],weights=y[take],minlength=len(counts))
    den=.25*counts+pred
    score=np.divide(1.25*tp,den,out=np.zeros(len(counts)),where=den>0)
    score[(counts==0)&(pred==0)]=1
    return score[lo:hi], pred, tp

if __name__=='__main__':
    truth=json.loads((W/'truth.json').read_text())
    queries=[s.split('\t') for s in (W/'train_pairs.queries').read_text().splitlines()]
    target_ids=(W/'train_pairs.targets').read_text().splitlines()
    chunks=[np.fromfile(W/f'train_pairs.{i}.bin',dtype=DT) for i in range(20)]
    rows=np.concatenate(chunks); del chunks
    q=rows['q'].copy(); x=np.ascontiguousarray(rows['x']); ti=rows['t'].copy();del rows
    actual=[set(truth[s[0]]) for s in queries]
    y=np.fromiter((target_ids[t] in actual[i] for i,t in zip(q,ti)),dtype=np.int8,count=len(q))
    counts=np.array([len(s) for s in actual]); countries=np.array([s[1] for s in queries]);del target_ids
    print('pairs',len(y),'positives',y.sum(),'candidate_recall',y.sum()/counts.sum(),flush=True)
    report={'candidate_recall':float(y.sum()/counts.sum()),'groups':{}}
    for c in sorted(set(countries)):
        mask=countries==c
        report['groups'][c]={'candidate_recall':float(y[mask[q]].sum()/counts[mask].sum())}
    train=q<35000; calib=(q>=35000)&(q<42500); hold=q>=42500
    model=lgb.LGBMClassifier(n_estimators=700,learning_rate=.055,num_leaves=63,max_depth=-1,
        min_child_samples=80,colsample_bytree=.9,reg_lambda=3,n_jobs=12,verbosity=-1,random_state=714)
    model.fit(x[train],y[train],eval_set=[(x[calib],y[calib])],callbacks=[lgb.early_stopping(45),lgb.log_evaluation(50)])
    p=model.predict_proba(x,num_threads=12)[:,1]
    results=[]
    for th in np.arange(.15,.951,.01):
        scores,_,_=metric(q,y,p,counts,th,35000,42500);results.append((float(scores.mean()),float(th)))
    score,th=max(results)
    scores,pred,tp=metric(q,y,p,counts,th,0,50000)
    report.update(threshold=th,calibration_macro_f05=score,holdout_macro_f05=float(scores[42500:].mean()),best_iteration=model.best_iteration_,holdout_entities=7500)
    for c in sorted(set(countries)):
        m=(countries==c)&(np.arange(50000)>=42500)
        report['groups'][c]['holdout_macro_f05']=float(scores[m].mean())
    report['holdout_singleton_accuracy']=float(scores[(counts==0)&(np.arange(50000)>=42500)].mean())
    print(json.dumps(report,indent=2),flush=True)
    (W/'metrics.json').write_text(json.dumps(report,indent=2))
    model.booster_.save_model(str(W/'model.txt'))
    np.savez_compressed(W/'validation.npz',q=q,t=ti,y=y,p=p,counts=counts,scores=scores)
    # Refit using training + calibration entities, retaining the untouched holdout.
    final=lgb.LGBMClassifier(n_estimators=model.best_iteration_,learning_rate=.055,num_leaves=63,
       min_child_samples=80,colsample_bytree=.9,reg_lambda=3,n_jobs=12,verbosity=-1,random_state=714)
    final.fit(x[~hold],y[~hold]); final.booster_.save_model(str(W/'final_model.txt'))
    p2=final.predict_proba(x[hold],num_threads=12)[:,1]
    s2,_,_=metric(q[hold],y[hold],p2,counts,th,42500,50000)
    report['refit_holdout_macro_f05']=float(s2.mean())
    (W/'metrics.json').write_text(json.dumps(report,indent=2));print('refit holdout',s2.mean(),flush=True)
