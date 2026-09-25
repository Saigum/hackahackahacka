import csv, re, random, time, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from unidecode import unidecode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'dataset/student_resource/dataset'
OUT = ROOT / 'work'
WORD = re.compile('[a-z0-9]+')
# Text normalization only; no outside business data.
ALIASES = dict(corporation='corp', incorporated='inc', company='co', limited='ltd',
              private='pvt', road='rd', street='st', avenue='ave', boulevard='blvd',
              drive='dr', lane='ln', highway='hwy', suite='ste', apartment='apt',
              floor='fl', building='bldg', centre='center')

def normalize(s):
    s = unidecode(s).lower().replace('&', ' and ')
    return ' '.join(ALIASES.get(w,w) for w in WORD.findall(s))

def prepare(job):
    split, source = job
    p = DATA / split / f'{split}_source{source}.tsv'
    out = OUT / f'{split}{source}.tsv'
    if out.exists(): return str(out) + ' exists'
    start=time.time()
    with p.open() as f, out.with_suffix('.tmp').open('w') as g:
        for r in csv.DictReader(f, delimiter='\t'):
            n, a = r['business_name'], r['business_address']
            g.write('\t'.join((r['entity_id'], normalize(n), normalize(a), r['country'],
                              str(int(not n.isascii())), str(int(not a.isascii()))))+'\n')
    out.with_suffix('.tmp').rename(out)
    return f'{out} done {time.time()-start:.1f}s'

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(prepare, [(s,i) for s in ('train','test') for i in (1,2,3)]):
            print(result, flush=True)
    # Deterministic entity-disjoint train / calibration / final holdout queries.
    rng=random.Random(61423)
    sample=[]
    with (OUT/'train1.tsv').open() as f:
        for i,line in enumerate(f):
            if len(sample)<50000: sample.append(line)
            else:
                j=rng.randrange(i+1)
                if j<len(sample): sample[j]=line
    rng.shuffle(sample)
    (OUT/'queries.tsv').write_text(''.join(sample))
    ids={line.split('\t')[0] for line in sample}
    with (DATA/'train/train_ground_truth.tsv').open() as f:
        truth={r['source1_entity_id']:r['matched_entity_ids'].split(',') if r['matched_entity_ids'] else []
               for r in csv.DictReader(f,delimiter='\t') if r['source1_entity_id'] in ids}
    (OUT/'truth.json').write_text(json.dumps(truth))
