"""Resume a timed study between saved complete paths, without changing its design.

The interruption is recorded. An interrupted, unsaved path is restarted from
scratch; its partial elapsed time is never substituted for a complete run.
"""
import hashlib,json
from datetime import datetime,timezone
from flint import ctx
from benchmark_strong_refinement import ROOT
from residual_refinement_design import install
from strong_refinement_search import search

def main():
    path=ROOT/'results/strong_refinement.json';r=json.loads(path.read_text());assert r['status']=='running'
    for n,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/'code'/n).read_bytes()).hexdigest()==sha
    r['source_sha256'][Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    r.setdefault('execution_segments',[]).append(dict(resumed_at_utc=datetime.now(timezone.utc).isoformat(),
        completed_paths=len(r['records']),reason='Paused between saved paths to develop new-geometry experiments without concurrent numerical work. Any unsaved partial path is discarded and restarted.'))
    design=r['design'];ctx.prec=192;install();index=0
    def save():path.write_text(json.dumps(r,indent=2)+'\n')
    save()
    for rep in range(design['repetitions']):
     for case in design['cases']:
      for n in design['grids']:
       for kind in (design['kinds'] if rep%2==0 else design['kinds'][::-1]):
        if index<len(r['records']):
            old=r['records'][index];assert (old['repetition'],old['case'],old['n'],old['kind'])==(rep,case,n,kind)
        else:
            row=search(n,case,kind);row['repetition']=rep;r['records'].append(row);save()
            print(rep,case,n,kind,row['selected']['certificate']['input_error_upper']['decimal'],round(row['total_seconds'],2),flush=True)
        index+=1
    r['status']='completed';save();path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
if __name__=='__main__':
    from pathlib import Path
    main()
