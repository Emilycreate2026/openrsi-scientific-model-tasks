#!/usr/bin/env python3
"""Author-side reproducibility CLI. This checkout is not an agent workspace."""
from pathlib import Path
import argparse, hashlib, json, math, os, shutil, subprocess, sys, tempfile, time

ROOT = Path(__file__).resolve().parent
KEYS = ('catalyst', 'treatment', 'alloy')

def verify_assets():
    for name, digest in json.loads((ROOT/'ASSET_SHA256.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('asset integrity mismatch: '+name)

def materialize(key, dest):
    """Solution path: copy a frozen evaluation-ready artifact; never train or score."""
    verify_assets()
    if dest.exists(): raise FileExistsError(dest)
    shutil.copytree(ROOT/'cases'/key/'baseline/artifact', dest)

def prepare_work(key, dest):
    verify_assets()
    if dest.exists(): raise FileExistsError(dest)
    case = ROOT/'cases'/key
    dest.mkdir(parents=True)
    for src, name in [(case/'inputs','data'), (case/'baseline/source','candidate_source'),
                      (case/'baseline/artifact','reference_artifact')]:
        shutil.copytree(src, dest/name)
    if key == 'catalyst': shutil.copytree(case/'simulator/mkm',dest/'mkm')
    shutil.copy2(case/'WORK_CONTRACT.md',dest/'README.md')
    # Deliberately exclude judge/, .git, author evidence, and all generators.

def evaluate(key, candidate, output):
    verify_assets()
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    case = ROOT/'cases'/key
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONHASHSEED='0', PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    start = time.monotonic()
    report = {'case':key, 'status':'infrastructure_error', 'score':None}
    with tempfile.TemporaryDirectory(prefix='openrsi-judge-') as temp:
        tmp = Path(temp)
        # A root-only judge tree plus an accessible parent for unprivileged policy IO.
        tmp.chmod(0o711)
        judge = tmp/'judge'
        shutil.copytree(case/'judge',judge)
        judge.chmod(0o700)
        metrics = tmp/'logs/metrics.json'
        env['PYTHONPATH'] = str(judge)
        if key == 'catalyst':
            cmd = [sys.executable,str(judge/'evaluate.py'),'--submission',str(candidate/'mechanism.json'),
                   '--data-dir',str(judge/'data'),'--hidden-dir',str(judge/'hidden'),'--out',str(metrics)]
        else:
            if key == 'alloy':
                env.update(ALLOY_TESTS_DIR=str(judge),ALLOY_PREDICTIONS=str(candidate/'predictions.csv'),
                           ALLOY_VERIFIER_LOG_DIR=str(metrics.parent))
                test = 'test_outputs.py'
            else:
                env.update(ATPR_TESTS_DIR=str(judge),ATPR_SUBMISSION_DIR=str(candidate),
                           ATPR_LOG_DIR=str(metrics.parent),ATPR_WORK_DIR=str(tmp/'policy-work'))
                test = 'test_policy.py'
            cmd = [sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(judge/test)]
        try:
            with (output/'author-only.log').open('w') as log:
                proc = subprocess.run(cmd,env=env,cwd=tmp,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=1500 if key=='treatment' else 1200)
            if proc.returncode == 0 and metrics.exists():
                data = json.loads(metrics.read_text())
                if key == 'catalyst': value = -data['prediction']['mean_nrmse']
                elif key == 'treatment': value = sum(data['worlds'][k]['policy_mean_utility'] for k in ('world_a','world_b'))/2
                else: value = sum(v['all']['skill'] for v in data['skill'].values())/len(data['skill'])
                if not math.isfinite(value): raise RuntimeError('nonfinite evaluator metric')
                report.update(status='scored',score=value)
            elif proc.returncode == 1:
                report['status'] = 'candidate_invalid'
            report['evaluator_returncode'] = proc.returncode
            if metrics.exists(): shutil.copy2(metrics,output/'author-only-metrics.json')
        except subprocess.TimeoutExpired:
            report['status']='timeout'
    report['seconds'] = round(time.monotonic()-start,3)
    (output/'feedback.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return 0 if report['status']=='scored' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['materialize','prepare-work','evaluate','verify'])
    p.add_argument('case',choices=KEYS,nargs='?')
    p.add_argument('--out',type=Path)
    p.add_argument('--candidate',type=Path)
    a=p.parse_args()
    if a.operation=='verify': verify_assets();print('asset hashes OK');return 0
    if not a.case or a.out is None: p.error('case and --out are required')
    dest=a.out.resolve()
    if a.operation=='materialize':materialize(a.case,dest)
    elif a.operation=='prepare-work':prepare_work(a.case,dest)
    else:
        if a.candidate is None:p.error('--candidate is required')
        return evaluate(a.case,a.candidate.resolve(),dest)
    return 0

if __name__=='__main__':raise SystemExit(main())
