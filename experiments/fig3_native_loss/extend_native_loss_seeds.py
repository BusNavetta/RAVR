import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'/'native_loss_allocation_10seeds'
ROOT=Path('D:/Research/Data/AJigma/qaq_official/sop')
SEEDS=[17,42,73,3067,4294,4996,5423,7520,7937,9794]


def command(script, args, log):
    env=dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUTF8='1')
    started=time.time()
    with log.open('w',encoding='utf-8') as handle:
        subprocess.run([sys.executable,'-X','utf8',str(HERE/script),*map(str,args)],
                       env=env,stdout=handle,stderr=subprocess.STDOUT,check=True,cwd=HERE.parent)
    print(f'{script}: complete in {time.time()-started:.0f}s',flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'design.json').write_text(json.dumps(dict(seeds=SEEDS,new_seeds=SEEDS[3:],
        stopping_rule='Run every listed seed for both codecs; no significance-based stopping.',
        main_comparisons=['RAVR vs native raw','RAVR vs native cosine adaptation'],
        fixed_queries=True,codec_retraining=True,source_three_seed_directory='native_loss_allocation'),indent=2))
    for seed in SEEDS[:3]:
        for codec in ('qaq','quip'):
            shutil.copy2(HERE/'outputs'/'native_loss_allocation'/f'{codec}_seed{seed}.json',OUT/f'{codec}_seed{seed}.json')
    for seed in SEEDS[3:]:
        protocol=ROOT/f'seed{seed}'
        print(f'Seed {seed}: preparing independent codec training draw',flush=True)
        command('prepare_qaq_sop_protocol.py',['--codec-seed',seed],OUT/f'seed{seed}_protocol.log')
        for codec in ('qaq','quip'):
            if (OUT/f'{codec}_seed{seed}.json').exists():
                print(f'{codec} seed {seed}: completed result exists',flush=True)
                continue
            print(f'{codec} seed {seed}: training frozen codec',flush=True)
            args=['--protocol',protocol]
            if codec=='qaq':args+=['--M',24,'--K',8,'--kv',100,'--sample-count',20,'--train-iterations',2]
            command(f'benchmark_{codec}_official.py',args,OUT/f'{codec}_seed{seed}_training.log')
            print(f'{codec} seed {seed}: calibrating RAVR',flush=True)
            command(f'benchmark_{codec}_ravr.py',['--protocol',protocol],OUT/f'{codec}_seed{seed}_ravr.log')
            print(f'{codec} seed {seed}: evaluating native-loss allocations',flush=True)
            env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUTF8='1')
            code=('from pathlib import Path; import torch; import benchmark_native_loss_allocation as b; '
                  f'b.OUT=Path({str(OUT)!r}); b.configure_compute_backend("cuda"); torch.set_num_threads(4); '
                  f'b.run({codec!r},{seed},Path({str(ROOT)!r}))')
            with (OUT/f'{codec}_seed{seed}_native.log').open('w',encoding='utf-8') as handle:
                subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=HERE,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
            print(f'{codec} seed {seed}: finished',flush=True)
    command('benchmark_native_loss_allocation.py',['--aggregate-only','--seeds',*SEEDS,'--output',OUT],OUT/'aggregate.log')
    print('All ten seeds complete.',flush=True)
