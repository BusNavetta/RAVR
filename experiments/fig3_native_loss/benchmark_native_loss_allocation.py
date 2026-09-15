from pathlib import Path
import argparse
import json
import math
import time
import numpy as np
import torch

from benchmark_faiss_fixed_rate import _load_dataset, _evaluate_reconstruction, _sha256
from benchmark_qaq_ravr import _load_state
from compute_backend import configure_compute_backend, backend_report
from qaq_pq import QAQPQIndex, unpack_codes, dimension_permutation
from qaq_ravr import QAQNestedIndex
from ravr_rq import solve_lagrangian
from audit_baseline_claims import crossed_bootstrap_means, summarize_bootstrap

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs' / 'native_loss_allocation'


def native_distortions(source, gallery, queries, order, allowed):
    torch.backends.cuda.matmul.allow_tf32 = False
    permutation = dimension_permutation(source.metadata, source.dimension)
    x = np.asarray(gallery if permutation is None else gallery[:, permutation], dtype=np.float32)
    q = np.asarray(queries if permutation is None else queries[:, permutation], dtype=np.float32)
    codes = unpack_codes(source.packed_codes, source.stages, source.nbits)
    qt = torch.tensor(q, device='cuda')
    ds = source.dimension // source.stages
    blocks = qt.reshape(len(q), source.stages, ds)
    cov = torch.einsum('nmd,nme->mde', blocks, blocks) / len(q)
    keys = ['qaq_exact_raw', 'qaq_exact_cosine', 'quip_block_raw', 'quip_block_cosine', 'qip_cosine']
    result = {key: np.empty((len(x), len(allowed)), dtype=np.float64) for key in keys}
    check_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(x), 1024):
            stop = min(start + 1024, len(x))
            xt = torch.tensor(x[start:stop], device='cuda')
            score = xt @ qt.T
            weight = torch.softmax(score, dim=1)
            recon = torch.zeros_like(xt)
            for position, stage in enumerate(order, 1):
                lo, hi = int(stage) * ds, (int(stage) + 1) * ds
                recon[:, lo:hi] = torch.tensor(source.codebooks[stage, codes[start:stop, stage].astype(int)], device='cuda')
                if position not in allowed:
                    continue
                option = allowed.index(position)
                for mode in ('raw', 'cosine'):
                    approx = recon if mode == 'raw' else recon / torch.linalg.vector_norm(recon, dim=1, keepdim=True).clamp_min(1e-9)
                    error = xt - approx
                    error_scores = error @ qt.T
                    result['qaq_exact_' + mode][start:stop, option] = (weight * error_scores.square()).sum(1).cpu().numpy()
                    eb = error.reshape(-1, source.stages, ds)
                    result['quip_block_' + mode][start:stop, option] = torch.einsum('bmd,mde,bme->b', eb, cov, eb).cpu().numpy()
                    if mode == 'cosine':
                        result['qip_cosine'][start:stop, option] = error_scores.square().mean(1).cpu().numpy()
                    if start == 0:
                        # Independent float64 quadratic-form check of Eq. 9.
                        e = error[0].double()
                        qd = qt.double()
                        w = torch.softmax(qd @ xt[0].double(), dim=0)
                        matrix = (qd.T * w) @ qd
                        direct = float(e @ matrix @ e)
                        computed = result['qaq_exact_' + mode][0, option]
                        check_error = max(check_error, abs(direct - computed))
            if start % 20480 == 0:
                print(f'  loss rows {stop}/{len(x)}', flush=True)
    assert check_error < 1e-6, check_error
    assert all(np.isfinite(v).all() and v.min() >= -1e-7 for v in result.values())
    return result, check_error


def run(codec, seed, root):
    started = time.time()
    protocol_dir = root / f'seed{seed}'
    protocol = json.loads((protocol_dir / 'protocol.json').read_text())
    base = _load_dataset('sop', Path(protocol['source_manifest']).parent.parent)
    assert _sha256(base['manifest_path']) == protocol['source_manifest_sha256']
    stem = (f'qaq_m24_k8_kv100_seed{seed}_s20_it2' if codec == 'qaq'
            else f'quip_covz_m6_k256_seed{seed}_it50_randomperm')
    source_path = protocol_dir / 'bundles' / (stem + ('.qaqix' if codec == 'qaq' else '.quipix'))
    source = QAQPQIndex.load(source_path)
    state_path = source_path.with_name(stem + '_ravr_cal5000.npz')
    state = _load_state(state_path)
    assert state['source_sha256'] == _sha256(source_path)
    order, allowed = state['subspace_order'], state['allowed_lengths']
    queries = np.load(protocol_dir / 'calibration.npy')
    for role in ('train', 'calibration', 'validation'):
        assert not np.intersect1d(np.load(protocol_dir / f'{role}_ids.npy'), base['exclude_ids']).size
    print(f'{codec} seed {seed}: computing calibration losses', flush=True)
    losses, check_error = native_distortions(source, base['gallery'], queries, order, allowed)
    losses['ravr'] = state['distortion']
    losses['reconstruction'] = state['prior_distortion']
    np.savez_compressed(OUT / f'{codec}_seed{seed}_losses.npz', **losses)
    costs = np.array([math.ceil(m * source.nbits / 8) for m in allowed])
    variable = costs - costs[0]
    offset = QAQNestedIndex.code_offset_for(source.codebooks[order], order, len(allowed), source.n_items)
    full_cap = 1497824
    original_budget = full_cap - offset - source.n_items * costs[0]
    ravr = solve_lagrangian(state['distortion'], variable, int(original_budget))
    matched_cap = int(offset + source.n_items * costs[0] + variable[ravr].sum())
    native = 'qaq_exact' if codec == 'qaq' else 'quip_block'
    chosen_losses = ['ravr', native + '_raw', native + '_cosine', 'qip_cosine', 'reconstruction']
    teacher_ids = np.load(protocol_dir / 'test_teacher_ids.npy')
    teacher_scores = np.load(protocol_dir / 'test_teacher_scores.npy')
    rows = []
    for budget_name, cap in [('matched_ravr_bytes', matched_cap), ('original_cap', full_cap)]:
        if budget_name == 'original_cap' and cap == matched_cap:
            continue
        budget = int(cap - offset - source.n_items * costs[0])
        for strategy in chosen_losses:
            selected = ravr if strategy == 'ravr' else solve_lagrangian(losses[strategy], variable, budget)
            index = QAQNestedIndex.from_allocation(source, order, state['prefix_norms'], allowed, selected, strategy=strategy)
            path = OUT / f'{codec}_seed{seed}_{budget_name}_{strategy}.qaqrvix'
            index.save(path)
            loaded = QAQNestedIndex.load(path)
            assert loaded.persistent_bytes <= cap
            assert path.stat().st_size == loaded.persistent_bytes
            metrics, per_query = _evaluate_reconstruction(loaded.decode_normalized(), base, teacher_ids, teacher_scores)
            row = dict(strategy=strategy, budget=budget_name, cap=cap,
                       persistent_bytes=loaded.persistent_bytes,
                       allocation_histogram=loaded.metadata['allocation_histogram'],
                       own_loss_mean=float(losses[strategy][np.arange(source.n_items), selected].mean()),
                       native_raw_loss_mean=float(losses[native + '_raw'][np.arange(source.n_items), selected].mean()),
                       **metrics, per_query=per_query)
            rows.append(row)
            print(f"  {budget_name} {strategy}: {loaded.persistent_bytes} bytes, R10={metrics['teacher_recall_at_10']:.6f}", flush=True)
            loaded.close()
    report = dict(codec=codec, seed=seed, source=str(source_path), source_sha256=_sha256(source_path),
                  calibration_state_sha256=_sha256(state_path), protocol_sha256=_sha256(protocol_dir / 'protocol.json'),
                  allowed_lengths=list(allowed), subspace_order=order.tolist(), calibration_queries=len(queries),
                  test_queries=len(base['queries']), gallery_items=source.n_items, compute_backend=backend_report(),
                  quadratic_check_max_abs_error=check_error, seconds=time.time()-started, rows=rows,
                  objective_note='QAQ exact Eq9 on all 5000 queries; no cluster/sample approximation. Raw=native IP, cosine=normalized adaptation. QUIP=block covariance. Calibration self-pairs included as in native objective. No test selection.')
    (OUT / f'{codec}_seed{seed}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    source.close()


def aggregate(seeds=(17, 42, 73)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ['# Native-loss matched allocation audit', '',
             f'SOP 100K; frozen FP32 codec; seeds {list(seeds)}; 5000 calibration queries; 400 test queries. '+
             'All allocations use the same serializer and exhaustive normalized scoring. '+
             'QAQ exact Eq.9 uses all calibration queries, without the native sampled/cluster approximation. '+
             'These are native-loss-inspired variable-rate controls, not claims about the original fixed-rate algorithms.', '',
             '| Codec | Budget | Allocation | R@10 | Mean bytes/item | RAVR minus allocation | Pointwise 95% CI |',
             '|---|---|---|---:|---:|---:|---|']
    comparisons, differences = [], []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, codec in zip(axes, ('qaq', 'quip')):
        reports = [json.loads((OUT / f'{codec}_seed{s}.json').read_text()) for s in seeds]
        budgets = sorted(set(r['budget'] for r in reports[0]['rows']))
        for budget in budgets:
            maps = [{r['strategy']: r for r in p['rows'] if r['budget'] == budget} for p in reports]
            labels, values = [], []
            for strategy in maps[0]:
                recall = np.array([m[strategy]['per_query']['teacher_recall_at_10'] for m in maps])
                ravr = np.array([m['ravr']['per_query']['teacher_recall_at_10'] for m in maps])
                delta = ravr - recall
                ci = np.quantile(crossed_bootstrap_means(delta[:,:,None],resamples=20000,seed=20260915),[.025,.975])
                bpi = np.mean([m[strategy]['persistent_bytes']/100000 for m in maps])
                lines.append(f'| {codec} | {budget} | {strategy} | {recall.mean():.6f} | {bpi:.5f} | {delta.mean():+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
                if strategy != 'ravr':
                    comparisons.append(dict(codec=codec,budget=budget,strategy=strategy,delta=float(delta.mean()),ci=ci.tolist()))
                    differences.append(delta)
                if budget == 'matched_ravr_bytes':
                    label = strategy.replace('qaq_exact','QAQ').replace('quip_block','QUIP').replace('_','\n')
                    if any(m[strategy]['persistent_bytes'] < m[strategy]['cap'] for m in maps):
                        label += '*'
                    labels.append(label)
                    values.append(recall.mean())
            if budget == 'matched_ravr_bytes':
                ax.bar(labels,values,color=['#243b53','#d78c45','#56a3a6','#8796a5','#b7bec7'])
                for i,v in enumerate(values): ax.text(i,v+.002,f'{v:.4f}',ha='center',fontsize=9)
                ax.set_title(codec.upper()+' frozen codec')
                ax.set_ylabel('Teacher Recall@10')
                ax.set_ylim(0,max(values)*1.18)
                ax.grid(axis='y',alpha=.2)
    cube=np.stack(differences,axis=2)
    boot=crossed_bootstrap_means(cube,resamples=20000,seed=20260915)
    _,_,_,simultaneous,_=summarize_bootstrap(cube,boot)
    for comparison, interval in zip(comparisons,simultaneous): comparison['simultaneous_95_ci']=interval.tolist()
    (OUT/'summary.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')
    (OUT/'evaluation.md').write_text('\n'.join(lines)+'\n\nIntervals resample codec seeds and shared queries independently (20,000 replicates). '+
        'summary.json also provides simultaneous intervals over every non-RAVR comparison in this audit. Query set remains fixed across seeds.\n',encoding='utf-8')
    fig.suptitle(f'Native-loss allocation controls at RAVR actual-byte caps\nSame codec, prefixes and test queries; {len(seeds)}-seed means')
    fig.text(.5,.015,'* At least one seed underuses its cap. Bars are means; actual bytes and paired intervals are in evaluation.md.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(OUT/'native_loss_allocation.png',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol-root',type=Path,default=Path('D:/Research/Data/AJigma/qaq_official/sop'))
    parser.add_argument('--aggregate-only',action='store_true')
    parser.add_argument('--seeds',type=int,nargs='+',default=[17,42,73])
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    configure_compute_backend('cuda')
    torch.set_num_threads(4)
    if not args.aggregate_only:
        for codec in ('qaq','quip'):
            for seed in args.seeds: run(codec,seed,args.protocol_root)
    aggregate(args.seeds)
