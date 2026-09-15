"""Crossed inference for the fixed ten-seed native-loss confirmation."""
from pathlib import Path
import json
import numpy as np
from audit_baseline_claims import crossed_bootstrap_means, summarize_bootstrap
from benchmark_faiss_fixed_rate import _sha256

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'/'native_loss_allocation_10seeds'
SEEDS=[17,42,73,3067,4294,4996,5423,7520,7937,9794]


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cube=[]
    details=[]
    checks=[]
    test_ids=None
    for codec in ('qaq','quip'):
        reports=[]
        for seed in SEEDS:
            report=json.loads((OUT/f'{codec}_seed{seed}.json').read_text())
            assert report['seed']==seed and report['codec']==codec
            assert report['source_sha256']==_sha256(Path(report['source']))
            assert report['compute_backend']['resolved']=='cuda'
            assert report['test_queries']==400 and report['calibration_queries']==5000
            protocol=Path(report['source']).parent.parent
            assert report['protocol_sha256']==_sha256(protocol/'protocol.json')
            ids=np.load(protocol/'test_ids.npy')
            if test_ids is None:test_ids=ids
            np.testing.assert_array_equal(test_ids,ids)
            assert report['quadratic_check_max_abs_error']<1e-6
            for row in report['rows']:
                assert row['persistent_bytes']<=row['cap']
                assert len(row['per_query']['teacher_recall_at_10'])==400
            reports.append(report)
        native='qaq_exact' if codec=='qaq' else 'quip_block'
        for mode in ('raw','cosine'):
            strategy=native+'_'+mode
            maps=[{r['strategy']:r for r in p['rows'] if r['budget']=='matched_ravr_bytes'} for p in reports]
            r=np.array([m['ravr']['per_query']['teacher_recall_at_10'] for m in maps])
            c=np.array([m[strategy]['per_query']['teacher_recall_at_10'] for m in maps])
            cube.append(r-c)
            details.append(dict(codec=codec,mode=mode,ravr_mean=float(r.mean()),control_mean=float(c.mean()),
                seed_deltas=(r-c).mean(1).tolist(),ravr_seed_means=r.mean(1).tolist(),control_seed_means=c.mean(1).tolist(),
                equal_actual_bytes=all(m['ravr']['persistent_bytes']==m[strategy]['persistent_bytes'] for m in maps)))
        for p in reports:
            m={r['strategy']:r for r in p['rows'] if r['budget']=='matched_ravr_bytes'}
            checks.append(dict(codec=codec,seed=p['seed'],native_raw_loss_reduced=
                m[native+'_raw']['native_raw_loss_mean']<=m['ravr']['native_raw_loss_mean']))
    data=np.stack(cube,axis=2)
    for name,section in [('all10',slice(None)),('new7',slice(3,None))]:
        values=data[section]
        boot=crossed_bootstrap_means(values,resamples=20000,seed=20260915)
        estimate,se,pointwise,simultaneous,_=summarize_bootstrap(values,boot)
        for j,d in enumerate(details):
            seed_delta=values[:,:,j].mean(1)
            d[name]=dict(mean_delta=float(estimate[j]),crossed_standard_error=float(se[j]),
                pointwise_95_ci=pointwise[j].tolist(),primary4_simultaneous_95_ci=simultaneous[j].tolist(),
                positive_seeds=int((seed_delta>1e-12).sum()),negative_seeds=int((seed_delta < -1e-12).sum()),
                tied_seeds=int((np.abs(seed_delta)<=1e-12).sum()),seed_count=len(values))
    (OUT/'confirmation.json').write_text(json.dumps(dict(seeds=SEEDS,details=details,validation=checks),indent=2),encoding='utf-8')
    lines=['# Ten-seed confirmation of native-loss allocation', '',
        'Fixed design: 3 original + 7 newly trained codec seeds; same 5000 calibration / 400 test queries. '+
        'Each seed uses a frozen codec and RAVR actual-byte cap. No significance-based stopping. '+
        'QAQ is exact empirical Eq.9; cosine is a serving-score adaptation. Intervals are crossed seed/query bootstrap (20,000).', '',
        '| Codec | Native objective | RAVR R@10 | Control R@10 | Delta (pp) | 95% CI (pp) | Primary-four simultaneous CI (pp) | Positive seeds |',
        '|---|---|---:|---:|---:|---|---|---|']
    def interval(x):return f'[{100*x[0]:+.3f}, {100*x[1]:+.3f}]'
    for d in details:
        a=d['all10']
        lines.append(f"| {d['codec']} | {d['mode']} | {d['ravr_mean']:.6f} | {d['control_mean']:.6f} | {100*a['mean_delta']:+.3f} | {interval(a['pointwise_95_ci'])} | {interval(a['primary4_simultaneous_95_ci'])} | {a['positive_seeds']}/10 |")
    lines+=['','## Seven new seeds only','',
        '| Codec | Native objective | Delta (pp) | 95% CI (pp) | Primary-four simultaneous CI (pp) | Positive seeds |',
        '|---|---|---:|---|---|---|']
    for d in details:
        a=d['new7']
        lines.append(f"| {d['codec']} | {d['mode']} | {100*a['mean_delta']:+.3f} | {interval(a['pointwise_95_ci'])} | {interval(a['primary4_simultaneous_95_ci'])} | {a['positive_seeds']}/7 |")
    lines+=['','## Per-seed differences (percentage points)','',
        '| Seed | QAQ raw | QAQ cosine | QUIP raw | QUIP cosine |', '|---|---:|---:|---:|---:|']
    for i,seed in enumerate(SEEDS):lines.append('| '+str(seed)+' | '+' | '.join(f'{100*d["seed_deltas"][i]:+.3f}' for d in details)+' |')
    lines+=['','## Interpretation limits','',
        '- New seeds vary codec training, not the held-out query sample. More seeds cannot remove uncertainty from 400 shared queries.',
        '- A confidence interval containing zero does not establish equivalence. No practical-equivalence margin was specified.',
        '- Native-loss allocation is a controlled variable-rate extension, not the original full fixed-rate algorithm.',
        '- evaluation.md additionally reports original-cap sensitivity and every allocation control; summary.json supplies simultaneous intervals over that larger family.',
        '- Source codec / protocol hashes, fixed test IDs, score lengths, byte caps and numerical formula checks were verified.']
    (OUT/'confirmation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),sharey=True)
    for axis,codec in zip(axes,('qaq','quip')):
        for j,d in enumerate([d for d in details if d['codec']==codec]):
            positions=j+np.linspace(-.11,.11,10)
            axis.scatter(positions[:3],100*np.array(d['seed_deltas'][:3]),color='#aab5c0',label='Original 3 seeds' if j==0 else None,zorder=3)
            axis.scatter(positions[3:],100*np.array(d['seed_deltas'][3:]),color='#327c9b',label='New 7 seeds' if j==0 else None,zorder=3)
            a=d['all10'];v=100*a['mean_delta'];ci=100*np.array(a['pointwise_95_ci'])
            axis.errorbar(j+.23,v,yerr=[[v-ci[0]],[ci[1]-v]],fmt='D',color='#b25c32',capsize=5,label='10-seed mean + crossed 95% CI' if j==0 else None)
        axis.axhline(0,color='black',lw=1)
        axis.set_xticks([0,1],['Native raw loss','Cosine adaptation'])
        axis.set_title(codec.upper()+' frozen codec')
        axis.set_xlim(-.4,1.55)
        axis.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('RAVR minus native-loss allocation (percentage points)')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('Independent codec-seed confirmation\nSame SOP 100K gallery and 400 test queries')
    fig.tight_layout(rect=(0,.09,1,1))
    fig.savefig(OUT/'seed_confirmation.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':summarize()
