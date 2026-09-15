from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


def bootstrap_means(values, resamples=20000, seed=20260915):
    """Paired crossed bootstrap: independent seed and shared-query resampling.

    values is [codec seed, test query, allocation method]. All methods share
    each bootstrap draw. Matches the experiment's existing inference routine.
    """
    ns, nq, nm = values.shape
    rng = np.random.default_rng(seed)
    result = np.empty((resamples, nm), dtype=np.float64)
    for start in range(0, resamples, 512):
        size = min(512, resamples-start)
        seed_draws = rng.integers(0, ns, size=(size, ns))
        query_draws = rng.integers(0, nq, size=(size, nq))
        batch = np.zeros((size, nm), dtype=np.float64)
        for s in range(ns):
            counts = (seed_draws == s).sum(axis=1) / ns
            batch += values[s][query_draws].mean(axis=1) * counts[:, None]
        result[start:start+size] = batch
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=here/'native_plot_data.json')
    parser.add_argument('--output', type=Path, default=here/'ravr_vs_native')
    args = parser.parse_args()
    data = json.loads(args.data.read_text(encoding='utf-8'))
    # Match the manuscript's serif typography; allow portable fallback.
    for name in ('times.ttf', 'timesbd.ttf'):
        path = Path('C:/Windows/Fonts')/name
        if path.exists():
            font_manager.fontManager.addfont(path)
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Times New Roman', 'Nimbus Roman', 'Liberation Serif')
                 if f in available), 'DejaVu Serif')
    plt.rcParams.update({'font.family':font, 'font.size':8,
                         'mathtext.fontset':'stix', 'pdf.fonttype':42,
                         'ps.fonttype':42, 'axes.linewidth':.55,
                         'xtick.major.width':.55, 'ytick.major.width':.55,
                         'ytick.major.size':2.3})
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.30), sharey=True)
    fig.subplots_adjust(left=.17, right=.985, bottom=.19, top=.72, wspace=.25)
    # Reuse the draft's Q-IP blue diamond and RAVR dark star.
    styles = [('#2563eb', 'D', 4.6), ('#111827', '*', 8.0)]
    stats = {}
    for ax, codec, letter in zip(axes, ('qaq','quip'), ('a','b')):
        entry = data['codecs'][codec]
        values = np.asarray(entry['per_query_hits'], dtype=np.float64) / 10.0
        assert values.shape == (10, 400, 2)
        assert np.array_equal(entry['native_bytes'], entry['ravr_bytes'])
        means = values.mean(axis=(0,1))
        boot = bootstrap_means(values)
        ci = np.quantile(boot, [.025,.975], axis=0)
        delta = float((values[:,:,1]-values[:,:,0]).mean())
        delta_ci = np.quantile(boot[:,1]-boot[:,0], [.025,.975])
        expected = entry['paired_reference']
        np.testing.assert_allclose(delta, expected['mean_delta'], atol=1e-12)
        np.testing.assert_allclose(delta_ci, expected['pointwise_95_ci'], atol=1e-12)
        ax.plot([0,1], means, color='#9ca3af', linewidth=.7, zorder=2)
        for j, (color,marker,size) in enumerate(styles):
            ax.errorbar(j, means[j], yerr=[[means[j]-ci[0,j]],[ci[1,j]-means[j]]],
                        fmt='none', ecolor='#9ca3af', elinewidth=.75, capsize=2.4, zorder=3)
            ax.plot(j, means[j], marker=marker, color=color, markersize=size,
                    markeredgecolor='white', markeredgewidth=.45, linestyle='none', zorder=4)
            ax.text(j, ci[1,j]+.0025, f'{means[j]:.4f}', ha='center', va='bottom', fontsize=7)
        bpi = np.mean(entry['ravr_bytes']) / data['gallery_items']
        ax.set_title(f'({letter}) {codec.upper()} codec\n{bpi:.4f} B/item',
                     fontsize=8, fontweight='bold', pad=5)
        ax.set_xlim(-.45,1.45)
        ax.set_ylim(.095,.188)
        ax.set_yticks([.10,.12,.14,.16,.18])
        ax.set_xticks([])
        ax.grid(axis='y', color='#e5e7eb', linewidth=.45)
        ax.set_axisbelow(True)
        ax.spines[['top','right','bottom']].set_visible(False)
        ax.text(.5,-.105,f'$\\Delta$ = {delta*100:+.2f} pp',transform=ax.transAxes,
                ha='center',va='top',fontsize=7.5)
        printed_ci = np.round(delta_ci*100 + 1e-9, 2)
        ax.text(.5,-.235,f'95% CI [{printed_ci[0]:.2f}, {printed_ci[1]:.2f}]',
                transform=ax.transAxes,ha='center',va='top',fontsize=6.3)
        stats[codec] = dict(native_mean=float(means[0]),ravr_mean=float(means[1]),
                           marginal_pointwise_95_ci=ci.T.tolist(),
                           paired_delta=delta,paired_pointwise_95_ci=delta_ci.tolist(),
                           mean_actual_bytes_per_item=float(bpi))
    axes[0].set_ylabel('Teacher Recall@10',labelpad=2)
    handles = [Line2D([0],[0],marker='D',color='#2563eb',markersize=4.2,
                      linestyle='none',label='Native-loss allocation'),
               Line2D([0],[0],marker='*',color='#111827',markersize=6.8,
                      linestyle='none',label='RAVR (Ordinal-KL)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),
               ncol=2,frameon=False,fontsize=7,handlelength=.8,
               columnspacing=1.1,handletextpad=.4)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.pdf'),metadata={
        'Title':'RAVR and native-loss allocation on frozen QAQ and QUIP codecs',
        'Subject':'SOP-100K; 10 seeds; matched actual bytes; pointwise crossed bootstrap intervals'})
    fig.savefig(args.output.with_suffix('.png'),dpi=300)
    plt.close(fig)
    args.output.with_suffix('.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
