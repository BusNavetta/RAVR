from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=here/'generated')
    args = parser.parse_args()
    try:
        import numpy as np
        import matplotlib
    except ImportError as exc:
        raise SystemExit('Install the plotting dependencies first: python -m pip install -r '
                         + str(here.parents[1]/'requirements.txt')) from exc
    manifest = json.loads((here/'MANIFEST.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = here/relative
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit('Missing or modified package input: '+relative)
    confirmation = json.loads((here/'outputs/native_loss_allocation_10seeds/confirmation.json').read_text(encoding='utf-8'))
    seeds = confirmation['seeds']
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError('Figure 3 requires the ten distinct archived seeds')
    data = dict(gallery_items=100000, seeds=seeds,
                method_order=['native_loss','ravr'], codecs={})
    for codec, strategy in [('qaq','qaq_exact_raw'),('quip','quip_block_raw')]:
        hits, native_bytes, ravr_bytes, provenance = [], [], [], {}
        for seed in seeds:
            file = here/'outputs/native_loss_allocation_10seeds'/f'{codec}_seed{seed}.json'
            report = json.loads(file.read_text(encoding='utf-8'))
            assert report['codec'] == codec and report['seed'] == seed
            assert report['gallery_items'] == 100000 and report['test_queries'] == 400
            assert report['calibration_queries'] == 5000
            rows = {r['strategy']:r for r in report['rows']
                    if r['budget'] == 'matched_ravr_bytes'}
            native, ravr = rows[strategy], rows['ravr']
            assert native['persistent_bytes'] == ravr['persistent_bytes']
            values = np.stack([native['per_query']['teacher_recall_at_10'],
                               ravr['per_query']['teacher_recall_at_10']],axis=1)
            assert values.shape == (400,2) and np.isfinite(values).all()
            counts = np.rint(values*10).astype(int)
            np.testing.assert_allclose(counts/10, values, atol=1e-12)
            assert counts.min() >= 0 and counts.max() <= 10
            hits.append(counts.tolist())
            native_bytes.append(native['persistent_bytes'])
            ravr_bytes.append(ravr['persistent_bytes'])
            provenance[file.name] = sha256(file)
        ref = next(r for r in confirmation['details']
                   if r['codec'] == codec and r['mode'] == 'raw')
        data['codecs'][codec] = dict(per_query_hits=hits,native_bytes=native_bytes,
                                    ravr_bytes=ravr_bytes,source_sha256=provenance,
                                    paired_reference=ref['all10'])
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=True)
    input_path = output/'reconstructed_plot_data.json'
    input_path.write_text(json.dumps(data,separators=(',',':')),encoding='utf-8')
    spec = importlib.util.spec_from_file_location('figure3_plot',here/'plot_ravr_native.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_argv = sys.argv
    try:
        sys.argv = [str(here/'plot_ravr_native.py'),'--data',str(input_path),
                    '--output',str(output/'figure3_native')]
        module.main()
    finally:
        sys.argv = original_argv
    actual = json.loads((output/'figure3_native.json').read_text())
    expected = json.loads((here/'figures/figure3_native.json').read_text())
    for codec in ('qaq','quip'):
        for key in expected[codec]:
            np.testing.assert_allclose(actual[codec][key],expected[codec][key],atol=1e-12,rtol=0)
    (output/'verification.json').write_text(json.dumps(dict(
        status='PASS', reports=20,seeds=seeds,test_queries=400,
        bootstrap_replicates=20000,paired_byte_matches=True,
        all_plotted_statistics_match_reference=True,
        numpy=np.__version__,matplotlib=matplotlib.__version__,
        input_manifest_sha256=sha256(here/'MANIFEST.json')),indent=2),encoding='utf-8')
    print('PASS: verified 20 reports; recomputed means and 20,000 crossed-bootstrap intervals.')
    print('Figure:',output/'figure3_native.pdf')
    print('Preview:',output/'figure3_native.png')


if __name__ == '__main__':
    main()
