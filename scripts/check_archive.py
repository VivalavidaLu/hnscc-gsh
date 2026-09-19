"""Offline consistency checks. No source data are modified; no packages required."""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]

def read(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def keyed(name, key='cell_id'):
    rows = read(name)
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'Duplicate identifiers: ' + name)
    return result

def check_scfea():
    full = keyed('data/processed/suppFigS2C_scFEA_full_ranked_modules_P1_vs_nonP1_all_modules.csv', 'module')
    panel = keyed('data/processed/section2_panelD_scFEA_heatmap_v8_module_ids_source.csv', 'module')
    p5 = keyed('data/processed/section2_scFEA_P1_vs_P5_direct.csv', 'module')
    old = keyed('data/processed/suppFigS2C_scFEA_full_ranked_modules_source.csv', 'module')
    require(len(full) == 168 and len(panel) == 7, 'Unexpected scFEA module counts')
    for module, row in panel.items():
        for left, right in [('d_P1_vs_nonP1','d'), ('P1_mean','P1_mean'), ('nonP1_mean','nonP1_mean')]:
            require(math.isclose(float(row[left]), float(full[module][right]), abs_tol=1e-9), 'scFEA contrast mismatch: '+module+'/'+left)
    require(math.isclose(float(full['M_80']['d']), 0.9957870843607348, abs_tol=1e-9), 'M80 reference mismatch')
    historical_matches = all(math.isclose(float(row['d']), float(p5[module]['d']), abs_tol=1e-9) for module,row in old.items())
    return {'modules':len(full), 'panel_modules':len(panel), 'matched_contrast':'P1 versus non-P1',
            'M80_d':float(full['M_80']['d']), 'M80_P1_vs_P5_d':float(p5['M_80']['d']),
            'historical_25_row_table_matches_P1_vs_P5':historical_matches}

def check_archive():
    sources = [p for p in ROOT.rglob('*.py') if not {'.venv','.git','__pycache__'}.intersection(p.relative_to(ROOT).parts)]
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=path.relative_to(ROOT).as_posix())
    provenance = read('DATA_PROVENANCE.csv')
    for row in provenance:
        path = ROOT / row['repository_path']
        require(path.is_file(), 'Missing data file: '+row['repository_path'])
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], 'Checksum mismatch: '+row['repository_path'])
        require(path.stat().st_size == int(row['bytes']), 'Size mismatch: '+row['repository_path'])
    bundled = {p.relative_to(ROOT).as_posix() for p in (ROOT/'data').rglob('*') if p.is_file() and p.suffix in {'.csv','.txt','.json'}}
    require(bundled == {row['repository_path'] for row in provenance}, 'Provenance does not cover every bundled data file')
    final = keyed('data/derived/malignant_DEFINITIVE_FINAL_cellids.csv')
    pure = keyed('data/derived/malignant_PURE_FINAL_cellids.csv')
    labels = keyed('data/processed/cell_level_malignant_ifn_apm_scores.csv')
    classification = keyed('data/processed/malignant_classification.csv')
    usage_rows = read('data/derived/cNMF_K6_usage_6165.csv')
    id_column = next(iter(usage_rows[0]))
    usage = {r[id_column]:r for r in usage_rows}
    require(len(usage)==len(usage_rows), 'Duplicate usage cell IDs')
    require(len(final)==6165 and len(pure)==10856 and set(final)<=set(pure), 'Frozen cell-universe mismatch')
    require(set(final)==set(labels)==set(usage), 'Labels/usage cell universes differ')
    require(all(classification[c]['MALIGNANT_DEFINITIVE'].lower() in {'true','1'} for c in final), 'Final cells lack frozen malignant evidence')
    require(Counter(r['program'] for r in labels.values()) == {'P1':1857,'P2':830,'P3':904,'P4':850,'P5':1081,'P6':643}, 'Final label counts differ')
    mismatch = p1_changes = 0
    argmax_counts = Counter()
    for cell,row in usage.items():
        values = {k:float(row[k]) for k in ['1','2','3','4','5','6']}
        require(all(math.isfinite(v) and v>=0 for v in values.values()), 'Invalid usage values')
        require(math.isclose(sum(values.values()),1,abs_tol=1e-6), 'Usage is not row-normalized')
        predicted = 'P'+max(values, key=values.get)
        observed = labels[cell]['program']
        argmax_counts[predicted]+=1
        mismatch += predicted != observed
        p1_changes += (predicted=='P1') != (observed=='P1')
    return {'syntax_files':len(sources), 'checksummed_data_files':len(provenance),
            'frozen_final_cells':len(final), 'frozen_purity_cells':len(pure),
            'usage_argmax_label_mismatch':mismatch, 'P1_membership_changes':p1_changes,
            'argmax_counts':dict(sorted(argmax_counts.items())), 'scfea':check_scfea(),
            'full_analysis_rerun':False,
            'unresolved':['Final usage/label provenance' if mismatch else None,
                          'Missing figure-specific methods and primary data; see KNOWN_ISSUES.md']}

def main():
    try:
        report = check_archive()
    except Exception as error:
        # Suppress filesystem paths from routine public-facing diagnostics.
        print('Archive validation failed: '+type(error).__name__)
        if isinstance(error, ValueError):
            print(str(error))
        return 1
    print(json.dumps(report, indent=2))
    if '--strict' in sys.argv and any(report['unresolved']):
        print('NOT READY for a claim of complete manuscript reproducibility.')
        return 2
    print('Archive integrity checks passed; scientific consistency issues remain.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
