"""The blob-helix score's tables, as the report quotes them: ``python blob_helix_tables.py RUN_DIR`` (a run's folder: its
``analysis.json`` from ``blob_helix_fit.py`` and its pulled ``volume/``)."""

import json, sys, math
base = sys.argv[1]
a = json.load(open(f'{base}/analysis.json'))
V = f'{base}/volume'
print('## sphere')
for n, row in a['sphere'].items():
    print(n, 'qu', {k: round(v, 3) for k, v in row['qu_quality'].items() if isinstance(v, float)})
    for k, v in row['solves'].items():
        print('  ', k, '%.3e' % v['max_abs'], v['method'], 'cond %.1e' % v.get('condition_at_least', float('nan')))
print('## samplings')
for k in sorted(a['samplings'], key=lambda s: -float(s.split('_')[1])):
    r = a['samplings'][k]
    for sm in ('own', 'qu'):
        q = r[sm]; reg = q['regions']
        print(k, r['samples'], sm, 'min %.2f p5 %.2f med %.2f below %.3f' % (q['nn_over_h_min'], q['nn_over_h_p5'], q['nn_over_h_median'], q['share_below_half_h']),
              '| ratio caps %.2f crest %.2f trough %.2f rim %.2f' % tuple(reg[x]['ratio'] for x in ('caps', 'crest_band', 'trough_band', 'rim_strip')),
              '| rim area %.3f' % reg['rim_strip']['area_share'], ('lloyd_last %s off %.2f' % (q.get('lloyd_last'), q.get('off_dense_over_h', 0)) if sm == 'qu' else ''))
print('## conditioning 0.16 / spectrum 0.3')
for lv in ('0.3000', '0.1600'):
    h = json.load(open(f'{V}/helix_{lv}.json'))
    for k, v in sorted(h['solves'].items()):
        if ':warm' in k or 'float32' in k: continue
        e = v.get('eigen', {})
        print(lv, k, v['method'][:8], 'cond>=%.2e' % v['condition_at_least'] if 'condition_at_least' in v else 'chol %s' % v['cholesky_finite'],
              ('eig ratio %.2e neg %d' % (e['ratio'], e['negative'])) if e else '')
print('## helix')
H = a['helix']
ref = 3.057296
for k in sorted(H):
    arm = H[k]
    if 'float32' in k: continue
    print(k, arm['spacing_mm'])
    print('   U   ', ' '.join('%.4f' % v for v in arm['axial_mm_s']), '| vs post', ' '.join('%+.2f%%' % (100 * (v / ref - 1)) for v in arm['axial_mm_s']))
    for q in ('R_xx', 'R_xOx'):
        print('   e%s' % q, ' '.join('%+.2e' % v for v in arm['errors_post'][q]))
    print('   ROxOx', ' '.join('%.5e' % v for v in arm['R_OxOx']))
    if 'orders' in arm:
        for q in ('axial_mm_s', 'R_xx', 'R_xOx', 'R_OxOx'):
            print('   ord', q, [(round(o['order'], 2) if o.get('order') else o.get('note', '')[:10]) for o in arm['orders'][q]],
                  [round(o['limit'], 4) for o in arm['orders'][q] if o.get('limit') and q == 'axial_mm_s'])
print('## float32')
h = json.load(open(f'{V}/helix_0.1600.json'))
for k, v in sorted(h['solves'].items()):
    if 'float32' in k:
        k64 = k.replace(':float32', '')
        u64 = h['solves'][k64]['axial_mm_s']
        print(k, v['method'], '%.6f vs %.6f rel %.2e' % (v['axial_mm_s'], u64, v['axial_mm_s'] / u64 - 1),
              'Rxx rel %.2e' % (v['R_xx'] / h['solves'][k64]['R_xx'] - 1))
print('## timing (warm)')
for k, v in sorted(h['solves'].items()):
    if 'warm' in k:
        print(k, {kk: round(v[kk], 3) for kk in ('assemble_s', 'factor_s', 'finish_s', 'seconds')})
print('## edges')
E = a['edges']
for b in ('spheroid', 'cylinder'):
    print(b, E[b]['samples'], {s: round(q['nn_over_h_min'], 2) for s, q in E[b]['quality'].items()})
    for k, arm in E[b]['arms'].items():
        if b == 'spheroid':
            print('  ', k, 'drag', ' '.join('%+.2e' % x for x in arm['drag_error']), 'ord', [round(x, 2) if x else None for x in arm['drag_error_orders']])
            print('  ', k, 'torq', ' '.join('%+.2e' % x for x in arm['torque_error']), 'ord', [round(x, 2) if x else None for x in arm['torque_error_orders']])
        else:
            for q in ('F_x', 'T_x', 'F_y'):
                vals = arm[q]
                print('  ', k, q, ' '.join('%.5e' % x for x in vals), 'rel steps', ' '.join('%+.2e' % (vals[i] / vals[i + 1] - 1) for i in range(len(vals) - 1)),
                      [(round(o['order'], 2) if o.get('order') else o.get('note', '')[:10]) for o in arm['orders'][q]])
