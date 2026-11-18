#!/usr/bin/env python3
"""Sync LaTeX generated tables/figures from outputs/full (or --root)."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return '---'
    return f'{float(x):.{digits}f}'

def pct(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return '---'
    return f'{100*float(x):.2f}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=REPO/'outputs'/'full')
    args = ap.parse_args()
    reports = args.root/'reports'
    gen = HERE/'generated'; gen.mkdir(exist_ok=True)
    names = ['ag_news','rotten_tomatoes','banking77']
    labels = {'ag_news':'AG News','rotten_tomatoes':'Rotten Tomatoes','banking77':'Banking77'}

    # Copy EDA figures into reports/figures/<dataset>/
    for n in names:
        src = reports/n/'eda.png'
        if src.exists():
            dst = REPO/'reports'/'figures'/n; dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst/'eda.png')

    # EDA summary
    rows=[]
    for n in names:
        meta = reports/n/'run_metadata.json'
        if not meta.exists(): continue
        d=json.loads(meta.read_text(encoding='utf8'))['data']
        rows.append((labels[n], d.get('n_fit'), d.get('n_val'), d.get('n_test'), d.get('classes'),
                     d.get('length_mean'), d.get('length_p95'), d.get('vocabulary'),
                     d.get('min_train_class'), d.get('max_train_class')))
    lines=['\\begin{tabular}{@{}lrrrrr@{}}','\\toprule',
           'Dataset & Fit/Val/Test & $K$ & Mean len & P95 & Vocab \\\\',
           '\\midrule']
    for r in rows:
        lines.append(f'{r[0]} & {r[1]}/{r[2]}/{r[3]} & {r[4]} & {r[5]:.1f} & {r[6]:.1f} & {r[7]} \\\\')
    lines += ['\\bottomrule','\\end{tabular}']
    (gen/'eda_summary.tex').write_text('\n'.join(lines)+'\n', encoding='utf8')

    # Main comparison
    blocks=['\\begin{table}[H]','\\centering',
            '\\caption{Held-out test results by corpus and model family (Acc/Macro-F1 in \\%).}','\\small',
            '\\begin{tabular}{@{}llrrrr@{}}','\\toprule',
            'Corpus & Family & Acc & Macro-F1 & Brier & ECE \\\\','\\midrule']
    best_lines=['\\begin{table}[H]','\\centering',
                '\\caption{Best Macro-F1 per corpus (descriptive).}','\\small',
                '\\begin{tabular}{@{}llll@{}}','\\toprule',
                'Corpus & Best family & Macro-F1 & Runner-up \\\\','\\midrule']
    cal_lines=['\\begin{table}[H]','\\centering',
               '\\caption{Calibration on test (probability models only).}','\\small',
               '\\begin{tabular}{@{}llrr@{}}','\\toprule',
               'Corpus & Family & Brier & ECE \\\\','\\midrule']
    neg_lines=['\\begin{table}[H]','\\centering',
               '\\caption{Negation vs.\\ other accuracy on test.}','\\small',
               '\\begin{tabular}{@{}llrrrr@{}}','\\toprule',
               'Corpus & Family & Acc$_{\\mathrm{neg}}$ & $n_{\\mathrm{neg}}$ & Acc$_{\\mathrm{other}}$ & Gap \\\\','\\midrule']

    for n in names:
        path=reports/n/'test_results.csv'
        if not path.exists(): continue
        df=pd.read_csv(path)
        for _,r in df.iterrows():
            star='\\textbf{'+pct(r['f1_macro'])+'}' if r['f1_macro']==df['f1_macro'].max() else pct(r['f1_macro'])
            blocks.append(f"{labels[n]} & {r['family']} & {pct(r['accuracy'])} & {star} & {fmt(r.get('brier_score'))} & {fmt(r.get('ece'))} \\\\")
            if pd.notna(r.get('brier_score')):
                cal_lines.append(f"{labels[n]} & {r['family']} & {fmt(r['brier_score'])} & {fmt(r['ece'])} \\\\")
        ranked=df.sort_values('f1_macro', ascending=False)
        lead, runner = ranked.iloc[0], ranked.iloc[1] if len(ranked)>1 else ranked.iloc[0]
        best_lines.append(f"{labels[n]} & {lead['family']} & {pct(lead['f1_macro'])} & {runner['family']} ({pct(runner['f1_macro'])}) \\\\")
        npath=reports/n/'negation.csv'
        if npath.exists():
            for _,r in pd.read_csv(npath).iterrows():
                neg_lines.append(f"{labels[n]} & {r['family']} & {pct(r['acc_negation'])} & {int(r['n_negation'])} & {pct(r['acc_other'])} & {fmt(r['gap'])} \\\\")

    blocks += ['\\bottomrule','\\end{tabular}','\\end{table}']
    best_lines += ['\\bottomrule','\\end{tabular}','\\end{table}']
    cal_lines += ['\\bottomrule','\\end{tabular}','\\end{table}']
    neg_lines += ['\\bottomrule','\\end{tabular}','\\end{table}']
    (gen/'main_comparison.tex').write_text('\n'.join(blocks)+'\n', encoding='utf8')
    (gen/'family_best.tex').write_text('\n'.join(best_lines)+'\n', encoding='utf8')
    (gen/'calibration.tex').write_text('\n'.join(cal_lines)+'\n', encoding='utf8')
    (gen/'negation.tex').write_text('\n'.join(neg_lines)+'\n', encoding='utf8')
    print('Synced generated TeX and EDA figures from', args.root)

if __name__ == '__main__':
    main()
