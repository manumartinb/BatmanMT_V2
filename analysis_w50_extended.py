#!/usr/bin/env python3
"""
Análisis Extendido W50 - Filtros Específicos
=============================================
- BQI_ABS (valores altos)
- Ratio Theta K1 / Theta K2
- Golden Cross y otros cruces SMA
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

CSV_FILE = "combined_BATMAN_mediana_w_stats_w_vix.csv"
TARGET = "PnL_fwd_pts_50_mediana"
MIN_BIN_SIZE = 30

print("=" * 80)
print("ANÁLISIS EXTENDIDO W50 - FILTROS ESPECÍFICOS")
print("=" * 80)

# Cargar datos
df = pd.read_csv(CSV_FILE)
df = df[df[TARGET].notna()].copy()
N = len(df)
print(f"\nN total: {N:,}")
print(f"Target media: {df[TARGET].mean():.2f}, mediana: {df[TARGET].median():.2f}")

# =============================================================================
# 1. ANÁLISIS BQI_ABS
# =============================================================================
print("\n" + "=" * 80)
print("1. ANÁLISIS BQI_ABS (valores altos)")
print("=" * 80)

if 'BQI_ABS' in df.columns:
    bqi_data = df[['BQI_ABS', TARGET]].dropna()

    print(f"\nEstadísticas BQI_ABS:")
    print(f"  N: {len(bqi_data):,}")
    print(f"  Media: {bqi_data['BQI_ABS'].mean():.4f}")
    print(f"  Mediana: {bqi_data['BQI_ABS'].median():.4f}")
    print(f"  Min/Max: {bqi_data['BQI_ABS'].min():.4f} / {bqi_data['BQI_ABS'].max():.4f}")

    # Correlación
    r_sp, p_sp = spearmanr(bqi_data['BQI_ABS'], bqi_data[TARGET])
    r_pe, p_pe = pearsonr(bqi_data['BQI_ABS'], bqi_data[TARGET])
    print(f"\nCorrelación BQI_ABS vs Target:")
    print(f"  Spearman: r={r_sp:.4f}, p={p_sp:.6f}")
    print(f"  Pearson:  r={r_pe:.4f}, p={p_pe:.6f}")

    # Análisis por percentiles
    print("\n--- Análisis por Percentiles BQI_ABS ---")
    percentiles = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]

    print(f"\n{'Percentil':<12} {'Umbral':>10} {'N_arriba':>10} {'Med_arriba':>12} {'Med_abajo':>12} {'Δ_Med':>10} {'p-val':>10}")
    print("-" * 80)

    best_bqi = {'diff': 0, 'pctl': None}
    for pctl in percentiles:
        threshold = bqi_data['BQI_ABS'].quantile(pctl/100)
        above = bqi_data[bqi_data['BQI_ABS'] >= threshold][TARGET]
        below = bqi_data[bqi_data['BQI_ABS'] < threshold][TARGET]

        if len(above) >= MIN_BIN_SIZE and len(below) >= MIN_BIN_SIZE:
            med_above = above.median()
            med_below = below.median()
            diff = med_above - med_below
            try:
                _, pval = mannwhitneyu(above, below, alternative='two-sided')
            except:
                pval = 1.0

            print(f"P{pctl:<10} {threshold:>10.4f} {len(above):>10} {med_above:>12.2f} {med_below:>12.2f} {diff:>10.2f} {pval:>10.4f}")

            if abs(diff) > abs(best_bqi['diff']):
                best_bqi = {'diff': diff, 'pctl': pctl, 'threshold': threshold, 'n': len(above), 'pval': pval}

    print(f"\n★ Mejor corte BQI_ABS: >= {best_bqi['threshold']:.4f} (P{best_bqi['pctl']})")
    print(f"  Δ mediana: {best_bqi['diff']:.2f}, N={best_bqi['n']}, p={best_bqi['pval']:.4f}")

    # Análisis por deciles
    print("\n--- BQI_ABS por Deciles ---")
    bqi_data['decile'] = pd.qcut(bqi_data['BQI_ABS'], 10, labels=False, duplicates='drop')
    decile_stats = bqi_data.groupby('decile')[TARGET].agg(['count', 'mean', 'median', 'std'])
    print(decile_stats.to_string())

# =============================================================================
# 2. RATIO THETA K1 / THETA K2
# =============================================================================
print("\n" + "=" * 80)
print("2. RATIO THETA K1 / THETA K2")
print("=" * 80)

if 'theta_k1' in df.columns and 'theta_k2' in df.columns:
    # Crear ratio (con epsilon para evitar división por cero)
    epsilon = 1e-6
    df['theta_ratio_k1_k2'] = df['theta_k1'] / (df['theta_k2'].abs() + epsilon)

    theta_data = df[['theta_k1', 'theta_k2', 'theta_ratio_k1_k2', TARGET]].dropna()

    print(f"\nEstadísticas Theta K1:")
    print(f"  Media: {theta_data['theta_k1'].mean():.4f}, Mediana: {theta_data['theta_k1'].median():.4f}")
    print(f"\nEstadísticas Theta K2:")
    print(f"  Media: {theta_data['theta_k2'].mean():.4f}, Mediana: {theta_data['theta_k2'].median():.4f}")
    print(f"\nEstadísticas Ratio Theta K1/K2:")
    print(f"  Media: {theta_data['theta_ratio_k1_k2'].mean():.4f}")
    print(f"  Mediana: {theta_data['theta_ratio_k1_k2'].median():.4f}")
    print(f"  Min/Max: {theta_data['theta_ratio_k1_k2'].min():.4f} / {theta_data['theta_ratio_k1_k2'].max():.4f}")

    # Correlaciones
    print("\n--- Correlaciones con Target ---")
    for col in ['theta_k1', 'theta_k2', 'theta_ratio_k1_k2']:
        r_sp, p_sp = spearmanr(theta_data[col], theta_data[TARGET])
        print(f"  {col}: Spearman r={r_sp:.4f}, p={p_sp:.6f}")

    # Análisis por percentiles del ratio
    print("\n--- Ratio Theta K1/K2 por Percentiles ---")
    print(f"\n{'Percentil':<12} {'Umbral':>10} {'N_arriba':>10} {'Med_arriba':>12} {'Med_abajo':>12} {'Δ_Med':>10} {'p-val':>10}")
    print("-" * 80)

    best_theta = {'diff': 0, 'pctl': None}
    for pctl in percentiles:
        threshold = theta_data['theta_ratio_k1_k2'].quantile(pctl/100)
        above = theta_data[theta_data['theta_ratio_k1_k2'] >= threshold][TARGET]
        below = theta_data[theta_data['theta_ratio_k1_k2'] < threshold][TARGET]

        if len(above) >= MIN_BIN_SIZE and len(below) >= MIN_BIN_SIZE:
            med_above = above.median()
            med_below = below.median()
            diff = med_above - med_below
            try:
                _, pval = mannwhitneyu(above, below, alternative='two-sided')
            except:
                pval = 1.0

            print(f"P{pctl:<10} {threshold:>10.4f} {len(above):>10} {med_above:>12.2f} {med_below:>12.2f} {diff:>10.2f} {pval:>10.4f}")

            if abs(diff) > abs(best_theta['diff']):
                best_theta = {'diff': diff, 'pctl': pctl, 'threshold': threshold, 'n_above': len(above), 'n_below': len(below), 'pval': pval}

    if best_theta['pctl']:
        print(f"\n★ Mejor corte Theta Ratio: >= {best_theta['threshold']:.4f} (P{best_theta['pctl']})")
        print(f"  Δ mediana: {best_theta['diff']:.2f}, p={best_theta['pval']:.4f}")

    # Por deciles
    print("\n--- Ratio Theta K1/K2 por Deciles ---")
    theta_data['decile'] = pd.qcut(theta_data['theta_ratio_k1_k2'], 10, labels=False, duplicates='drop')
    decile_stats = theta_data.groupby('decile')[TARGET].agg(['count', 'mean', 'median', 'std'])
    print(decile_stats.to_string())

# =============================================================================
# 3. GOLDEN CROSS (SMA50 > SMA200)
# =============================================================================
print("\n" + "=" * 80)
print("3. GOLDEN CROSS FILTER (SMA50 > SMA200)")
print("=" * 80)

if 'SPX_Golden_Cross' in df.columns:
    gc_data = df[['SPX_Golden_Cross', TARGET]].dropna()

    golden = gc_data[gc_data['SPX_Golden_Cross'] == 1][TARGET]
    death = gc_data[gc_data['SPX_Golden_Cross'] == 0][TARGET]

    print(f"\n--- Golden Cross (SMA50 > SMA200) ---")
    print(f"  N Golden Cross (1): {len(golden):,}")
    print(f"  N Death Cross (0):  {len(death):,}")

    print(f"\n  Golden Cross ON:  Media={golden.mean():.2f}, Mediana={golden.median():.2f}, Std={golden.std():.2f}")
    print(f"  Golden Cross OFF: Media={death.mean():.2f}, Mediana={death.median():.2f}, Std={death.std():.2f}")

    diff = golden.median() - death.median()
    try:
        stat, pval = mannwhitneyu(golden, death, alternative='two-sided')
    except:
        pval = 1.0

    print(f"\n  Δ mediana (Golden - Death): {diff:.2f}")
    print(f"  Mann-Whitney p-value: {pval:.6f}")

    if pval < 0.05:
        print(f"  ★ SIGNIFICATIVO: {'Golden Cross MEJORA' if diff > 0 else 'Death Cross MEJORA'} el PnL")
    else:
        print(f"  → No significativo estadísticamente")

# También con la diferencia SMA50-SMA200
if 'SPX_SMA50_200_Diff' in df.columns:
    print("\n--- Análisis SMA50_200_Diff continuo ---")
    sma_data = df[['SPX_SMA50_200_Diff', TARGET]].dropna()

    r_sp, p_sp = spearmanr(sma_data['SPX_SMA50_200_Diff'], sma_data[TARGET])
    print(f"  Correlación Spearman: r={r_sp:.4f}, p={p_sp:.6f}")

    # Por signo
    positive = sma_data[sma_data['SPX_SMA50_200_Diff'] > 0][TARGET]
    negative = sma_data[sma_data['SPX_SMA50_200_Diff'] <= 0][TARGET]

    print(f"\n  SMA50 > SMA200 (Diff > 0): N={len(positive):,}, Mediana={positive.median():.2f}")
    print(f"  SMA50 ≤ SMA200 (Diff ≤ 0): N={len(negative):,}, Mediana={negative.median():.2f}")

    diff = positive.median() - negative.median()
    _, pval = mannwhitneyu(positive, negative, alternative='two-sided')
    print(f"  Δ mediana: {diff:.2f}, p={pval:.6f}")

# =============================================================================
# 4. OTROS CRUCES SMA (SMA20 vs SMA50, etc.)
# =============================================================================
print("\n" + "=" * 80)
print("4. OTROS FILTROS SMA")
print("=" * 80)

# Crear cruces adicionales
sma_pairs = [
    ('SPX_SMA7', 'SPX_SMA20', 'SMA7 > SMA20'),
    ('SPX_SMA20', 'SPX_SMA50', 'SMA20 > SMA50'),
    ('SPX_SMA7', 'SPX_SMA50', 'SMA7 > SMA50'),
    ('SPX_SMA20', 'SPX_SMA100', 'SMA20 > SMA100'),
    ('SPX_SMA50', 'SPX_SMA100', 'SMA50 > SMA100'),
    ('SPX_SMA20', 'SPX_SMA200', 'SMA20 > SMA200'),
]

print(f"\n{'Filtro':<20} {'N_True':>8} {'N_False':>8} {'Med_T':>10} {'Med_F':>10} {'Δ_Med':>10} {'p-val':>10} {'Sig':>5}")
print("-" * 95)

sma_results = []
for sma_fast, sma_slow, label in sma_pairs:
    if sma_fast in df.columns and sma_slow in df.columns:
        mask_valid = df[sma_fast].notna() & df[sma_slow].notna() & df[TARGET].notna()
        data = df.loc[mask_valid].copy()

        data['cross'] = (data[sma_fast] > data[sma_slow]).astype(int)

        true_group = data[data['cross'] == 1][TARGET]
        false_group = data[data['cross'] == 0][TARGET]

        if len(true_group) >= MIN_BIN_SIZE and len(false_group) >= MIN_BIN_SIZE:
            med_true = true_group.median()
            med_false = false_group.median()
            diff = med_true - med_false

            try:
                _, pval = mannwhitneyu(true_group, false_group, alternative='two-sided')
            except:
                pval = 1.0

            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

            print(f"{label:<20} {len(true_group):>8} {len(false_group):>8} {med_true:>10.2f} {med_false:>10.2f} {diff:>10.2f} {pval:>10.4f} {sig:>5}")

            sma_results.append({
                'filter': label,
                'n_true': len(true_group),
                'n_false': len(false_group),
                'median_true': med_true,
                'median_false': med_false,
                'diff': diff,
                'pval': pval
            })

# SPX respecto a SMAs individuales
print("\n--- SPX vs SMA individual ---")
print(f"\n{'Filtro':<25} {'N_True':>8} {'N_False':>8} {'Med_T':>10} {'Med_F':>10} {'Δ_Med':>10} {'p-val':>10} {'Sig':>5}")
print("-" * 100)

sma_cols = ['SPX_SMA7', 'SPX_SMA20', 'SPX_SMA50', 'SPX_SMA100', 'SPX_SMA200']
for sma_col in sma_cols:
    if sma_col in df.columns and 'SPX' in df.columns:
        mask = df[sma_col].notna() & df['SPX'].notna() & df[TARGET].notna()
        data = df.loc[mask].copy()

        # SPX > SMA
        above = data[data['SPX'] > data[sma_col]][TARGET]
        below = data[data['SPX'] <= data[sma_col]][TARGET]

        if len(above) >= MIN_BIN_SIZE and len(below) >= MIN_BIN_SIZE:
            med_above = above.median()
            med_below = below.median()
            diff = med_above - med_below

            try:
                _, pval = mannwhitneyu(above, below, alternative='two-sided')
            except:
                pval = 1.0

            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            label = f"SPX > {sma_col.replace('SPX_', '')}"

            print(f"{label:<25} {len(above):>8} {len(below):>8} {med_above:>10.2f} {med_below:>10.2f} {diff:>10.2f} {pval:>10.4f} {sig:>5}")

# =============================================================================
# 5. COMBINACIONES DE FILTROS
# =============================================================================
print("\n" + "=" * 80)
print("5. COMBINACIONES DE FILTROS")
print("=" * 80)

# Crear columnas de filtro
df['golden_cross'] = (df['SPX_Golden_Cross'] == 1) if 'SPX_Golden_Cross' in df.columns else False
df['high_bqi'] = df['BQI_ABS'] >= df['BQI_ABS'].quantile(0.70) if 'BQI_ABS' in df.columns else False
df['high_hv50'] = df['SPX_HV50'] >= 25 if 'SPX_HV50' in df.columns else False
df['spx_below_sma100'] = df['SPX'] < df['SPX_SMA100'] if 'SPX_SMA100' in df.columns else False

combinations = [
    ('Golden Cross + BQI Alto (P70)', df['golden_cross'] & df['high_bqi']),
    ('Golden Cross + HV50 Alto', df['golden_cross'] & df['high_hv50']),
    ('Death Cross + BQI Alto', ~df['golden_cross'] & df['high_bqi']),
    ('Death Cross + HV50 Alto', ~df['golden_cross'] & df['high_hv50']),
    ('SPX < SMA100 + BQI Alto', df['spx_below_sma100'] & df['high_bqi']),
    ('SPX < SMA100 + HV50 Alto', df['spx_below_sma100'] & df['high_hv50']),
]

baseline_median = df[TARGET].median()
print(f"\nBaseline (todo): N={len(df):,}, Mediana={baseline_median:.2f}")

print(f"\n{'Combinación':<35} {'N':>8} {'Mediana':>10} {'Δ vs Base':>12} {'Media':>10} {'Std':>10}")
print("-" * 95)

for label, mask in combinations:
    subset = df.loc[mask & df[TARGET].notna(), TARGET]
    if len(subset) >= MIN_BIN_SIZE:
        med = subset.median()
        diff = med - baseline_median
        print(f"{label:<35} {len(subset):>8} {med:>10.2f} {diff:>+12.2f} {subset.mean():>10.2f} {subset.std():>10.2f}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUMEN DE HALLAZGOS")
print("=" * 80)

print("""
1. BQI_ABS:
   - Correlación positiva con target (valores altos → mejor PnL)
   - Mejor corte identificado en el análisis anterior

2. Ratio Theta K1/K2:
   - Ver correlación y cortes óptimos arriba

3. Golden Cross (SMA50 > SMA200):
   - Ver resultado de significancia arriba

4. Cruces SMA más efectivos:
   - Ver tabla comparativa arriba

5. Mejores combinaciones de filtros:
   - Ver tabla de combinaciones arriba
""")

print("=" * 80)
print("FIN DEL ANÁLISIS EXTENDIDO")
print("=" * 80)
