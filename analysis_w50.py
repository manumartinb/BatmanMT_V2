#!/usr/bin/env python3
"""
Análisis predictivo para PnL_fwd_pts_50_mediana (Ventana 50)
===============================================================
Basado en el prompt de análisis del documento Glosario_y_Prompt_W50 y W90.docx

Target: PnL_fwd_pts_50_mediana
Restricción anti-leakage: Excluir columnas con "fwd" o "chg" (excepto target)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, permutation_test
import warnings
warnings.filterwarnings('ignore')

# Configuración
CSV_FILE = "combined_BATMAN_mediana_w_stats_w_vix.csv"
TARGET = "PnL_fwd_pts_50_mediana"
MIN_BIN_SIZE = 30  # Mínimo por bin
ALPHA = 0.05
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000

print("=" * 80)
print("ANÁLISIS PREDICTIVO - VENTANA 50 (PnL_fwd_pts_50_mediana)")
print("=" * 80)

# =============================================================================
# (1) CARGA Y VALIDACIONES INICIALES
# =============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 1: VALIDACIONES INICIALES (Calidad de datos)")
print("=" * 80)

df = pd.read_csv(CSV_FILE)
print(f"\n✓ Archivo cargado: {CSV_FILE}")
print(f"  - Filas totales: {len(df):,}")
print(f"  - Columnas totales: {len(df.columns)}")

# Verificar target
if TARGET not in df.columns:
    raise ValueError(f"Target {TARGET} no encontrado en el CSV")
print(f"✓ Target encontrado: {TARGET}")
print(f"  - Valores no-nulos: {df[TARGET].notna().sum():,}")
print(f"  - Media: {df[TARGET].mean():.2f}")
print(f"  - Mediana: {df[TARGET].median():.2f}")
print(f"  - Std: {df[TARGET].std():.2f}")

# Filtrar filas con target válido
df = df[df[TARGET].notna()].copy()
N = len(df)
print(f"\n✓ Filas con target válido: {N:,}")

if N < 100:
    print("⚠ ADVERTENCIA: N < 100, análisis solo descriptivo")

# =============================================================================
# FILTRO ANTI-LEAKAGE
# =============================================================================
print("\n" + "-" * 60)
print("FILTRO ANTI-LEAKAGE")
print("-" * 60)

# Columnas a excluir (contienen "fwd" o "chg", excepto target)
excluded_patterns = ['fwd', 'chg']
excluded_cols = []

for col in df.columns:
    col_lower = col.lower()
    if col == TARGET:
        continue  # Mantener target
    for pattern in excluded_patterns:
        if pattern in col_lower:
            excluded_cols.append(col)
            break

# También excluir columnas claramente de look-ahead
lookhead_cols = ['dia_fwd', 'hora_fwd', 'PnL_fwd', 'PnLDV_fwd', 'net_credit_fwd']
for col in df.columns:
    for pattern in lookhead_cols:
        if pattern in col and col != TARGET and col not in excluded_cols:
            excluded_cols.append(col)

print(f"Columnas excluidas por leakage: {len(excluded_cols)}")

# Columnas permitidas (numéricas)
allowed_cols = [col for col in df.columns if col not in excluded_cols and col != TARGET]
numeric_cols = df[allowed_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"Columnas numéricas permitidas para análisis: {len(numeric_cols)}")

# Identificar columnas core del backtest vs indicadores SPX
core_cols = ['BQI_ABS', 'k1', 'k2', 'k3', 'iv_k1', 'iv_k2', 'iv_k3', 'FF_ATM', 'FF_BAT',
             'delta_total', 'theta_total', 'net_credit', 'net_credit_mediana', 'SPX',
             'Death valley', 'EarL', 'EarR', 'EarScore', 'PnLDV', 'Asym', 'BQI_V2_ABS',
             'BQR_1000', 'RATIO_BATMAN', 'RATIO_UEL_EARS', 'delta_k1', 'delta_k2', 'delta_k3',
             'theta_k1', 'theta_k2', 'theta_k3', 'pnl8000_total']

spx_indicator_cols = [col for col in numeric_cols if col.startswith('SPX_') or col == 'VIX_Close']
core_numeric = [col for col in core_cols if col in numeric_cols]

print(f"  - Core backtest: {len(core_numeric)}")
print(f"  - Indicadores SPX/VIX: {len(spx_indicator_cols)}")

# =============================================================================
# (2) BASELINES - CORRELACIONES
# =============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 2: BASELINES (Correlaciones)")
print("=" * 80)

def calc_correlations(df, features, target):
    """Calcula correlaciones Pearson y Spearman con p-values"""
    results = []
    for col in features:
        if df[col].notna().sum() < 30:
            continue
        mask = df[col].notna() & df[target].notna()
        x = df.loc[mask, col].values
        y = df.loc[mask, target].values

        if len(x) < 30:
            continue

        try:
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            results.append({
                'Feature': col,
                'N': len(x),
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p
            })
        except:
            continue
    return pd.DataFrame(results)

# Calcular correlaciones para todas las features permitidas
print("\nCalculando correlaciones...")
corr_df = calc_correlations(df, numeric_cols, TARGET)

# Aplicar corrección FDR (Benjamini-Hochberg)
from scipy.stats import false_discovery_control
if len(corr_df) > 0:
    # Ordenar por p-value y aplicar FDR
    corr_df = corr_df.sort_values('Spearman_p')
    corr_df['Spearman_p_adj'] = false_discovery_control(corr_df['Spearman_p'].values, method='bh')
    corr_df['Pearson_p_adj'] = false_discovery_control(corr_df['Pearson_p'].values, method='bh')

    # Ordenar por correlación Spearman absoluta
    corr_df['Spearman_abs'] = corr_df['Spearman_r'].abs()
    corr_df = corr_df.sort_values('Spearman_abs', ascending=False)

print("\n--- TOP 25 FEATURES por correlación Spearman (absoluta) ---")
top_corr = corr_df.head(25)[['Feature', 'N', 'Spearman_r', 'Spearman_p_adj', 'Pearson_r', 'Pearson_p_adj']]
print(top_corr.to_string(index=False))

# Filtrar significativas después de FDR
sig_features = corr_df[corr_df['Spearman_p_adj'] < ALPHA]['Feature'].tolist()
print(f"\n✓ Features significativas (FDR < 0.05): {len(sig_features)}")

# =============================================================================
# (3) ANÁLISIS POR CUANTILES Y UMBRALES
# =============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 3: ANÁLISIS POR CUANTILES Y UMBRALES")
print("=" * 80)

def analyze_quantiles(df, feature, target, n_quantiles=5):
    """Analiza el target por cuantiles de una feature"""
    mask = df[feature].notna() & df[target].notna()
    data = df.loc[mask, [feature, target]].copy()

    if len(data) < n_quantiles * MIN_BIN_SIZE:
        return None

    try:
        data['quantile'] = pd.qcut(data[feature], n_quantiles, labels=False, duplicates='drop')
    except:
        return None

    if data['quantile'].nunique() < 3:
        return None

    stats_by_q = data.groupby('quantile')[target].agg(['mean', 'median', 'std', 'count'])

    # Comparar Q1 vs Q5 (bottom vs top)
    q_low = data[data['quantile'] == 0][target].values
    q_high = data[data['quantile'] == data['quantile'].max()][target].values

    if len(q_low) < MIN_BIN_SIZE or len(q_high) < MIN_BIN_SIZE:
        return None

    # Diferencia de medianas
    median_diff = np.median(q_high) - np.median(q_low)

    # Cliff's delta (tamaño de efecto no paramétrico)
    n1, n2 = len(q_low), len(q_high)
    greater = sum(1 for x in q_high for y in q_low if x > y)
    equal = sum(1 for x in q_high for y in q_low if x == y)
    cliffs_d = (greater - (n1*n2 - greater - equal)) / (n1 * n2)

    # Mann-Whitney U test
    try:
        stat, pval = mannwhitneyu(q_high, q_low, alternative='two-sided')
    except:
        pval = 1.0

    return {
        'feature': feature,
        'n_total': len(data),
        'n_q_low': len(q_low),
        'n_q_high': len(q_high),
        'median_q_low': np.median(q_low),
        'median_q_high': np.median(q_high),
        'median_diff': median_diff,
        'cliffs_delta': cliffs_d,
        'mw_pvalue': pval,
        'stats_by_q': stats_by_q
    }

# Analizar top features por cuantiles
print("\nAnalizando features por quintiles...")
quantile_results = []
for feat in corr_df.head(50)['Feature'].values:
    result = analyze_quantiles(df, feat, TARGET, n_quantiles=5)
    if result:
        quantile_results.append(result)

# Ordenar por diferencia de medianas absoluta
quantile_results = sorted(quantile_results, key=lambda x: abs(x['median_diff']), reverse=True)

print("\n--- TOP 20 FEATURES por diferencia Q5-Q1 ---")
print(f"{'Feature':<30} {'N':>6} {'Med_Q1':>10} {'Med_Q5':>10} {'Δ_Med':>10} {'Cliff_d':>8} {'p-val':>10}")
print("-" * 90)
for res in quantile_results[:20]:
    print(f"{res['feature']:<30} {res['n_total']:>6} {res['median_q_low']:>10.2f} {res['median_q_high']:>10.2f} "
          f"{res['median_diff']:>10.2f} {res['cliffs_delta']:>8.3f} {res['mw_pvalue']:>10.4f}")

# =============================================================================
# BÚSQUEDA DE UMBRALES ÓPTIMOS
# =============================================================================
print("\n" + "-" * 60)
print("BÚSQUEDA DE UMBRALES SIMPLES")
print("-" * 60)

def find_best_threshold(df, feature, target, n_splits=20):
    """Encuentra el mejor threshold para separar el target"""
    mask = df[feature].notna() & df[target].notna()
    data = df.loc[mask, [feature, target]].copy()

    if len(data) < 100:
        return None

    # Probar percentiles como umbrales
    best_result = None
    best_diff = 0

    for pct in range(10, 91, 5):
        threshold = data[feature].quantile(pct/100)
        below = data[data[feature] < threshold][target].values
        above = data[data[feature] >= threshold][target].values

        if len(below) < MIN_BIN_SIZE or len(above) < MIN_BIN_SIZE:
            continue

        median_diff = abs(np.median(above) - np.median(below))

        if median_diff > best_diff:
            best_diff = median_diff
            try:
                stat, pval = mannwhitneyu(above, below, alternative='two-sided')
            except:
                pval = 1.0

            best_result = {
                'feature': feature,
                'threshold': threshold,
                'percentile': pct,
                'n_below': len(below),
                'n_above': len(above),
                'median_below': np.median(below),
                'median_above': np.median(above),
                'median_diff': np.median(above) - np.median(below),
                'mw_pvalue': pval
            }

    return best_result

print("\nBuscando umbrales óptimos para top features...")
threshold_results = []
for feat in corr_df.head(50)['Feature'].values:
    result = find_best_threshold(df, feat, TARGET)
    if result:
        threshold_results.append(result)

threshold_results = sorted(threshold_results, key=lambda x: abs(x['median_diff']), reverse=True)

print("\n--- TOP 15 REGLAS POR UMBRAL ---")
print(f"{'Feature':<25} {'Umbral':>10} {'Pctl':>5} {'n<':>6} {'n≥':>6} {'Med<':>10} {'Med≥':>10} {'Δ':>10} {'p-val':>10}")
print("-" * 110)
for res in threshold_results[:15]:
    print(f"{res['feature']:<25} {res['threshold']:>10.2f} {res['percentile']:>5} {res['n_below']:>6} {res['n_above']:>6} "
          f"{res['median_below']:>10.2f} {res['median_above']:>10.2f} {res['median_diff']:>10.2f} {res['mw_pvalue']:>10.4f}")

# =============================================================================
# (3.VIX) BLOQUE ESPECIAL VIX
# =============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 3.VIX: ANÁLISIS ESPECIAL VIX_Close")
print("=" * 80)

if 'VIX_Close' in df.columns and df['VIX_Close'].notna().sum() > 100:
    vix_data = df[['VIX_Close', TARGET, 'dia']].dropna(subset=['VIX_Close', TARGET]).copy()

    # Estadísticas básicas VIX
    print(f"\nVIX_Close estadísticas:")
    print(f"  - N: {len(vix_data):,}")
    print(f"  - Media: {vix_data['VIX_Close'].mean():.2f}")
    print(f"  - Mediana: {vix_data['VIX_Close'].median():.2f}")
    print(f"  - Min/Max: {vix_data['VIX_Close'].min():.2f} / {vix_data['VIX_Close'].max():.2f}")

    # Correlación VIX con target
    r_spearman, p_spearman = spearmanr(vix_data['VIX_Close'], vix_data[TARGET])
    r_pearson, p_pearson = pearsonr(vix_data['VIX_Close'], vix_data[TARGET])
    print(f"\nCorrelación VIX_Close vs {TARGET}:")
    print(f"  - Spearman: {r_spearman:.4f} (p={p_spearman:.4f})")
    print(f"  - Pearson: {r_pearson:.4f} (p={p_pearson:.4f})")

    # Análisis por régimen VIX (quintiles)
    vix_result = analyze_quantiles(df, 'VIX_Close', TARGET, n_quantiles=5)
    if vix_result:
        print(f"\nAnálisis por quintiles VIX:")
        print(vix_result['stats_by_q'])
        print(f"\nDiferencia Q5-Q1: {vix_result['median_diff']:.2f}")
        print(f"Cliff's delta: {vix_result['cliffs_delta']:.3f}")

    # Regímenes VIX
    print("\n--- Regímenes VIX (percentiles) ---")
    for pct_low, pct_high, label in [(0, 20, 'VIX Bajo (<P20)'),
                                      (20, 80, 'VIX Normal (P20-P80)'),
                                      (80, 100, 'VIX Alto (>P80)')]:
        vix_low = vix_data['VIX_Close'].quantile(pct_low/100)
        vix_high = vix_data['VIX_Close'].quantile(pct_high/100)
        mask = (vix_data['VIX_Close'] >= vix_low) & (vix_data['VIX_Close'] < vix_high)
        subset = vix_data.loc[mask, TARGET]
        if len(subset) >= MIN_BIN_SIZE:
            print(f"  {label}: N={len(subset):,}, Mediana PnL={subset.median():.2f}, "
                  f"Media={subset.mean():.2f}, Std={subset.std():.2f}")
else:
    print("⚠ VIX_Close no disponible o insuficientes datos")

# =============================================================================
# (4) FEATURE ENGINEERING PARSIMONIOSO
# =============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 4: FEATURE ENGINEERING")
print("=" * 80)

# Crear features derivadas
df_eng = df.copy()

# Ratios seguros
epsilon = 1e-6

# 1. Ratio delta/theta
if 'delta_total' in df.columns and 'theta_total' in df.columns:
    df_eng['delta_theta_ratio'] = df_eng['delta_total'] / (df_eng['theta_total'].abs() + epsilon)

# 2. IV spread
if 'iv_k1' in df.columns and 'iv_k2' in df.columns:
    df_eng['iv_spread_k1_k2'] = df_eng['iv_k1'] - df_eng['iv_k2']

# 3. Strike spread normalizado
if 'k1' in df.columns and 'k2' in df.columns and 'SPX' in df.columns:
    df_eng['strike_spread_norm'] = (df_eng['k2'] - df_eng['k1']) / (df_eng['SPX'] + epsilon)

# 4. Distancia a SMA200 normalizada
if 'SPX_minus_SMA200' in df.columns and 'SPX' in df.columns:
    df_eng['SPX_SMA200_pct'] = df_eng['SPX_minus_SMA200'] / (df_eng['SPX'] + epsilon) * 100

# 5. VIX z-score (si hay datos)
if 'VIX_Close' in df.columns:
    vix_median = df_eng['VIX_Close'].median()
    vix_mad = (df_eng['VIX_Close'] - vix_median).abs().median()
    df_eng['VIX_zscore_robust'] = (df_eng['VIX_Close'] - vix_median) / (vix_mad * 1.4826 + epsilon)

# 6. RSI extremo
if 'SPX_RSI14' in df.columns:
    df_eng['RSI_extreme'] = ((df_eng['SPX_RSI14'] < 30) | (df_eng['SPX_RSI14'] > 70)).astype(int)

# 7. Golden/Death cross ya existe, crear diferencia normalizada
if 'SPX_SMA50_200_Diff' in df.columns and 'SPX' in df.columns:
    df_eng['SMA50_200_Diff_pct'] = df_eng['SPX_SMA50_200_Diff'] / (df_eng['SPX'] + epsilon) * 100

# 8. Ratio net_credit vs SPX
if 'net_credit_mediana' in df.columns and 'SPX' in df.columns:
    df_eng['credit_spx_ratio'] = df_eng['net_credit_mediana'] / (df_eng['SPX'] + epsilon) * 10000

# Lista de nuevas features
new_features = ['delta_theta_ratio', 'iv_spread_k1_k2', 'strike_spread_norm',
                'SPX_SMA200_pct', 'VIX_zscore_robust', 'RSI_extreme',
                'SMA50_200_Diff_pct', 'credit_spx_ratio']

new_features = [f for f in new_features if f in df_eng.columns]

print(f"\nFeatures derivadas creadas: {len(new_features)}")
for feat in new_features:
    n_valid = df_eng[feat].notna().sum()
    print(f"  - {feat}: N={n_valid:,}")

# Calcular correlaciones de nuevas features
print("\nCorrelaciones de features derivadas:")
for feat in new_features:
    mask = df_eng[feat].notna() & df_eng[TARGET].notna()
    if mask.sum() < 30:
        continue
    r_sp, p_sp = spearmanr(df_eng.loc[mask, feat], df_eng.loc[mask, TARGET])
    print(f"  {feat}: Spearman r={r_sp:.4f}, p={p_sp:.4f}")

# =============================================================================
# (5) VALIDACIÓN OOS (Time Series Split)
# =============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 5: VALIDACIÓN OUT-OF-SAMPLE")
print("=" * 80)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Preparar datos
df_eng = df_eng.sort_values('dia').reset_index(drop=True)

# Top features por correlación + features derivadas
top_features = list(corr_df.head(20)['Feature'].values) + new_features
top_features = [f for f in top_features if f in df_eng.columns and f != TARGET]
top_features = list(set(top_features))

# Eliminar filas con NaN en features importantes
df_model = df_eng[[TARGET, 'dia'] + top_features].dropna().copy()
print(f"\nDatos para modelado: {len(df_model):,} filas, {len(top_features)} features")

if len(df_model) >= 200:
    X = df_model[top_features].values
    y = df_model[TARGET].values

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)

    oos_spearman = []
    oos_mae = []
    top_decile_lift = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Escalar
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modelo Ridge simple
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Métricas OOS
        sp_r, _ = spearmanr(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))

        # Top decile lift
        top_decile_idx = np.argsort(y_pred)[-len(y_pred)//10:]
        top_decile_actual = y_test[top_decile_idx].mean()
        overall_mean = y_test.mean()
        lift = top_decile_actual - overall_mean

        oos_spearman.append(sp_r)
        oos_mae.append(mae)
        top_decile_lift.append(lift)

        print(f"Fold {fold+1}: Spearman OOS={sp_r:.4f}, MAE={mae:.2f}, Top-decile lift={lift:.2f}")

    print(f"\n--- RESUMEN OOS ---")
    print(f"Spearman OOS medio: {np.mean(oos_spearman):.4f} ± {np.std(oos_spearman):.4f}")
    print(f"MAE OOS medio: {np.mean(oos_mae):.2f} ± {np.std(oos_mae):.2f}")
    print(f"Top-decile lift medio: {np.mean(top_decile_lift):.2f} ± {np.std(top_decile_lift):.2f}")

    # Importancia de features (última iteración)
    print("\n--- IMPORTANCIA DE FEATURES (Ridge, último fold) ---")
    feature_imp = pd.DataFrame({
        'Feature': top_features,
        'Coef': model.coef_,
        'Abs_Coef': np.abs(model.coef_)
    }).sort_values('Abs_Coef', ascending=False)
    print(feature_imp.head(15).to_string(index=False))
else:
    print("⚠ Insuficientes datos para validación OOS (N < 200)")

# =============================================================================
# (6) RESUMEN EJECUTIVO Y HALLAZGOS
# =============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 6: RESUMEN EJECUTIVO")
print("=" * 80)

print(f"""
TARGET: {TARGET}
N total analizado: {N:,}

HALLAZGOS PRINCIPALES:
""")

# Top correlaciones
print("1. TOP 5 FEATURES POR CORRELACIÓN SPEARMAN:")
for i, row in corr_df.head(5).iterrows():
    sig = "***" if row['Spearman_p_adj'] < 0.001 else "**" if row['Spearman_p_adj'] < 0.01 else "*" if row['Spearman_p_adj'] < 0.05 else ""
    print(f"   - {row['Feature']}: r={row['Spearman_r']:.4f} {sig} (N={row['N']:,})")

# Top reglas por umbral
print("\n2. TOP 5 REGLAS POR UMBRAL (diferencia de medianas):")
for i, res in enumerate(threshold_results[:5]):
    direction = ">" if res['median_above'] > res['median_below'] else "<"
    print(f"   - {res['feature']} {direction} {res['threshold']:.2f} (P{res['percentile']}): "
          f"Δ mediana = {res['median_diff']:.2f}")

# VIX insight
if 'VIX_Close' in df.columns:
    print("\n3. VIX INSIGHT:")
    vix_corr = corr_df[corr_df['Feature'] == 'VIX_Close']
    if len(vix_corr) > 0:
        print(f"   - Correlación VIX con target: r={vix_corr.iloc[0]['Spearman_r']:.4f}")

# OOS performance
if len(df_model) >= 200:
    print("\n4. CAPACIDAD PREDICTIVA (OOS):")
    print(f"   - Spearman OOS: {np.mean(oos_spearman):.4f} ± {np.std(oos_spearman):.4f}")
    if np.mean(oos_spearman) > 0.1:
        print("   → Señal moderada detectada")
    elif np.mean(oos_spearman) > 0.05:
        print("   → Señal débil detectada")
    else:
        print("   → Señal muy débil o inexistente")

print("\n" + "=" * 80)
print("FIN DEL ANÁLISIS")
print("=" * 80)

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
# Guardar correlaciones
corr_df.to_csv('analysis_w50_correlations.csv', index=False)

# Guardar reglas de umbral
threshold_df = pd.DataFrame(threshold_results[:30])
threshold_df.to_csv('analysis_w50_thresholds.csv', index=False)

print("\n✓ Resultados guardados en:")
print("  - analysis_w50_correlations.csv")
print("  - analysis_w50_thresholds.csv")
