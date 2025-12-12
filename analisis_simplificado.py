"""
Análisis simplificado de correlaciones
Enfoque en variables originales y combinaciones simples más interpretables
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

TARGET = 'PnL_fwd_pts_50_mediana'
MIN_SAMPLES = 100

print("="*80)
print("ANÁLISIS SIMPLIFICADO - VARIABLES MÁS INTERPRETABLES")
print("="*80)

# Cargar datos
df = pd.read_csv('combined_mediana.csv')

# Filtrar columnas FWD
all_cols = df.columns.tolist()
valid_cols = [col for col in all_cols if 'fwd' not in col.lower() or col == TARGET]
df_valid = df[valid_cols].copy()

# Limpiar y split
df_clean = df_valid.dropna(subset=[TARGET])
df_train, df_test = train_test_split(df_clean, test_size=0.3, random_state=42)

print(f"\nDataset: {len(df_train)} train, {len(df_test)} test")

# Identificar columnas numéricas
numeric_cols = df_valid.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

# ============================================================================
# 1. CORRELACIONES DIRECTAS (VARIABLES ORIGINALES)
# ============================================================================
print("\n" + "="*80)
print("1. TOP 20 VARIABLES ORIGINALES CON MEJOR CORRELACIÓN")
print("="*80)

original_corrs = []
for col in numeric_cols:
    valid_train = df_train[[col, TARGET]].dropna()
    if len(valid_train) >= MIN_SAMPLES:
        try:
            train_corr, train_pval = stats.pearsonr(valid_train[col], valid_train[TARGET])

            valid_test = df_test[[col, TARGET]].dropna()
            if len(valid_test) >= MIN_SAMPLES:
                test_corr, _ = stats.pearsonr(valid_test[col], valid_test[TARGET])

                original_corrs.append({
                    'variable': col,
                    'train_corr': train_corr,
                    'test_corr': test_corr,
                    'abs_train': abs(train_corr),
                    'abs_test': abs(test_corr),
                    'diff': abs(train_corr - test_corr),
                    'pval': train_pval
                })
        except:
            continue

df_orig = pd.DataFrame(original_corrs).sort_values('abs_train', ascending=False)

print("\nRank | Variable                          | Train Corr | Test Corr | Diff   | P-value")
print("-"*90)
for idx, row in df_orig.head(20).iterrows():
    print(f"{idx+1:4d} | {row['variable']:35s} | {row['train_corr']:10.4f} | {row['test_corr']:9.4f} | {row['diff']:6.4f} | {row['pval']:.2e}")

# ============================================================================
# 2. MEJORES COMBINACIONES SIMPLES (2 VARIABLES)
# ============================================================================
print("\n" + "="*80)
print("2. TOP 15 COMBINACIONES MATEMÁTICAS SIMPLES")
print("="*80)

# Tomar las top 8 variables originales
top_vars = df_orig.head(8)['variable'].tolist()
simple_combinations = []

from itertools import combinations

for var1, var2 in combinations(top_vars, 2):
    try:
        # RATIO
        df_train[f'temp'] = df_train[var1] / (df_train[var2].abs() + 1e-10)
        df_test[f'temp'] = df_test[var1] / (df_test[var2].abs() + 1e-10)

        valid_train = df_train[['temp', TARGET]].dropna()
        if len(valid_train) >= MIN_SAMPLES:
            train_corr, _ = stats.pearsonr(valid_train['temp'], valid_train[TARGET])
            valid_test = df_test[['temp', TARGET]].dropna()
            if len(valid_test) >= MIN_SAMPLES:
                test_corr, _ = stats.pearsonr(valid_test['temp'], valid_test[TARGET])
                simple_combinations.append({
                    'formula': f'{var1} / {var2}',
                    'train': train_corr,
                    'test': test_corr,
                    'abs_train': abs(train_corr),
                    'diff': abs(train_corr - test_corr)
                })

        # PRODUCTO
        df_train['temp'] = df_train[var1] * df_train[var2]
        df_test['temp'] = df_test[var1] * df_test[var2]

        valid_train = df_train[['temp', TARGET]].dropna()
        if len(valid_train) >= MIN_SAMPLES:
            train_corr, _ = stats.pearsonr(valid_train['temp'], valid_train[TARGET])
            valid_test = df_test[['temp', TARGET]].dropna()
            if len(valid_test) >= MIN_SAMPLES:
                test_corr, _ = stats.pearsonr(valid_test['temp'], valid_test[TARGET])
                simple_combinations.append({
                    'formula': f'{var1} * {var2}',
                    'train': train_corr,
                    'test': test_corr,
                    'abs_train': abs(train_corr),
                    'diff': abs(train_corr - test_corr)
                })

        # DIFERENCIA
        df_train['temp'] = df_train[var1] - df_train[var2]
        df_test['temp'] = df_test[var1] - df_test[var2]

        valid_train = df_train[['temp', TARGET]].dropna()
        if len(valid_train) >= MIN_SAMPLES:
            train_corr, _ = stats.pearsonr(valid_train['temp'], valid_train[TARGET])
            valid_test = df_test[['temp', TARGET]].dropna()
            if len(valid_test) >= MIN_SAMPLES:
                test_corr, _ = stats.pearsonr(valid_test['temp'], valid_test[TARGET])
                simple_combinations.append({
                    'formula': f'{var1} - {var2}',
                    'train': train_corr,
                    'test': test_corr,
                    'abs_train': abs(train_corr),
                    'diff': abs(train_corr - test_corr)
                })
    except:
        continue

df_simple = pd.DataFrame(simple_combinations).sort_values('abs_train', ascending=False)

print("\nRank | Formula                                              | Train Corr | Test Corr | Diff")
print("-"*100)
for idx, row in df_simple.head(15).iterrows():
    print(f"{idx+1:4d} | {row['formula']:55s} | {row['train']:10.4f} | {row['test']:9.4f} | {row['diff']:6.4f}")

# ============================================================================
# 3. ANÁLISIS POR PERCENTILES DE LAS TOP VARIABLES
# ============================================================================
print("\n" + "="*80)
print("3. ANÁLISIS DE PERCENTILES (TOP 5 VARIABLES)")
print("="*80)

for var in top_vars[:5]:
    print(f"\n{var}:")
    print("-" * 60)

    # Calcular cuartiles en train
    q25 = df_train[var].quantile(0.25)
    q50 = df_train[var].quantile(0.50)
    q75 = df_train[var].quantile(0.75)
    q90 = df_train[var].quantile(0.90)

    # Calcular PnL medio en cada rango
    train_data = df_train[[var, TARGET]].dropna()
    test_data = df_test[[var, TARGET]].dropna()

    # Rangos
    ranges = [
        ('Q1 (0-25%)', train_data[var] <= q25, test_data[var] <= q25),
        ('Q2 (25-50%)', (train_data[var] > q25) & (train_data[var] <= q50),
                        (test_data[var] > q25) & (test_data[var] <= q50)),
        ('Q3 (50-75%)', (train_data[var] > q50) & (train_data[var] <= q75),
                        (test_data[var] > q50) & (test_data[var] <= q75)),
        ('Q4 (75-90%)', (train_data[var] > q75) & (train_data[var] <= q90),
                        (test_data[var] > q75) & (test_data[var] <= q90)),
        ('Q5 (90-100%)', train_data[var] > q90, test_data[var] > q90)
    ]

    print(f"  Rango        | Train PnL (mean) | Test PnL (mean) | N train | N test")
    print(f"  " + "-" * 70)

    for range_name, train_mask, test_mask in ranges:
        train_mean = train_data[train_mask][TARGET].mean()
        test_mean = test_data[test_mask][TARGET].mean()
        n_train = train_mask.sum()
        n_test = test_mask.sum()

        print(f"  {range_name:12s} | {train_mean:16.2f} | {test_mean:15.2f} | {n_train:7d} | {n_test:6d}")

# ============================================================================
# 4. RESUMEN EJECUTIVO
# ============================================================================
print("\n" + "="*80)
print("4. RESUMEN EJECUTIVO")
print("="*80)

print("\n[A] VARIABLES ORIGINALES MÁS PREDICTIVAS:")
print("-" * 60)
for idx, row in df_orig.head(5).iterrows():
    direction = "Positiva" if row['train_corr'] > 0 else "Negativa"
    print(f"  {idx+1}. {row['variable']}")
    print(f"     Correlación: {row['train_corr']:.4f} ({direction})")
    print(f"     Robustez (train-test diff): {row['diff']:.4f}")
    print()

print("\n[B] MEJORES COMBINACIONES SIMPLES:")
print("-" * 60)
for idx, row in df_simple.head(3).iterrows():
    direction = "Positiva" if row['train'] > 0 else "Negativa"
    print(f"  {idx+1}. {row['formula']}")
    print(f"     Correlación: {row['train']:.4f} ({direction})")
    print(f"     Robustez (train-test diff): {row['diff']:.4f}")
    print()

print("\n[C] HALLAZGOS CLAVE:")
print("-" * 60)

# Identificar las familias de variables más importantes
var_families = {}
for var in df_orig.head(10)['variable'].tolist():
    # Extraer prefijos comunes
    if 'SPX' in var:
        family = 'SPX (Market)'
    elif 'BQI' in var or 'BQR' in var:
        family = 'Batman Quality Indicators'
    elif 'iv_' in var or 'IV' in var:
        family = 'Implied Volatility'
    elif 'price_' in var:
        family = 'Option Prices'
    elif 'delta_' in var or 'theta_' in var:
        family = 'Greeks'
    elif 'net_credit' in var:
        family = 'Credit Metrics'
    elif 'FF_' in var:
        family = 'Factor Feeder'
    else:
        family = 'Other'

    var_families[family] = var_families.get(family, 0) + 1

print("  Familias de variables más importantes:")
for family, count in sorted(var_families.items(), key=lambda x: x[1], reverse=True):
    print(f"    - {family}: {count} variables en top 10")

# Calcular correlación promedio de las top variables
avg_corr = df_orig.head(10)['abs_train'].mean()
print(f"\n  Correlación promedio (top 10): {avg_corr:.4f}")

# Mejor correlación encontrada
best = df_orig.iloc[0]
print(f"\n  Mejor predictor individual: {best['variable']}")
print(f"    Correlación train: {best['train_corr']:.4f}")
print(f"    Correlación test: {best['test_corr']:.4f}")

# Guardar resultados simplificados
df_orig.to_csv('correlaciones_simples_originales.csv', index=False)
df_simple.to_csv('correlaciones_simples_combinaciones.csv', index=False)

print("\n[D] ARCHIVOS GENERADOS:")
print("-" * 60)
print("  - correlaciones_simples_originales.csv")
print("  - correlaciones_simples_combinaciones.csv")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
