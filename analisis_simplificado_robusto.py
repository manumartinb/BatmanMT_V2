"""
Análisis simplificado enfocado en features ROBUSTAS e INTERPRETABLES
"""

import pandas as pd
import numpy as np

print("="*90)
print("ANÁLISIS SIMPLIFICADO - FEATURES MÁS ROBUSTAS E INTERPRETABLES")
print("="*90)

# Cargar resultados
df_results = pd.read_csv('correlaciones_iterativas_final.csv', index_col=0)

# Convertir a numérico
numeric_cols = ['train_corr', 'test_corr', 'abs_train', 'abs_test', 'diff', 'pval', 'n_samples', 'generation']
for col in numeric_cols:
    if col in df_results.columns:
        df_results[col] = pd.to_numeric(df_results[col], errors='coerce')

print(f"\nTotal features analizadas: {len(df_results)}")

# ============================================================================
# 1. FILTRAR POR ROBUSTEZ (baja diferencia train-test)
# ============================================================================
print("\n" + "="*90)
print("1. TOP 20 FEATURES MÁS ROBUSTAS (menor diferencia train-test)")
print("="*90)

# Filtrar features con correlación mínima razonable
robust = df_results[df_results['abs_train'] >= 0.15].copy()
robust = robust.sort_values('diff', ascending=True)

print(f"\nFeatures con |r_train| >= 0.15: {len(robust)}")
print(f"\nRank | Train Corr | Test Corr  | Diff    | Formula")
print("-"*90)

for idx, (name, row) in enumerate(robust.head(20).iterrows(), 1):
    formula_short = row['formula'][:65] if len(row['formula']) <= 65 else row['formula'][:62] + "..."
    print(f"{idx:4d} | {row['train_corr']:10.4f} | {row['test_corr']:10.4f} | {row['diff']:7.4f} | {formula_short}")

# ============================================================================
# 2. VARIABLES ORIGINALES
# ============================================================================
print("\n" + "="*90)
print("2. MEJORES VARIABLES ORIGINALES")
print("="*90)

original = df_results[df_results['formula'].str.startswith('original')].copy()
original = original.sort_values('abs_train', ascending=False)

print(f"\nRank | Train Corr | Test Corr  | Robustez | Variable")
print("-"*90)

for idx, (name, row) in enumerate(original.head(20).iterrows(), 1):
    var_name = row['formula'].replace('original: ', '')
    print(f"{idx:4d} | {row['train_corr']:10.4f} | {row['test_corr']:10.4f} | {row['diff']:8.4f} | {var_name}")

# ============================================================================
# 3. FEATURES SIMPLES MÁS INTERPRETABLES
# ============================================================================
print("\n" + "="*90)
print("3. TOP 15 FEATURES SIMPLES MÁS INTERPRETABLES (generación 1-2)")
print("="*90)

# Filtrar solo generaciones tempranas (más simples)
simple = df_results[(df_results['generation'] >= 1) & (df_results['generation'] <= 2)].copy()
simple = simple[simple['abs_train'] >= 0.20]  # Correlación mínima 0.20
simple = simple.sort_values('abs_test', ascending=False)  # Ordenar por test (más robusto)

print(f"\nRank | Train Corr | Test Corr  | Test/Train | Formula")
print("-"*90)

for idx, (name, row) in enumerate(simple.head(15).iterrows(), 1):
    ratio = abs(row['test_corr'] / row['train_corr']) if row['train_corr'] != 0 else 0
    formula_short = row['formula'][:60] if len(row['formula']) <= 60 else row['formula'][:57] + "..."
    print(f"{idx:4d} | {row['train_corr']:10.4f} | {row['test_corr']:10.4f} | {ratio:10.2%} | {formula_short}")

# ============================================================================
# 4. IDENTIFICAR COMPONENTES MÁS IMPORTANTES
# ============================================================================
print("\n" + "="*90)
print("4. ANÁLISIS DE COMPONENTES MÁS FRECUENTES EN TOP FEATURES")
print("="*90)

# Tomar top 100 features
top_100 = df_results.nlargest(100, 'abs_test')

# Analizar qué variables base aparecen más frecuentemente
component_count = {}
components_to_track = ['EarScore', 'BQI_V2_ABS', 'theta_k1', 'theta_k2', 'theta_k3',
                       'SPX_MACD', 'SPX_minus_SMA', 'SPX_ZScore', 'iv_k', 'delta_',
                       'net_credit', 'price_mid', 'SPX_ROC', 'SPX_BB', 'SPX_Williams']

for comp in components_to_track:
    count = sum(1 for formula in top_100['formula'] if comp in formula)
    if count > 0:
        component_count[comp] = count

# Ordenar por frecuencia
sorted_components = sorted(component_count.items(), key=lambda x: x[1], reverse=True)

print(f"\nComponente                | Apariciones en Top 100 | % ")
print("-"*90)
for comp, count in sorted_components:
    pct = count / 100 * 100
    print(f"{comp:25s} | {count:22d} | {pct:5.1f}%")

# ============================================================================
# 5. MEJOR FEATURE POR SIMPLICIDAD Y ROBUSTEZ
# ============================================================================
print("\n" + "="*90)
print("5. MEJOR FEATURE BALANCEANDO SIMPLICIDAD, CORRELACIÓN Y ROBUSTEZ")
print("="*90)

# Filtrar features simples (generación <= 3)
candidates = df_results[df_results['generation'] <= 3].copy()
candidates = candidates[candidates['abs_train'] >= 0.20]

# Calcular score: abs_test * (1 - diff) * simplicidad
# Simplicidad = 1 / (generation + 1)
candidates['simplicity'] = 1 / (candidates['generation'] + 1)
candidates['robustness'] = 1 - candidates['diff']
candidates['score'] = candidates['abs_test'] * candidates['robustness'] * candidates['simplicity']

candidates = candidates.sort_values('score', ascending=False)

print(f"\nTop 10 features considerando simplicidad, correlación y robustez:")
print(f"\nRank | Score  | Train  | Test   | Diff   | Gen | Formula")
print("-"*90)

for idx, (name, row) in enumerate(candidates.head(10).iterrows(), 1):
    formula_short = row['formula'][:55] if len(row['formula']) <= 55 else row['formula'][:52] + "..."
    print(f"{idx:4d} | {row['score']:6.4f} | {row['train_corr']:6.3f} | {row['test_corr']:6.3f} | {row['diff']:6.4f} | {int(row['generation']):3d} | {formula_short}")

# ============================================================================
# 6. RECOMENDACIONES FINALES
# ============================================================================
print("\n" + "="*90)
print("6. RECOMENDACIONES FINALES")
print("="*90)

best_original = original.iloc[0]
best_simple = candidates.iloc[0]
best_robust = robust.iloc[0]

print(f"\n[A] MEJOR VARIABLE ORIGINAL:")
print(f"    {best_original['formula'].replace('original: ', '')}")
print(f"    r_train = {best_original['train_corr']:.4f}")
print(f"    r_test  = {best_original['test_corr']:.4f}")
print(f"    Robustez: diff = {best_original['diff']:.4f}")

print(f"\n[B] MEJOR FEATURE DERIVADA (simplicidad + robustez + correlación):")
print(f"    {best_simple['formula'][:80]}")
print(f"    r_train = {best_simple['train_corr']:.4f}")
print(f"    r_test  = {best_simple['test_corr']:.4f}")
print(f"    Score   = {best_simple['score']:.4f}")
print(f"    Generación: {int(best_simple['generation'])}")

print(f"\n[C] FEATURE MÁS ROBUSTA (menor diferencia train-test):")
print(f"    {best_robust['formula'][:80]}")
print(f"    r_train = {best_robust['train_corr']:.4f}")
print(f"    r_test  = {best_robust['test_corr']:.4f}")
print(f"    Diff    = {best_robust['diff']:.4f}")

print(f"\n[D] COMPONENTES CLAVE IDENTIFICADOS:")
print(f"    Los componentes más importantes (aparecen frecuentemente en top features):")
for idx, (comp, count) in enumerate(sorted_components[:5], 1):
    print(f"    {idx}. {comp} ({count} apariciones en top 100)")

# Guardar recomendaciones
recommendations = pd.DataFrame({
    'Tipo': ['Original', 'Derivada_Balanceada', 'Más_Robusta'],
    'Formula': [
        best_original['formula'].replace('original: ', ''),
        best_simple['formula'],
        best_robust['formula']
    ],
    'Train_Corr': [best_original['train_corr'], best_simple['train_corr'], best_robust['train_corr']],
    'Test_Corr': [best_original['test_corr'], best_simple['test_corr'], best_robust['test_corr']],
    'Diff': [best_original['diff'], best_simple['diff'], best_robust['diff']]
})

recommendations.to_csv('recomendaciones_features.csv', index=False)

# Guardar top robustas
robust.head(30).to_csv('top_30_robustas.csv')

# Guardar top simples
simple.head(30).to_csv('top_30_simples_interpretables.csv')

print(f"\n[E] ARCHIVOS GENERADOS:")
print(f"    - recomendaciones_features.csv")
print(f"    - top_30_robustas.csv")
print(f"    - top_30_simples_interpretables.csv")

print("\n" + "="*90)
print("ANÁLISIS COMPLETADO")
print("="*90)
