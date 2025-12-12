"""
Análisis iterativo exhaustivo de correlaciones con PnL_fwd_pts_50_mediana
SIN VARIABLES FUTURAS - Solo información disponible en el momento del trade

Proceso generativo que aprende y mejora en cada iteración
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from itertools import combinations, permutations
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
TARGET = 'PnL_fwd_pts_50_mediana'
MIN_SAMPLES = 100
CORRELATION_THRESHOLD = 0.15  # Umbral más bajo para explorar más
MAX_ITERATIONS = 10  # Más iteraciones
TOP_N_PER_ITERATION = 20  # Top variables a considerar por iteración

print("="*90)
print("ANÁLISIS ITERATIVO GENERATIVO - BÚSQUEDA DE CORRELACIONES CON PNL")
print("EXCLUSIÓN TOTAL DE VARIABLES FUTURAS (fwd, SPX_chg_pct)")
print("="*90)

# ============================================================================
# 1. CARGA Y FILTRADO DE DATOS
# ============================================================================
print("\n[1] Cargando y filtrando datos...")
df = pd.read_csv('combined_mediana.csv')
print(f"   Dataset original: {df.shape}")

# Identificar columnas a EXCLUIR
all_cols = df.columns.tolist()
excluded_cols = []

for col in all_cols:
    col_lower = col.lower()
    # Excluir FWD (excepto target)
    if 'fwd' in col_lower and col != TARGET:
        excluded_cols.append(col)
    # Excluir SPX_chg_pct (son cambios futuros)
    elif 'spx_chg_pct' in col_lower:
        excluded_cols.append(col)

valid_cols = [col for col in all_cols if col not in excluded_cols]
print(f"   Columnas excluidas (futuras): {len(excluded_cols)}")
print(f"   Columnas válidas: {len(valid_cols)}")

df_valid = df[valid_cols].copy()

# Identificar columnas numéricas
numeric_cols = df_valid.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

print(f"   Columnas numéricas disponibles: {len(numeric_cols)}")
print(f"   Primeras 10 variables: {numeric_cols[:10]}")

# ============================================================================
# 2. PREPARACIÓN TRAIN/TEST
# ============================================================================
print("\n[2] Preparando datos...")
df_clean = df_valid.dropna(subset=[TARGET])
print(f"   Rows after removing NaN target: {len(df_clean)}")

df_train, df_test = train_test_split(df_clean, test_size=0.3, random_state=42)
print(f"   Train set: {len(df_train)} rows")
print(f"   Test set: {len(df_test)} rows")

# ============================================================================
# 3. REGISTRO DE CORRELACIONES
# ============================================================================
all_features = {}  # Diccionario para almacenar todas las features generadas

def calculate_and_store_correlation(train_data, test_data, feature_name, feature_values_train,
                                     feature_values_test, formula="", generation=0):
    """Calcula correlación y almacena la feature"""
    try:
        # Agregar temporalmente al dataframe
        train_temp = train_data.copy()
        test_temp = test_data.copy()

        train_temp['_temp_feature'] = feature_values_train
        test_temp['_temp_feature'] = feature_values_test

        # Calcular correlación
        valid_train = train_temp[['_temp_feature', TARGET]].dropna()

        if len(valid_train) >= MIN_SAMPLES:
            train_corr, train_pval = stats.pearsonr(valid_train['_temp_feature'],
                                                     valid_train[TARGET])

            valid_test = test_temp[['_temp_feature', TARGET]].dropna()

            if len(valid_test) >= MIN_SAMPLES:
                test_corr, _ = stats.pearsonr(valid_test['_temp_feature'],
                                              valid_test[TARGET])

                # Almacenar feature
                all_features[feature_name] = {
                    'values_train': feature_values_train,
                    'values_test': feature_values_test,
                    'train_corr': train_corr,
                    'test_corr': test_corr,
                    'abs_train': abs(train_corr),
                    'abs_test': abs(test_corr),
                    'diff': abs(train_corr - test_corr),
                    'pval': train_pval,
                    'n_samples': len(valid_train),
                    'formula': formula,
                    'generation': generation
                }
                return True
    except:
        pass
    return False

# ============================================================================
# 4. GENERACIÓN ITERATIVA DE VARIABLES
# ============================================================================

print("\n" + "="*90)
print("PROCESO ITERATIVO DE GENERACIÓN Y APRENDIZAJE")
print("="*90)

# ITERACIÓN 0: Variables originales
print(f"\n{'='*90}")
print(f"ITERACIÓN 0: Análisis de variables originales")
print(f"{'='*90}")

for col in numeric_cols:
    calculate_and_store_correlation(
        df_train, df_test,
        col,
        df_train[col].values,
        df_test[col].values,
        f"original: {col}",
        generation=0
    )

print(f"Features generadas en iteración 0: {len(all_features)}")

# Mostrar top 10
features_df = pd.DataFrame.from_dict(all_features, orient='index')
# Convertir columnas numéricas
numeric_columns = ['train_corr', 'test_corr', 'abs_train', 'abs_test', 'diff', 'pval', 'n_samples', 'generation']
for col in numeric_columns:
    if col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

top_10 = features_df.nlargest(10, 'abs_train')
print(f"\nTop 10 variables originales:")
for idx, (name, row) in enumerate(top_10.iterrows(), 1):
    print(f"  {idx:2d}. {name:40s} | r_train={row['train_corr']:7.4f} | r_test={row['test_corr']:7.4f}")

# ITERACIONES 1-N: Generación y aprendizaje
for iteration in range(1, MAX_ITERATIONS + 1):
    print(f"\n{'='*90}")
    print(f"ITERACIÓN {iteration}: Generación de variables derivadas")
    print(f"{'='*90}")

    features_before = len(all_features)

    # Obtener las mejores features de la iteración anterior
    features_df = pd.DataFrame.from_dict(all_features, orient='index')
    for col in numeric_columns:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    top_features = features_df.nlargest(TOP_N_PER_ITERATION, 'abs_train')
    top_names = top_features.index.tolist()

    print(f"Trabajando con top {len(top_names)} features de iteraciones previas...")

    # ========================================================================
    # A. TRANSFORMACIONES NO LINEALES
    # ========================================================================
    print(f"\n  [A] Transformaciones no lineales...")
    transformations_count = 0

    for name in top_names[:15]:  # Top 15 para transformaciones
        try:
            values_train = all_features[name]['values_train']
            values_test = all_features[name]['values_test']

            # Log (solo valores positivos)
            if np.all(values_train > 0):
                calculate_and_store_correlation(
                    df_train, df_test,
                    f'log_{name}',
                    np.log(values_train),
                    np.log(values_test),
                    f"log({name})",
                    generation=iteration
                )
                transformations_count += 1

            # Exponencial (solo si valores razonables)
            if np.all(np.abs(values_train) < 10):
                calculate_and_store_correlation(
                    df_train, df_test,
                    f'exp_{name}',
                    np.exp(values_train),
                    np.exp(values_test),
                    f"exp({name})",
                    generation=iteration
                )
                transformations_count += 1

            # Cuadrado
            calculate_and_store_correlation(
                df_train, df_test,
                f'sq_{name}',
                values_train ** 2,
                values_test ** 2,
                f"({name})²",
                generation=iteration
            )
            transformations_count += 1

            # Raíz cuadrada (valores positivos)
            if np.all(values_train >= 0):
                calculate_and_store_correlation(
                    df_train, df_test,
                    f'sqrt_{name}',
                    np.sqrt(values_train),
                    np.sqrt(values_test),
                    f"√({name})",
                    generation=iteration
                )
                transformations_count += 1

            # Cubo
            calculate_and_store_correlation(
                df_train, df_test,
                f'cube_{name}',
                values_train ** 3,
                values_test ** 3,
                f"({name})³",
                generation=iteration
            )
            transformations_count += 1

            # Inversa
            if np.all(values_train != 0):
                calculate_and_store_correlation(
                    df_train, df_test,
                    f'inv_{name}',
                    1 / (values_train + 1e-10),
                    1 / (values_test + 1e-10),
                    f"1/({name})",
                    generation=iteration
                )
                transformations_count += 1

            # Valor absoluto
            calculate_and_store_correlation(
                df_train, df_test,
                f'abs_{name}',
                np.abs(values_train),
                np.abs(values_test),
                f"abs({name})",
                generation=iteration
            )
            transformations_count += 1

            # Signo
            calculate_and_store_correlation(
                df_train, df_test,
                f'sign_{name}',
                np.sign(values_train),
                np.sign(values_test),
                f"sign({name})",
                generation=iteration
            )
            transformations_count += 1

        except:
            continue

    print(f"      Transformaciones generadas: {transformations_count}")

    # ========================================================================
    # B. COMBINACIONES BINARIAS (ratios, productos, sumas, diferencias)
    # ========================================================================
    print(f"\n  [B] Combinaciones binarias...")
    combinations_count = 0

    for name1, name2 in combinations(top_names[:12], 2):  # Top 12, combinaciones de 2
        try:
            v1_train = all_features[name1]['values_train']
            v1_test = all_features[name1]['values_test']
            v2_train = all_features[name2]['values_train']
            v2_test = all_features[name2]['values_test']

            # Ratio
            if np.all(v2_train != 0):
                calculate_and_store_correlation(
                    df_train, df_test,
                    f'ratio_{name1}_{name2}',
                    v1_train / (v2_train + 1e-10),
                    v1_test / (v2_test + 1e-10),
                    f"({name1}) / ({name2})",
                    generation=iteration
                )
                combinations_count += 1

            # Producto
            calculate_and_store_correlation(
                df_train, df_test,
                f'prod_{name1}_{name2}',
                v1_train * v2_train,
                v1_test * v2_test,
                f"({name1}) × ({name2})",
                generation=iteration
            )
            combinations_count += 1

            # Suma
            calculate_and_store_correlation(
                df_train, df_test,
                f'sum_{name1}_{name2}',
                v1_train + v2_train,
                v1_test + v2_test,
                f"({name1}) + ({name2})",
                generation=iteration
            )
            combinations_count += 1

            # Diferencia
            calculate_and_store_correlation(
                df_train, df_test,
                f'diff_{name1}_{name2}',
                v1_train - v2_train,
                v1_test - v2_test,
                f"({name1}) - ({name2})",
                generation=iteration
            )
            combinations_count += 1

            # Promedio ponderado
            calculate_and_store_correlation(
                df_train, df_test,
                f'wavg_{name1}_{name2}',
                0.7 * v1_train + 0.3 * v2_train,
                0.7 * v1_test + 0.3 * v2_test,
                f"0.7×({name1}) + 0.3×({name2})",
                generation=iteration
            )
            combinations_count += 1

            if combinations_count >= 200:  # Limitar para no explotar
                break

        except:
            continue

    print(f"      Combinaciones generadas: {combinations_count}")

    # ========================================================================
    # C. PERCENTILES Y RANGOS
    # ========================================================================
    print(f"\n  [C] Análisis de percentiles...")
    percentiles_count = 0

    percentiles = [10, 25, 50, 75, 90, 95]

    for name in top_names[:10]:  # Top 10 para percentiles
        try:
            values_train = all_features[name]['values_train']
            values_test = all_features[name]['values_test']

            for p in percentiles:
                # Calcular percentil en train
                pct_val = np.percentile(values_train, p)

                # Variable binaria: por encima del percentil
                calculate_and_store_correlation(
                    df_train, df_test,
                    f'{name}_above_p{p}',
                    (values_train > pct_val).astype(float),
                    (values_test > pct_val).astype(float),
                    f"({name} > p{p})",
                    generation=iteration
                )
                percentiles_count += 1

                # Distancia al percentil
                calculate_and_store_correlation(
                    df_train, df_test,
                    f'{name}_dist_p{p}',
                    values_train - pct_val,
                    values_test - pct_val,
                    f"({name} - p{p})",
                    generation=iteration
                )
                percentiles_count += 1

                # Distancia normalizada
                std_val = np.std(values_train)
                if std_val > 0:
                    calculate_and_store_correlation(
                        df_train, df_test,
                        f'{name}_norm_dist_p{p}',
                        (values_train - pct_val) / std_val,
                        (values_test - pct_val) / std_val,
                        f"(({name} - p{p}) / std)",
                        generation=iteration
                    )
                    percentiles_count += 1
        except:
            continue

    print(f"      Variables de percentiles generadas: {percentiles_count}")

    # ========================================================================
    # D. COMBINACIONES TERNARIAS (solo las mejores)
    # ========================================================================
    print(f"\n  [D] Combinaciones ternarias...")
    ternary_count = 0

    for name1, name2, name3 in combinations(top_names[:8], 3):  # Top 8, combinaciones de 3
        try:
            v1_train = all_features[name1]['values_train']
            v1_test = all_features[name1]['values_test']
            v2_train = all_features[name2]['values_train']
            v2_test = all_features[name2]['values_test']
            v3_train = all_features[name3]['values_train']
            v3_test = all_features[name3]['values_test']

            # Promedio ponderado 3 variables
            calculate_and_store_correlation(
                df_train, df_test,
                f'wavg3_{name1}_{name2}_{name3}',
                0.5 * v1_train + 0.3 * v2_train + 0.2 * v3_train,
                0.5 * v1_test + 0.3 * v2_test + 0.2 * v3_test,
                f"0.5×{name1} + 0.3×{name2} + 0.2×{name3}",
                generation=iteration
            )
            ternary_count += 1

            # Producto de 3
            calculate_and_store_correlation(
                df_train, df_test,
                f'prod3_{name1}_{name2}_{name3}',
                v1_train * v2_train * v3_train,
                v1_test * v2_test * v3_test,
                f"{name1} × {name2} × {name3}",
                generation=iteration
            )
            ternary_count += 1

            if ternary_count >= 50:  # Limitar
                break

        except:
            continue

    print(f"      Combinaciones ternarias generadas: {ternary_count}")

    # Resumen de la iteración
    features_after = len(all_features)
    new_features = features_after - features_before
    print(f"\n  TOTAL features nuevas en iteración {iteration}: {new_features}")
    print(f"  TOTAL features acumuladas: {features_after}")

    # Mostrar mejores de esta iteración
    features_df = pd.DataFrame.from_dict(all_features, orient='index')
    for col in numeric_columns:
        if col in features_df.columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    current_gen = features_df[features_df['generation'] == iteration]

    if len(current_gen) > 0:
        top_current = current_gen.nlargest(5, 'abs_train')
        print(f"\n  Top 5 de esta iteración:")
        for idx, (name, row) in enumerate(top_current.iterrows(), 1):
            print(f"    {idx}. r={row['train_corr']:.4f} | {row['formula'][:80]}")

# ============================================================================
# 5. REPORTE FINAL
# ============================================================================
print("\n" + "="*90)
print("REPORTE FINAL")
print("="*90)

features_df = pd.DataFrame.from_dict(all_features, orient='index')
for col in numeric_columns:
    if col in features_df.columns:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')

print(f"\n[A] ESTADÍSTICAS GENERALES")
print("-"*90)
print(f"  Total features generadas: {len(features_df)}")
print(f"  Features por generación:")
for gen in range(MAX_ITERATIONS + 1):
    count = len(features_df[features_df['generation'] == gen])
    print(f"    Generación {gen}: {count} features")

# Filtrar significativas
significant = features_df[features_df['abs_train'] >= CORRELATION_THRESHOLD].copy()
significant = significant.sort_values('abs_train', ascending=False)

print(f"\n  Features significativas (|r| >= {CORRELATION_THRESHOLD}): {len(significant)}")
if len(significant) > 0:
    print(f"  Mejor correlación train: {significant['train_corr'].iloc[0]:.4f}")
    print(f"  Mejor correlación test: {significant.nlargest(1, 'abs_test')['test_corr'].iloc[0]:.4f}")

# TOP 30 Features
print(f"\n[B] TOP 30 FEATURES MÁS CORRELACIONADAS")
print("-"*90)
print(f"{'Rank':<5} {'Train Corr':<12} {'Test Corr':<12} {'Diff':<8} {'Formula':<60}")
print("-"*90)

for idx, (name, row) in enumerate(significant.head(30).iterrows(), 1):
    formula_short = row['formula'][:55] if len(row['formula']) <= 55 else row['formula'][:52] + "..."
    print(f"{idx:<5} {row['train_corr']:<12.4f} {row['test_corr']:<12.4f} {row['diff']:<8.4f} {formula_short}")

# Mejores por robustez (menor diferencia train-test)
print(f"\n[C] TOP 10 MÁS ROBUSTAS (menor diferencia train-test)")
print("-"*90)
robust = significant.nsmallest(10, 'diff')
for idx, (name, row) in enumerate(robust.iterrows(), 1):
    formula_short = row['formula'][:60] if len(row['formula']) <= 60 else row['formula'][:57] + "..."
    print(f"  {idx:2d}. r_train={row['train_corr']:7.4f} | r_test={row['test_corr']:7.4f} | diff={row['diff']:.4f}")
    print(f"      {formula_short}")

# Variables originales vs derivadas
print(f"\n[D] ANÁLISIS POR TIPO")
print("-"*90)
original = significant[significant['formula'].str.startswith('original')]
print(f"  Variables originales: {len(original)}")
if len(original) > 0:
    print(f"    Mejor: {original.iloc[0]['formula']} (r={original.iloc[0]['train_corr']:.4f})")

derived = significant[~significant['formula'].str.startswith('original')]
print(f"  Variables derivadas: {len(derived)}")
if len(derived) > 0:
    print(f"    Mejor: {derived.iloc[0]['formula'][:70]} (r={derived.iloc[0]['train_corr']:.4f})")

# Guardar resultados
output_file = 'correlaciones_iterativas_final.csv'
significant.to_csv(output_file, index=True)
print(f"\n[E] Resultados guardados en: {output_file}")

# Guardar top features completas
top_features_file = 'top_features_completas.csv'
significant.head(50).to_csv(top_features_file, index=True)
print(f"[F] Top 50 features guardadas en: {top_features_file}")

print("\n" + "="*90)
print("ANÁLISIS COMPLETADO")
print("="*90)
