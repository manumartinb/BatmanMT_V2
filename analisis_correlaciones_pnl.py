"""
Análisis avanzado de correlaciones con PnL_fwd_pts_50_mediana
Genera variables derivadas y busca relaciones matemáticas de forma iterativa
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Configuración
TARGET = 'PnL_fwd_pts_50_mediana'
MIN_SAMPLES = 100  # Mínimo de muestras para considerar una correlación válida
CORRELATION_THRESHOLD = 0.3  # Umbral mínimo de correlación absoluta para reportar
MAX_ITERATIONS = 5  # Iteraciones de generación de variables
TOP_N = 30  # Top N correlaciones a reportar

print("="*80)
print("ANÁLISIS DE CORRELACIONES CON PnL_fwd_pts_50_mediana")
print("="*80)

# 1. CARGA DE DATOS
print("\n[1] Cargando datos...")
df = pd.read_csv('combined_mediana.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Total rows: {len(df)}")

# 2. FILTRADO DE COLUMNAS (Excluir FWD excepto target)
print("\n[2] Filtrando columnas...")
all_cols = df.columns.tolist()

# Excluir columnas con 'fwd' excepto el target
valid_cols = [col for col in all_cols if 'fwd' not in col.lower() or col == TARGET]
print(f"   Columnas totales: {len(all_cols)}")
print(f"   Columnas excluidas (fwd): {len(all_cols) - len(valid_cols)}")
print(f"   Columnas válidas: {len(valid_cols)}")

df_valid = df[valid_cols].copy()

# Identificar columnas numéricas
numeric_cols = df_valid.select_dtypes(include=[np.number]).columns.tolist()
if TARGET in numeric_cols:
    numeric_cols.remove(TARGET)

print(f"   Columnas numéricas disponibles: {len(numeric_cols)}")

# 3. LIMPIEZA Y PREPARACIÓN
print("\n[3] Preparando datos...")
# Eliminar filas con target NaN
df_clean = df_valid.dropna(subset=[TARGET])
print(f"   Rows after removing NaN target: {len(df_clean)}")

# Split train/test para evitar overfitting
df_train, df_test = train_test_split(df_clean, test_size=0.3, random_state=42)
print(f"   Train set: {len(df_train)} rows")
print(f"   Test set: {len(df_test)} rows")

# 4. ALMACENAMIENTO DE RESULTADOS
all_correlations = []

def add_correlation(name, train_corr, test_corr, train_pval, n_samples, formula=""):
    """Añade una correlación al registro"""
    all_correlations.append({
        'feature': name,
        'train_corr': train_corr,
        'test_corr': test_corr,
        'train_pval': train_pval,
        'n_samples': n_samples,
        'abs_train_corr': abs(train_corr),
        'abs_test_corr': abs(test_corr),
        'formula': formula
    })

def calculate_correlation(train_data, test_data, col_name, formula=""):
    """Calcula correlación en train y test"""
    # Train
    valid_train = train_data[[col_name, TARGET]].dropna()
    if len(valid_train) >= MIN_SAMPLES:
        train_corr, train_pval = stats.pearsonr(valid_train[col_name], valid_train[TARGET])

        # Test
        valid_test = test_data[[col_name, TARGET]].dropna()
        if len(valid_test) >= MIN_SAMPLES:
            test_corr, _ = stats.pearsonr(valid_test[col_name], valid_test[TARGET])
            add_correlation(col_name, train_corr, test_corr, train_pval, len(valid_train), formula)

# 5. CORRELACIONES DIRECTAS
print("\n[5] Calculando correlaciones directas con variables originales...")
for col in numeric_cols:
    calculate_correlation(df_train, df_test, col, f"original: {col}")

print(f"   Correlaciones calculadas: {len(all_correlations)}")

# 6. PERCENTILES Y CUANTILES
print("\n[6] Generando percentiles y cuantiles de variables...")
percentiles = [10, 25, 50, 75, 90, 95, 99]

for col in numeric_cols[:20]:  # Top 20 variables para evitar explosión combinatoria
    try:
        # Calcular percentiles en train
        for p in percentiles:
            pct_val = df_train[col].quantile(p/100)

            # Crear variables binarias: está por encima/debajo del percentil
            df_train[f'{col}_above_p{p}'] = (df_train[col] > pct_val).astype(int)
            df_test[f'{col}_above_p{p}'] = (df_test[col] > pct_val).astype(int)

            calculate_correlation(df_train, df_test, f'{col}_above_p{p}',
                                f"above_percentile({col}, {p})")

            # Distancia al percentil
            df_train[f'{col}_dist_p{p}'] = df_train[col] - pct_val
            df_test[f'{col}_dist_p{p}'] = df_test[col] - pct_val

            calculate_correlation(df_train, df_test, f'{col}_dist_p{p}',
                                f"distance_to_percentile({col}, {p})")
    except:
        continue

print(f"   Total correlaciones hasta ahora: {len(all_correlations)}")

# 7. COMBINACIONES MATEMÁTICAS DE VARIABLES
print("\n[7] Generando combinaciones matemáticas de variables...")

# Seleccionar las top variables con mejor correlación directa
temp_df = pd.DataFrame(all_correlations)
if len(temp_df) > 0:
    top_vars = temp_df.nlargest(15, 'abs_train_corr')['feature'].tolist()
    top_vars = [v for v in top_vars if v in df_train.columns]

    print(f"   Top {len(top_vars)} variables seleccionadas para combinaciones")

    # Combinaciones de 2 variables
    for var1, var2 in combinations(top_vars[:10], 2):  # Top 10 para evitar explosión
        try:
            # Ratio
            mask = (df_train[var2] != 0) & (df_test[var2] != 0)
            if mask.sum() >= MIN_SAMPLES:
                df_train[f'ratio_{var1}_{var2}'] = df_train[var1] / (df_train[var2] + 1e-10)
                df_test[f'ratio_{var1}_{var2}'] = df_test[var1] / (df_test[var2] + 1e-10)
                calculate_correlation(df_train, df_test, f'ratio_{var1}_{var2}',
                                    f"{var1} / {var2}")

            # Producto
            df_train[f'prod_{var1}_{var2}'] = df_train[var1] * df_train[var2]
            df_test[f'prod_{var1}_{var2}'] = df_test[var1] * df_test[var2]
            calculate_correlation(df_train, df_test, f'prod_{var1}_{var2}',
                                f"{var1} * {var2}")

            # Diferencia
            df_train[f'diff_{var1}_{var2}'] = df_train[var1] - df_train[var2]
            df_test[f'diff_{var1}_{var2}'] = df_test[var1] - df_test[var2]
            calculate_correlation(df_train, df_test, f'diff_{var1}_{var2}',
                                f"{var1} - {var2}")

            # Suma
            df_train[f'sum_{var1}_{var2}'] = df_train[var1] + df_train[var2]
            df_test[f'sum_{var1}_{var2}'] = df_test[var1] + df_test[var2]
            calculate_correlation(df_train, df_test, f'sum_{var1}_{var2}',
                                f"{var1} + {var2}")
        except:
            continue

print(f"   Total correlaciones hasta ahora: {len(all_correlations)}")

# 8. TRANSFORMACIONES NO LINEALES
print("\n[8] Generando transformaciones no lineales...")

top_simple = temp_df[temp_df['formula'].str.startswith('original')].nlargest(10, 'abs_train_corr')['feature'].tolist()

for col in top_simple:
    try:
        # Log (valores positivos)
        if (df_train[col] > 0).all():
            df_train[f'log_{col}'] = np.log(df_train[col])
            df_test[f'log_{col}'] = np.log(df_test[col])
            calculate_correlation(df_train, df_test, f'log_{col}', f"log({col})")

        # Square
        df_train[f'sq_{col}'] = df_train[col] ** 2
        df_test[f'sq_{col}'] = df_test[col] ** 2
        calculate_correlation(df_train, df_test, f'sq_{col}', f"{col}²")

        # Square root (valores positivos)
        if (df_train[col] >= 0).all():
            df_train[f'sqrt_{col}'] = np.sqrt(df_train[col])
            df_test[f'sqrt_{col}'] = np.sqrt(df_test[col])
            calculate_correlation(df_train, df_test, f'sqrt_{col}', f"√{col}")

        # Inverse (valores no cero)
        if (df_train[col] != 0).all():
            df_train[f'inv_{col}'] = 1 / (df_train[col] + 1e-10)
            df_test[f'inv_{col}'] = 1 / (df_test[col] + 1e-10)
            calculate_correlation(df_train, df_test, f'inv_{col}', f"1/{col}")
    except:
        continue

print(f"   Total correlaciones hasta ahora: {len(all_correlations)}")

# 9. ITERACIÓN GENERATIVA
print("\n[9] Proceso iterativo de generación de variables derivadas...")

for iteration in range(MAX_ITERATIONS):
    print(f"\n   Iteración {iteration + 1}/{MAX_ITERATIONS}")

    # Obtener las mejores variables de esta iteración
    temp_df = pd.DataFrame(all_correlations)
    recent_top = temp_df.nlargest(5, 'abs_train_corr')['feature'].tolist()
    recent_top = [v for v in recent_top if v in df_train.columns]

    if len(recent_top) < 2:
        break

    # Generar nuevas combinaciones de las mejores
    generated = 0
    for var1, var2 in combinations(recent_top[:5], 2):
        try:
            # Ratio ponderado
            df_train[f'iter{iteration}_weighted_{var1}_{var2}'] = (
                0.7 * df_train[var1] + 0.3 * df_train[var2]
            )
            df_test[f'iter{iteration}_weighted_{var1}_{var2}'] = (
                0.7 * df_test[var1] + 0.3 * df_test[var2]
            )
            calculate_correlation(
                df_train, df_test,
                f'iter{iteration}_weighted_{var1}_{var2}',
                f"0.7*{var1} + 0.3*{var2}"
            )
            generated += 1

            if generated >= 10:  # Limitar generación
                break
        except:
            continue

    print(f"      Variables generadas: {generated}")

print(f"\n   Total correlaciones finales: {len(all_correlations)}")

# 10. REPORTE FINAL
print("\n" + "="*80)
print("RESULTADOS FINALES")
print("="*80)

results_df = pd.DataFrame(all_correlations)

if len(results_df) > 0:
    # Filtrar por umbral
    significant = results_df[results_df['abs_train_corr'] >= CORRELATION_THRESHOLD].copy()

    # Ordenar por correlación en train
    significant = significant.sort_values('abs_train_corr', ascending=False)

    print(f"\n[A] TOP {TOP_N} CORRELACIONES (|r| >= {CORRELATION_THRESHOLD})")
    print("-"*80)

    for idx, row in significant.head(TOP_N).iterrows():
        print(f"\n{idx+1}. {row['feature']}")
        print(f"   Formula: {row['formula']}")
        print(f"   Train Correlation: {row['train_corr']:.4f} (p-value: {row['train_pval']:.2e})")
        print(f"   Test Correlation:  {row['test_corr']:.4f}")
        print(f"   Diferencia Train-Test: {abs(row['train_corr'] - row['test_corr']):.4f}")
        print(f"   N samples: {row['n_samples']}")

    # Guardar resultados completos
    output_file = 'correlaciones_pnl_resultados.csv'
    significant.to_csv(output_file, index=False)
    print(f"\n[B] Resultados completos guardados en: {output_file}")

    # Estadísticas generales
    print(f"\n[C] ESTADÍSTICAS GENERALES")
    print("-"*80)
    print(f"   Total correlaciones calculadas: {len(results_df)}")
    print(f"   Correlaciones significativas (|r| >= {CORRELATION_THRESHOLD}): {len(significant)}")
    print(f"   Mejor correlación (train): {results_df['abs_train_corr'].max():.4f}")
    print(f"   Mejor correlación (test): {results_df['abs_test_corr'].max():.4f}")

    # Top 5 más robustas (mejor test correlation)
    print(f"\n[D] TOP 5 MÁS ROBUSTAS (mejor correlación en test)")
    print("-"*80)
    robust = results_df.nlargest(5, 'abs_test_corr')
    for idx, row in robust.iterrows():
        print(f"   {row['feature']}: train={row['train_corr']:.4f}, test={row['test_corr']:.4f}")

    # Variables originales vs derivadas
    print(f"\n[E] ANÁLISIS POR TIPO DE VARIABLE")
    print("-"*80)
    original = significant[significant['formula'].str.startswith('original')]
    print(f"   Variables originales: {len(original)} (mejor: {original['abs_train_corr'].max():.4f})")

    percentile_vars = significant[significant['formula'].str.contains('percentile')]
    print(f"   Variables de percentiles: {len(percentile_vars)} (mejor: {percentile_vars['abs_train_corr'].max():.4f if len(percentile_vars)>0 else 0})")

    math_vars = significant[significant['formula'].str.contains('[+\-*/]', regex=True)]
    print(f"   Combinaciones matemáticas: {len(math_vars)} (mejor: {math_vars['abs_train_corr'].max():.4f if len(math_vars)>0 else 0})")

else:
    print("No se encontraron correlaciones significativas")

print("\n" + "="*80)
print("ANÁLISIS COMPLETADO")
print("="*80)
