"""
Análisis Estadístico Completo: Correlaciones de Etiquetas de Ventas
Análisis de correlaciones entre puntos forward de PnL y variables driver
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos profesionales
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AnalisisCorrelacionEtiquetasVentas:
    """Análisis completo de correlaciones para etiquetas de ventas y PnL"""

    def __init__(self, csv_path):
        """Inicializar con ruta del archivo de datos"""
        self.csv_path = csv_path
        self.df = None
        self.pnl_columns = []
        self.driver_columns = []
        self.results = {}

    @staticmethod
    def interpretar_correlacion(corr_valor):
        """Interpreta la fuerza de una correlación"""
        abs_corr = abs(corr_valor)
        if abs_corr < 0.20:
            return "Muy Débil", "🔵"
        elif abs_corr < 0.40:
            return "Débil", "🟢"
        elif abs_corr < 0.60:
            return "Moderada", "🟡"
        elif abs_corr < 0.80:
            return "Fuerte", "🟠"
        else:
            return "Muy Fuerte", "🔴"

    @staticmethod
    def interpretar_significancia(p_valor):
        """Interpreta el nivel de significancia estadística"""
        if p_valor < 0.001:
            return "Altamente Significativa (p < 0.001)", "***"
        elif p_valor < 0.01:
            return "Muy Significativa (p < 0.01)", "**"
        elif p_valor < 0.05:
            return "Significativa (p < 0.05)", "*"
        else:
            return "No Significativa (p ≥ 0.05)", "ns"

    def cargar_datos(self):
        """Cargar y preparar dataset"""
        print("=" * 80)
        print("CARGANDO DATASET")
        print("=" * 80)

        self.df = pd.read_csv(self.csv_path)
        print(f"\n✓ Dataset cargado exitosamente")
        print(f"  Dimensiones: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas")

        # Identificar columnas PnL (PnL_fwd_pts_*_mediana)
        self.pnl_columns = [col for col in self.df.columns if 'PnL_fwd_pts_' in col and '_mediana' in col]
        # Ordenar por ventana (01, 05, 25, 50, 90)
        def extraer_ventana(col):
            parts = col.split('_')
            for i, part in enumerate(parts):
                if part == 'pts' and i + 1 < len(parts):
                    return int(parts[i + 1])
            return 0
        self.pnl_columns = sorted(self.pnl_columns, key=extraer_ventana)

        # Columnas driver
        self.driver_columns = ['LABEL_GENERAL_SCORE', 'BQI_ABS', 'FF_ATM',
                               'delta_total', 'theta_total', 'FF_BAT']

        # Filtrar columnas disponibles
        self.driver_columns = [col for col in self.driver_columns if col in self.df.columns]

        print(f"\n  Variables PnL ({len(self.pnl_columns)}):")
        for col in self.pnl_columns:
            print(f"    - {col}")

        print(f"\n  Variables Driver ({len(self.driver_columns)}):")
        for col in self.driver_columns:
            print(f"    - {col}")

        # Eliminar filas con todos NaN en drivers o PnL
        filas_iniciales = len(self.df)
        self.df = self.df.dropna(subset=self.driver_columns + self.pnl_columns, how='all')
        filas_finales = len(self.df)

        print(f"\n  Filas después de limpieza: {filas_finales:,} (eliminadas {filas_iniciales - filas_finales:,} filas vacías)")

        return self

    def estadisticas_descriptivas(self):
        """Generar estadísticas descriptivas para todas las variables"""
        print("\n" + "=" * 80)
        print("1. ESTADÍSTICAS DESCRIPTIVAS")
        print("=" * 80)

        # Estadísticas de drivers
        print("\n>>> VARIABLES DRIVER <<<\n")
        driver_stats = self.df[self.driver_columns].describe().T
        driver_stats['count'] = driver_stats['count'].astype(int)
        print(driver_stats.to_string())

        # Estadísticas de PnL
        print("\n>>> VARIABLES PNL FORWARD POINTS <<<\n")
        pnl_stats = self.df[self.pnl_columns].describe().T
        pnl_stats['count'] = pnl_stats['count'].astype(int)
        print(pnl_stats.to_string())

        self.results['estadisticas_descriptivas'] = {
            'drivers': driver_stats,
            'pnl': pnl_stats
        }

        return self

    def analisis_correlaciones(self):
        """Calcular correlaciones de Pearson y Spearman con p-valores"""
        print("\n" + "=" * 80)
        print("2. ANÁLISIS DE CORRELACIONES CON PNL")
        print("=" * 80)

        # Correlación de Pearson
        pearson_corr = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)
        pearson_pval = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)

        # Correlación de Spearman
        spearman_corr = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)
        spearman_pval = pd.DataFrame(index=self.driver_columns, columns=self.pnl_columns)

        for driver in self.driver_columns:
            for pnl in self.pnl_columns:
                # Eliminar pares con NaN
                mask = self.df[[driver, pnl]].notna().all(axis=1)
                x = self.df.loc[mask, driver]
                y = self.df.loc[mask, pnl]

                if len(x) > 2:
                    # Pearson
                    p_corr, p_pval = stats.pearsonr(x, y)
                    pearson_corr.loc[driver, pnl] = p_corr
                    pearson_pval.loc[driver, pnl] = p_pval

                    # Spearman
                    s_corr, s_pval = stats.spearmanr(x, y)
                    spearman_corr.loc[driver, pnl] = s_corr
                    spearman_pval.loc[driver, pnl] = s_pval

        # Convertir a numérico
        pearson_corr = pearson_corr.astype(float)
        pearson_pval = pearson_pval.astype(float)
        spearman_corr = spearman_corr.astype(float)
        spearman_pval = spearman_pval.astype(float)

        print("\n>>> CORRELACIÓN DE PEARSON (Lineal) <<<\n")
        print(pearson_corr.to_string())

        print("\n>>> P-VALORES DE PEARSON <<<\n")
        print(pearson_pval.to_string())

        print("\n>>> CORRELACIÓN DE SPEARMAN (Monotónica) <<<\n")
        print(spearman_corr.to_string())

        print("\n>>> P-VALORES DE SPEARMAN <<<\n")
        print(spearman_pval.to_string())

        # Interpretación de correlaciones
        print("\n>>> INTERPRETACIÓN DE CALIDAD DE CORRELACIONES <<<\n")
        print("Escala de interpretación:")
        print("  🔵 Muy Débil:  |r| < 0.20")
        print("  🟢 Débil:      0.20 ≤ |r| < 0.40")
        print("  🟡 Moderada:   0.40 ≤ |r| < 0.60")
        print("  🟠 Fuerte:     0.60 ≤ |r| < 0.80")
        print("  🔴 Muy Fuerte: |r| ≥ 0.80")
        print("\nSignificancia estadística:")
        print("  *** p < 0.001 (Altamente significativa)")
        print("  **  p < 0.01  (Muy significativa)")
        print("  *   p < 0.05  (Significativa)")
        print("  ns  p ≥ 0.05  (No significativa)")

        self.results['correlaciones'] = {
            'pearson_corr': pearson_corr,
            'pearson_pval': pearson_pval,
            'spearman_corr': spearman_corr,
            'spearman_pval': spearman_pval
        }

        return self

    def ranking_drivers(self):
        """Ranking de drivers por correlación absoluta promedio"""
        print("\n" + "=" * 80)
        print("3. RANKING DE DRIVERS POR PODER PREDICTIVO")
        print("=" * 80)

        pearson_corr = self.results['correlaciones']['pearson_corr']
        spearman_corr = self.results['correlaciones']['spearman_corr']

        ranking_data = []

        for driver in self.driver_columns:
            pearson_abs_mean = pearson_corr.loc[driver].abs().mean()
            spearman_abs_mean = spearman_corr.loc[driver].abs().mean()
            avg_abs_corr = (pearson_abs_mean + spearman_abs_mean) / 2
            n_windows = len(self.pnl_columns)

            # Interpretar calidad
            calidad, emoji = self.interpretar_correlacion(avg_abs_corr)

            ranking_data.append({
                'Driver': driver,
                'Correlacion_Abs_Promedio': avg_abs_corr,
                'Pearson_Promedio': pearson_abs_mean,
                'Spearman_Promedio': spearman_abs_mean,
                'N_Ventanas': n_windows,
                'Calidad': calidad,
                'Emoji': emoji
            })

        ranking_df = pd.DataFrame(ranking_data)
        ranking_df = ranking_df.sort_values('Correlacion_Abs_Promedio', ascending=False)
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        ranking_df = ranking_df[['Rank', 'Driver', 'Correlacion_Abs_Promedio', 'Pearson_Promedio',
                                 'Spearman_Promedio', 'Calidad', 'Emoji', 'N_Ventanas']]

        print("\n>>> RANKING DE DRIVERS <<<\n")
        print(ranking_df.to_string(index=False))

        self.results['ranking'] = ranking_df
        self.mejor_driver = ranking_df.iloc[0]['Driver']

        print(f"\n🏆 MEJOR DRIVER: {self.mejor_driver}")
        print(f"   Correlación Absoluta Promedio: {ranking_df.iloc[0]['Correlacion_Abs_Promedio']:.4f}")
        print(f"   Calidad: {ranking_df.iloc[0]['Calidad']} {ranking_df.iloc[0]['Emoji']}")

        return self

    def analisis_rangos_mejor_driver(self):
        """Analizar PnL por rangos de percentiles para el mejor driver"""
        print("\n" + "=" * 80)
        print(f"4. ANÁLISIS POR RANGOS: {self.mejor_driver}")
        print("=" * 80)

        driver = self.mejor_driver
        percentiles = [25, 50, 75, 90]

        resultados_rangos = []

        for percentil in percentiles:
            umbral = self.df[driver].quantile(percentil / 100)

            print(f"\n>>> PERCENTIL {percentil} (Umbral: {umbral:.4f}) <<<\n")

            mascara_arriba = self.df[driver] >= umbral
            mascara_abajo = self.df[driver] < umbral

            n_arriba = mascara_arriba.sum()
            n_abajo = mascara_abajo.sum()

            print(f"  Por encima del umbral: {n_arriba:,} operaciones")
            print(f"  Por debajo del umbral: {n_abajo:,} operaciones")

            for pnl in self.pnl_columns:
                pnl_arriba = self.df.loc[mascara_arriba, pnl].mean()
                pnl_abajo = self.df.loc[mascara_abajo, pnl].mean()
                diferencial = pnl_arriba - pnl_abajo
                ganador = "ARRIBA" if diferencial > 0 else "ABAJO"

                resultados_rangos.append({
                    'Percentil': percentil,
                    'Variable_PnL': pnl,
                    'N_Arriba': n_arriba,
                    'N_Abajo': n_abajo,
                    'PnL_Medio_Arriba': pnl_arriba,
                    'PnL_Medio_Abajo': pnl_abajo,
                    'Diferencial': diferencial,
                    'Ganador': ganador
                })

                print(f"  {pnl}:")
                print(f"    Arriba: {pnl_arriba:>10.4f} | Abajo: {pnl_abajo:>10.4f} | Dif: {diferencial:>10.4f} | Ganador: {ganador}")

        rangos_df = pd.DataFrame(resultados_rangos)
        self.results['analisis_rangos'] = rangos_df

        return self

    def analisis_cuartiles(self):
        """Analizar PnL por cuartiles para todos los drivers"""
        print("\n" + "=" * 80)
        print("5. ANÁLISIS POR CUARTILES (TODOS LOS DRIVERS)")
        print("=" * 80)

        resultados_cuartiles = []

        for driver in self.driver_columns:
            print(f"\n>>> DRIVER: {driver} <<<\n")

            # Calcular cuartiles
            q1 = self.df[driver].quantile(0.25)
            q2 = self.df[driver].quantile(0.50)
            q3 = self.df[driver].quantile(0.75)

            # Asignar cuartiles
            etiquetas_cuartiles = pd.cut(self.df[driver],
                                    bins=[-np.inf, q1, q2, q3, np.inf],
                                    labels=['Q1 (25% inferior)', 'Q2', 'Q3', 'Q4 (25% superior)'])

            for pnl in self.pnl_columns:
                for etiqueta_q in ['Q1 (25% inferior)', 'Q2', 'Q3', 'Q4 (25% superior)']:
                    mascara = etiquetas_cuartiles == etiqueta_q
                    n_operaciones = mascara.sum()
                    pnl_medio = self.df.loc[mascara, pnl].mean()
                    std_pnl = self.df.loc[mascara, pnl].std()

                    resultados_cuartiles.append({
                        'Driver': driver,
                        'Variable_PnL': pnl,
                        'Cuartil': etiqueta_q,
                        'N_Operaciones': n_operaciones,
                        'PnL_Medio': pnl_medio,
                        'Desv_Std_PnL': std_pnl
                    })

            # Imprimir resumen para este driver
            resumen_driver = pd.DataFrame(resultados_cuartiles)
            resumen_driver = resumen_driver[resumen_driver['Driver'] == driver]
            pivot = resumen_driver.pivot_table(
                index='Cuartil',
                columns='Variable_PnL',
                values='PnL_Medio'
            )
            pivot = pivot.reindex(['Q1 (25% inferior)', 'Q2', 'Q3', 'Q4 (25% superior)'])
            print(pivot.to_string())

        cuartiles_df = pd.DataFrame(resultados_cuartiles)
        self.results['analisis_cuartiles'] = cuartiles_df

        return self

    def analisis_top_bottom(self):
        """Analizar Top 10% vs Bottom 10% para cada driver"""
        print("\n" + "=" * 80)
        print("6. ANÁLISIS TOP 10% vs BOTTOM 10%")
        print("=" * 80)

        resultados_top_bottom = []

        for driver in self.driver_columns:
            print(f"\n>>> DRIVER: {driver} <<<\n")

            p10 = self.df[driver].quantile(0.10)
            p90 = self.df[driver].quantile(0.90)

            mascara_bottom = self.df[driver] <= p10
            mascara_top = self.df[driver] >= p90

            n_bottom = mascara_bottom.sum()
            n_top = mascara_top.sum()

            print(f"  Bottom 10% (≤ {p10:.4f}): {n_bottom:,} operaciones")
            print(f"  Top 10% (≥ {p90:.4f}): {n_top:,} operaciones")
            print()

            for pnl in self.pnl_columns:
                media_bottom = self.df.loc[mascara_bottom, pnl].mean()
                media_top = self.df.loc[mascara_top, pnl].mean()
                spread = media_top - media_bottom
                direccion = "POSITIVO (Esperado)" if spread > 0 else "NEGATIVO (Inverso)"

                resultados_top_bottom.append({
                    'Driver': driver,
                    'Variable_PnL': pnl,
                    'N_Bottom': n_bottom,
                    'N_Top': n_top,
                    'PnL_Medio_Bottom': media_bottom,
                    'PnL_Medio_Top': media_top,
                    'Spread': spread,
                    'Direccion': direccion
                })

                print(f"  {pnl}:")
                print(f"    Bottom 10%: {media_bottom:>10.4f} | Top 10%: {media_top:>10.4f}")
                print(f"    SPREAD: {spread:>10.4f} ({direccion})")

        top_bottom_df = pd.DataFrame(resultados_top_bottom)
        self.results['analisis_top_bottom'] = top_bottom_df

        return self

    def escenarios_extremos(self):
        """Analizar escenarios extremos para el mejor driver"""
        print("\n" + "=" * 80)
        print(f"7. ESCENARIOS EXTREMOS: {self.mejor_driver}")
        print("=" * 80)

        driver = self.mejor_driver
        percentiles = [75, 85, 95]

        resultados_extremos = []

        for percentil in percentiles:
            umbral = self.df[driver].quantile(percentil / 100)
            mascara = self.df[driver] >= umbral
            n_operaciones = mascara.sum()
            pct_retencion = (n_operaciones / len(self.df)) * 100

            print(f"\n>>> PERCENTIL {percentil} (Umbral: {umbral:.4f}) <<<")
            print(f"  Operaciones retenidas: {n_operaciones:,} ({pct_retencion:.2f}%)")
            print()

            for pnl in self.pnl_columns:
                pnl_medio = self.df.loc[mascara, pnl].mean()
                std_pnl = self.df.loc[mascara, pnl].std()

                resultados_extremos.append({
                    'Percentil': percentil,
                    'Umbral': umbral,
                    'N_Operaciones': n_operaciones,
                    'Retencion_%': pct_retencion,
                    'Variable_PnL': pnl,
                    'PnL_Medio': pnl_medio,
                    'Desv_Std_PnL': std_pnl,
                    'Limite_Inferior': pnl_medio - std_pnl,
                    'Limite_Superior': pnl_medio + std_pnl
                })

                print(f"  {pnl}:")
                print(f"    Media: {pnl_medio:>10.4f} ± {std_pnl:.4f}")
                print(f"    Rango: [{pnl_medio - std_pnl:>10.4f}, {pnl_medio + std_pnl:>10.4f}]")

        extremos_df = pd.DataFrame(resultados_extremos)
        self.results['escenarios_extremos'] = extremos_df

        return self

    def recomendaciones_filtros(self):
        """Generar recomendaciones de filtros"""
        print("\n" + "=" * 80)
        print(f"8. RECOMENDACIONES DE FILTROS: {self.mejor_driver}")
        print("=" * 80)

        driver = self.mejor_driver
        configs_filtro = [
            ('Conservador', 75),
            ('Equilibrado', 90),
            ('Agresivo', 95)
        ]

        resultados_filtros = []

        for nombre_filtro, percentil in configs_filtro:
            umbral = self.df[driver].quantile(percentil / 100)
            mascara = self.df[driver] >= umbral
            n_operaciones = mascara.sum()
            pct_retencion = (n_operaciones / len(self.df)) * 100

            print(f"\n>>> FILTRO {nombre_filtro.upper()} (P{percentil}) <<<")
            print(f"  Umbral: {driver} ≥ {umbral:.4f}")
            print(f"  Retención: {n_operaciones:,} operaciones ({pct_retencion:.2f}%)")
            print(f"\n  PnL Esperado:")

            for pnl in self.pnl_columns:
                pnl_esperado = self.df.loc[mascara, pnl].mean()

                resultados_filtros.append({
                    'Tipo_Filtro': nombre_filtro,
                    'Percentil': percentil,
                    'Umbral': umbral,
                    'N_Operaciones': n_operaciones,
                    'Retencion_%': pct_retencion,
                    'Variable_PnL': pnl,
                    'PnL_Esperado': pnl_esperado
                })

                print(f"    {pnl}: {pnl_esperado:>10.4f}")

        # Anti-filtros (zonas de bajo rendimiento)
        print("\n>>> ANTI-FILTROS (ZONAS A EVITAR) <<<")

        p25 = self.df[driver].quantile(0.25)
        mascara_bajo = self.df[driver] <= p25
        n_bajo = mascara_bajo.sum()

        print(f"\n  🚫 ZONA BAJA: {driver} ≤ {p25:.4f}")
        print(f"     Afecta: {n_bajo:,} operaciones ({(n_bajo/len(self.df)*100):.2f}%)")
        print(f"     PnL Esperado (típicamente más bajo):")

        for pnl in self.pnl_columns:
            pnl_medio_bajo = self.df.loc[mascara_bajo, pnl].mean()
            print(f"       {pnl}: {pnl_medio_bajo:>10.4f}")

        filtros_df = pd.DataFrame(resultados_filtros)
        self.results['recomendaciones_filtros'] = filtros_df

        return self

    def generar_visualizaciones(self):
        """Generar todas las visualizaciones"""
        print("\n" + "=" * 80)
        print("9. GENERANDO VISUALIZACIONES")
        print("=" * 80)

        directorio_salida = Path('analisis_resultados')
        directorio_salida.mkdir(exist_ok=True)

        # 1. Mapa de calor de correlación (Pearson)
        print("\n  [1/6] Mapa de Calor de Correlaciones (Pearson)...")
        fig, ax = plt.subplots(figsize=(14, 8))
        pearson_corr = self.results['correlaciones']['pearson_corr']

        # Simplificar nombres de columnas para visualización
        cols_simplificados = [col.replace('PnL_fwd_pts_', '').replace('_mediana', '') for col in self.pnl_columns]
        pearson_display = pearson_corr.copy()
        pearson_display.columns = cols_simplificados

        sns.heatmap(pearson_display.astype(float), annot=True, fmt='.3f',
                   cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlación de Pearson: Drivers vs PnL Forward Points',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Ventanas PnL', fontsize=12, fontweight='bold')
        plt.ylabel('Variables Driver', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(directorio_salida / '01_mapa_calor_pearson.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Mapa de calor de correlación (Spearman)
        print("  [2/6] Mapa de Calor de Correlaciones (Spearman)...")
        fig, ax = plt.subplots(figsize=(14, 8))
        spearman_corr = self.results['correlaciones']['spearman_corr']
        spearman_display = spearman_corr.copy()
        spearman_display.columns = cols_simplificados

        sns.heatmap(spearman_display.astype(float), annot=True, fmt='.3f',
                   cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlación de Spearman: Drivers vs PnL Forward Points',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Ventanas PnL', fontsize=12, fontweight='bold')
        plt.ylabel('Variables Driver', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(directorio_salida / '02_mapa_calor_spearman.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Gráfico de barras de ranking de drivers
        print("  [3/6] Ranking de Drivers...")
        fig, ax = plt.subplots(figsize=(12, 8))
        ranking = self.results['ranking'].copy()

        barras = ax.barh(ranking['Driver'], ranking['Correlacion_Abs_Promedio'],
                      color=sns.color_palette("husl", len(ranking)))
        ax.set_xlabel('Correlación Absoluta Promedio', fontsize=12, fontweight='bold')
        ax.set_ylabel('Variable Driver', fontsize=12, fontweight='bold')
        ax.set_title('Ranking de Drivers por Poder Predictivo', fontsize=16, fontweight='bold', pad=20)
        ax.invert_yaxis()

        # Añadir etiquetas de valores y calidad
        for i, (barra, val, calidad) in enumerate(zip(barras, ranking['Correlacion_Abs_Promedio'], ranking['Calidad'])):
            ax.text(val + 0.001, barra.get_y() + barra.get_height()/2,
                   f'{val:.4f} ({calidad})', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(directorio_salida / '03_ranking_drivers.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Análisis de cuartiles para mejor driver
        print("  [4/6] Análisis de Cuartiles (Mejor Driver)...")
        fig, ax = plt.subplots(figsize=(14, 8))

        datos_cuartiles = self.results['analisis_cuartiles']
        datos_mejor_driver = datos_cuartiles[datos_cuartiles['Driver'] == self.mejor_driver]

        pivot = datos_mejor_driver.pivot(index='Cuartil', columns='Variable_PnL', values='PnL_Medio')
        pivot = pivot.reindex(['Q1 (25% inferior)', 'Q2', 'Q3', 'Q4 (25% superior)'])
        pivot.columns = cols_simplificados

        x = np.arange(len(pivot.index))
        ancho = 0.15
        multiplicador = 0

        for i, pnl in enumerate(pivot.columns):
            offset = ancho * multiplicador
            barras = ax.bar(x + offset, pivot[pnl], ancho, label=pnl)
            multiplicador += 1

        ax.set_xlabel('Cuartil', fontsize=12, fontweight='bold')
        ax.set_ylabel('PnL Medio', fontsize=12, fontweight='bold')
        ax.set_title(f'Análisis por Cuartiles: {self.mejor_driver}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + ancho * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(pivot.index, rotation=0)
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(directorio_salida / '04_analisis_cuartiles_mejor_driver.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Spread Top 10% vs Bottom 10%
        print("  [5/6] Análisis Top vs Bottom 10%...")
        fig, ax = plt.subplots(figsize=(14, 10))

        top_bottom = self.results['analisis_top_bottom']

        n_drivers = len(self.driver_columns)
        n_pnl = len(self.pnl_columns)

        x = np.arange(n_pnl)
        ancho = 0.12

        for i, driver in enumerate(self.driver_columns):
            datos_driver = top_bottom[top_bottom['Driver'] == driver]
            spreads = datos_driver['Spread'].values
            offset = ancho * (i - n_drivers/2 + 0.5)
            ax.bar(x + offset, spreads, ancho, label=driver)

        ax.set_xlabel('Ventana PnL', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spread (Top 10% - Bottom 10%)', fontsize=12, fontweight='bold')
        ax.set_title('Spread Top 10% vs Bottom 10% por Driver', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(cols_simplificados, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=9)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(directorio_salida / '05_spread_top_bottom.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Rendimiento de filtros recomendados
        print("  [6/6] Recomendaciones de Filtros...")
        fig, ax = plt.subplots(figsize=(14, 8))

        datos_filtros = self.results['recomendaciones_filtros']

        tipos_filtro = datos_filtros['Tipo_Filtro'].unique()
        x = np.arange(len(cols_simplificados))
        ancho = 0.25

        for i, tipo_filtro in enumerate(tipos_filtro):
            subconjunto_filtro = datos_filtros[datos_filtros['Tipo_Filtro'] == tipo_filtro]
            pnls_esperados = subconjunto_filtro['PnL_Esperado'].values
            offset = ancho * (i - 1)
            barras = ax.bar(x + offset, pnls_esperados, ancho, label=tipo_filtro)

        ax.set_xlabel('Ventana PnL', fontsize=12, fontweight='bold')
        ax.set_ylabel('PnL Esperado', fontsize=12, fontweight='bold')
        ax.set_title(f'Recomendaciones de Filtros: {self.mejor_driver}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(cols_simplificados, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(directorio_salida / '06_recomendaciones_filtros.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Todas las visualizaciones guardadas en: {directorio_salida}/")

        return self

    def generar_informe_ejecutivo(self):
        """Generar resumen ejecutivo y conclusiones en Markdown"""
        print("\n" + "=" * 80)
        print("10. RESUMEN EJECUTIVO Y CONCLUSIONES")
        print("=" * 80)

        directorio_salida = Path('analisis_resultados')
        archivo_informe = directorio_salida / 'INFORME_EJECUTIVO.md'

        with open(archivo_informe, 'w', encoding='utf-8') as f:
            # Encabezado
            f.write("# 📊 ANÁLISIS ESTADÍSTICO COMPLETO\n")
            f.write("## Correlaciones entre Etiquetas de Ventas y PnL Forward Points\n\n")
            f.write("---\n\n")

            # Resumen del dataset
            f.write("## 1. 📋 RESUMEN DEL DATASET\n\n")
            f.write(f"- **Total de Observaciones:** {len(self.df):,}\n")
            f.write(f"- **Variables PnL Analizadas:** {len(self.pnl_columns)}\n")
            f.write(f"- **Variables Driver Analizadas:** {len(self.driver_columns)}\n\n")

            f.write("### Variables PnL:\n")
            for pnl in self.pnl_columns:
                f.write(f"- `{pnl}`\n")

            f.write("\n### Variables Driver:\n")
            for driver in self.driver_columns:
                f.write(f"- `{driver}`\n")

            f.write("\n---\n\n")

            # Escala de interpretación
            f.write("## 2. 📐 ESCALA DE INTERPRETACIÓN DE CORRELACIONES\n\n")
            f.write("### Fuerza de la Correlación:\n\n")
            f.write("| Rango | Interpretación | Emoji |\n")
            f.write("|-------|----------------|-------|\n")
            f.write("| \\|r\\| < 0.20 | Muy Débil | 🔵 |\n")
            f.write("| 0.20 ≤ \\|r\\| < 0.40 | Débil | 🟢 |\n")
            f.write("| 0.40 ≤ \\|r\\| < 0.60 | Moderada | 🟡 |\n")
            f.write("| 0.60 ≤ \\|r\\| < 0.80 | Fuerte | 🟠 |\n")
            f.write("| \\|r\\| ≥ 0.80 | Muy Fuerte | 🔴 |\n\n")

            f.write("### Significancia Estadística:\n\n")
            f.write("| Símbolo | P-valor | Interpretación |\n")
            f.write("|---------|---------|----------------|\n")
            f.write("| *** | p < 0.001 | Altamente significativa |\n")
            f.write("| ** | p < 0.01 | Muy significativa |\n")
            f.write("| * | p < 0.05 | Significativa |\n")
            f.write("| ns | p ≥ 0.05 | No significativa |\n\n")

            f.write("---\n\n")

            # Top 3 drivers
            f.write("## 3. 🏆 TOP 3 DRIVERS POR PODER PREDICTIVO\n\n")
            ranking = self.results['ranking'].head(3)

            for idx, row in ranking.iterrows():
                f.write(f"### #{int(row['Rank'])}. **{row['Driver']}** {row['Emoji']}\n\n")
                f.write(f"- **Correlación Absoluta Promedio:** {row['Correlacion_Abs_Promedio']:.4f}\n")
                f.write(f"- **Calidad de Correlación:** {row['Calidad']}\n")
                f.write(f"- **Pearson Promedio:** {row['Pearson_Promedio']:.4f}\n")
                f.write(f"- **Spearman Promedio:** {row['Spearman_Promedio']:.4f}\n\n")

            f.write("---\n\n")

            # Hallazgos clave
            f.write("## 4. 🔍 HALLAZGOS CLAVE\n\n")

            mejor_driver = self.results['ranking'].iloc[0]['Driver']
            mejor_corr = self.results['ranking'].iloc[0]['Correlacion_Abs_Promedio']
            mejor_calidad = self.results['ranking'].iloc[0]['Calidad']

            f.write(f"### ✅ MEJOR DRIVER: **{mejor_driver}**\n\n")
            f.write(f"- Muestra la correlación más fuerte con PnL (promedio: **{mejor_corr:.4f}**)\n")
            f.write(f"- Calidad de correlación: **{mejor_calidad}**\n\n")

            # Análisis detallado de calidad de correlaciones
            f.write("### 📊 CALIDAD DE LAS CORRELACIONES (Análisis Detallado)\n\n")

            ranking_completo = self.results['ranking']

            f.write("**Resumen por Calidad:**\n\n")

            # Contar drivers por calidad
            conteo_calidad = ranking_completo['Calidad'].value_counts()

            for calidad in ['Muy Débil', 'Débil', 'Moderada', 'Fuerte', 'Muy Fuerte']:
                if calidad in conteo_calidad:
                    drivers_en_categoria = ranking_completo[ranking_completo['Calidad'] == calidad]
                    f.write(f"- **{calidad}:** {conteo_calidad[calidad]} driver(s)\n")
                    for _, driver_row in drivers_en_categoria.iterrows():
                        f.write(f"  - `{driver_row['Driver']}` (r = {driver_row['Correlacion_Abs_Promedio']:.4f})\n")
                    f.write("\n")

            f.write("**Interpretación General:**\n\n")

            if mejor_corr < 0.20:
                f.write("⚠️ **ADVERTENCIA:** Todas las correlaciones son **MUY DÉBILES**. Esto indica que:\n")
                f.write("- Los drivers analizados tienen un poder predictivo muy limitado sobre el PnL\n")
                f.write("- Pueden existir otros factores no capturados que influyen más en el rendimiento\n")
                f.write("- Se recomienda precaución al aplicar filtros basados en estos drivers\n\n")
            elif mejor_corr < 0.40:
                f.write("⚠️ Las correlaciones son **DÉBILES**. Consideraciones:\n")
                f.write("- Los drivers muestran alguna relación con el PnL, pero es limitada\n")
                f.write("- Los filtros pueden proporcionar mejoras modestas en el rendimiento\n")
                f.write("- Se recomienda combinar múltiples drivers para mejorar la predicción\n\n")
            elif mejor_corr < 0.60:
                f.write("✅ Las correlaciones son **MODERADAS**. Esto significa:\n")
                f.write("- Los drivers tienen capacidad predictiva razonable\n")
                f.write("- Los filtros basados en estos drivers pueden ser efectivos\n")
                f.write("- Se recomienda validación con datos fuera de muestra\n\n")
            elif mejor_corr < 0.80:
                f.write("🎯 Excelente! Las correlaciones son **FUERTES**:\n")
                f.write("- Los drivers tienen alto poder predictivo\n")
                f.write("- Los filtros serán altamente efectivos\n")
                f.write("- Alta confianza en las recomendaciones\n\n")
            else:
                f.write("🌟 Excepcional! Las correlaciones son **MUY FUERTES**:\n")
                f.write("- Los drivers son excelentes predictores del PnL\n")
                f.write("- Máxima efectividad esperada de los filtros\n")
                f.write("- Considerar la posibilidad de sobreajuste y validar\n\n")

            # Análisis top vs bottom
            top_bottom = self.results['analisis_top_bottom']
            mejor_driver_tb = top_bottom[top_bottom['Driver'] == mejor_driver]
            spread_promedio = mejor_driver_tb['Spread'].mean()

            f.write(f"### 📈 RENDIMIENTO TOP 10% vs BOTTOM 10%\n\n")
            f.write(f"- **Spread Promedio:** {spread_promedio:.4f}\n")
            if spread_promedio > 0:
                f.write(f"- **Dirección:** POSITIVA ✅ (Mayor {mejor_driver} → Mayor PnL)\n\n")
            else:
                f.write(f"- **Dirección:** NEGATIVA ⚠️ (Relación inversa)\n\n")

            # Paradojas y anomalías
            f.write("### ⚠️ PARADOJAS Y ANOMALÍAS DETECTADAS\n\n")

            pearson_corr = self.results['correlaciones']['pearson_corr']
            anomalias_encontradas = False

            for driver in self.driver_columns[:3]:  # Top 3
                avg_corr = pearson_corr.loc[driver].mean()
                if avg_corr < 0:
                    f.write(f"- ⚠️ `{driver}` muestra correlación promedio NEGATIVA ({avg_corr:.4f})\n")
                    anomalias_encontradas = True

            # Correlaciones mixtas
            for driver in self.driver_columns:
                driver_corrs = pearson_corr.loc[driver].values
                if (driver_corrs > 0).any() and (driver_corrs < 0).any():
                    f.write(f"- ⚠️ `{driver}` muestra correlaciones MIXTAS entre ventanas (inconsistente)\n")
                    anomalias_encontradas = True

            if not anomalias_encontradas:
                f.write("✅ No se detectaron anomalías significativas. Las correlaciones son consistentes.\n")

            f.write("\n---\n\n")

            # Recomendaciones de filtros
            f.write("## 5. 🎯 RECOMENDACIONES DE FILTROS\n\n")

            recomendaciones_filtros = self.results['recomendaciones_filtros']

            for tipo_filtro in ['Conservador', 'Equilibrado', 'Agresivo']:
                subconjunto_filtro = recomendaciones_filtros[recomendaciones_filtros['Tipo_Filtro'] == tipo_filtro]
                if len(subconjunto_filtro) > 0:
                    fila = subconjunto_filtro.iloc[0]
                    pnl_promedio = subconjunto_filtro['PnL_Esperado'].mean()

                    if tipo_filtro == 'Conservador':
                        emoji = "🛡️"
                    elif tipo_filtro == 'Equilibrado':
                        emoji = "⚖️"
                    else:
                        emoji = "🚀"

                    f.write(f"### {emoji} FILTRO {tipo_filtro.upper()} (P{int(fila['Percentil'])})\n\n")
                    f.write(f"- **Umbral:** `{mejor_driver}` ≥ {fila['Umbral']:.4f}\n")
                    f.write(f"- **Retención:** {fila['N_Operaciones']:,.0f} operaciones ({fila['Retencion_%']:.2f}%)\n")
                    f.write(f"- **PnL Esperado Promedio:** {pnl_promedio:.4f} puntos\n\n")

                    f.write("**PnL Esperado por Ventana:**\n\n")
                    for _, row in subconjunto_filtro.iterrows():
                        ventana = row['Variable_PnL'].replace('PnL_fwd_pts_', '').replace('_mediana', '')
                        f.write(f"- Ventana {ventana}: {row['PnL_Esperado']:.2f} pts\n")
                    f.write("\n")

            # Anti-filtros
            f.write("### 🚫 ANTI-FILTROS (ZONAS A EVITAR)\n\n")
            p25 = self.df[mejor_driver].quantile(0.25)
            mascara_bajo = self.df[mejor_driver] <= p25
            n_bajo = mascara_bajo.sum()

            f.write(f"**ZONA BAJA:** `{mejor_driver}` ≤ {p25:.4f}\n\n")
            f.write(f"- **Operaciones Afectadas:** {n_bajo:,} ({(n_bajo/len(self.df)*100):.2f}%)\n")
            f.write(f"- **Motivo:** Rendimiento significativamente inferior\n\n")

            f.write("**PnL Esperado (Zona Baja):**\n\n")
            for pnl in self.pnl_columns:
                pnl_medio_bajo = self.df.loc[mascara_bajo, pnl].mean()
                ventana = pnl.replace('PnL_fwd_pts_', '').replace('_mediana', '')
                f.write(f"- Ventana {ventana}: {pnl_medio_bajo:.2f} pts\n")

            f.write("\n---\n\n")

            # Recomendaciones finales
            f.write("## 6. 💡 RECOMENDACIONES FINALES\n\n")

            f.write(f"### 1. 🎯 FILTRO PRINCIPAL\n\n")
            f.write(f"**Usar `{mejor_driver}` como criterio de selección principal**\n\n")
            f.write(f"- Estrategia recomendada: **Filtro Equilibrado (P90)**\n")
            f.write(f"- Ofrece el mejor balance entre selectividad y retención\n")
            f.write(f"- Mejora sustancial del PnL esperado con riesgo controlado\n\n")

            f.write(f"### 2. 🔗 FILTROS SECUNDARIOS\n\n")
            f.write(f"Considerar combinar con:\n\n")
            if len(ranking) > 1:
                segundo_mejor = ranking.iloc[1]['Driver']
                f.write(f"- **`{segundo_mejor}`** (Rank #2)\n")
            if len(ranking) > 2:
                tercer_mejor = ranking.iloc[2]['Driver']
                f.write(f"- **`{tercer_mejor}`** (Rank #3)\n")
            f.write("\nLa combinación de múltiples drivers puede mejorar la robustez del sistema de filtrado.\n\n")

            f.write(f"### 3. 🚫 EXCLUSIONES\n\n")
            f.write(f"Evitar operaciones donde:\n\n")
            f.write(f"- `{mejor_driver}` < {p25:.4f} (25% inferior)\n")
            f.write(f"- Estas operaciones muestran rendimiento consistentemente bajo\n\n")

            f.write(f"### 4. 📊 MONITOREO Y VALIDACIÓN\n\n")
            f.write(f"- **Seguimiento continuo:** Rastrear estabilidad de correlaciones en el tiempo\n")
            f.write(f"- **Validación out-of-sample:** Testear filtros con datos no utilizados en este análisis\n")
            f.write(f"- **Adaptación:** Las relaciones pueden evolucionar con cambios en condiciones de mercado\n")
            f.write(f"- **Revisión periódica:** Re-ejecutar este análisis trimestral o semestralmente\n\n")

            f.write(f"### 5. ⚠️ ADVERTENCIAS IMPORTANTES\n\n")

            if mejor_corr < 0.40:
                f.write(f"- ⚠️ **Correlaciones débiles:** El poder predictivo es limitado\n")
                f.write(f"- Los filtros pueden ofrecer mejoras modestas pero no garantizadas\n")
                f.write(f"- Considerar otros factores no capturados en este análisis\n")
                f.write(f"- Validación rigurosa es crítica antes de implementación en producción\n\n")
            else:
                f.write(f"- ⚠️ Rendimientos pasados no garantizan resultados futuros\n")
                f.write(f"- Riesgo de sobreajuste: validar con datos independientes\n")
                f.write(f"- Considerar costos de transacción y slippage en implementación real\n\n")

            f.write("---\n\n")

            # Tabla resumen de rendimiento
            f.write("## 7. 📈 TABLA RESUMEN DE RENDIMIENTO\n\n")
            f.write("### Comparación: Filtro Equilibrado (P90) vs Sin Filtro\n\n")

            filtro_equilibrado = recomendaciones_filtros[recomendaciones_filtros['Tipo_Filtro'] == 'Equilibrado']

            f.write("| Ventana | PnL Sin Filtro | PnL Con Filtro | Mejora | Mejora % |\n")
            f.write("|---------|----------------|----------------|--------|----------|\n")

            pnl_stats = self.results['estadisticas_descriptivas']['pnl']

            for _, row in filtro_equilibrado.iterrows():
                pnl_col = row['Variable_PnL']
                ventana = pnl_col.replace('PnL_fwd_pts_', '').replace('_mediana', '')
                pnl_sin_filtro = pnl_stats.loc[pnl_col, 'mean']
                pnl_con_filtro = row['PnL_Esperado']
                mejora = pnl_con_filtro - pnl_sin_filtro
                mejora_pct = (mejora / abs(pnl_sin_filtro) * 100) if pnl_sin_filtro != 0 else 0

                f.write(f"| {ventana} días | {pnl_sin_filtro:.2f} | {pnl_con_filtro:.2f} | {mejora:+.2f} | {mejora_pct:+.1f}% |\n")

            f.write("\n---\n\n")

            # Metodología
            f.write("## 8. 📚 METODOLOGÍA\n\n")
            f.write("### Técnicas Estadísticas Aplicadas:\n\n")
            f.write("1. **Correlación de Pearson:** Mide relación lineal entre variables\n")
            f.write("2. **Correlación de Spearman:** Mide relación monotónica (robusta a outliers)\n")
            f.write("3. **Análisis por Percentiles:** Identifica umbrales óptimos de filtrado\n")
            f.write("4. **Análisis por Cuartiles:** Evalúa distribución de rendimiento\n")
            f.write("5. **Top/Bottom Analysis:** Compara extremos de distribución\n\n")

            f.write("### Datos Analizados:\n\n")
            f.write(f"- **Periodo:** Dataset completo disponible\n")
            f.write(f"- **N Observaciones:** {len(self.df):,}\n")
            f.write(f"- **Variables:** {len(self.driver_columns)} drivers × {len(self.pnl_columns)} ventanas PnL\n\n")

            f.write("---\n\n")

            # Footer
            f.write("## 📌 CONCLUSIÓN\n\n")

            if mejor_corr >= 0.40:
                f.write(f"Este análisis identifica **{mejor_driver}** como el driver con mayor poder predictivo ")
                f.write(f"({mejor_calidad} correlación: {mejor_corr:.4f}). ")
                f.write(f"La implementación del filtro equilibrado (P90) puede mejorar significativamente ")
                f.write(f"el PnL esperado, reteniendo el 10% de operaciones con mejor score en {mejor_driver}.\n\n")
            else:
                f.write(f"Este análisis identifica **{mejor_driver}** como el driver con mayor poder predictivo, ")
                f.write(f"aunque las correlaciones son {mejor_calidad.lower()} ({mejor_corr:.4f}). ")
                f.write(f"Se recomienda precaución al implementar filtros y considerar validación exhaustiva ")
                f.write(f"con datos out-of-sample antes de uso en producción.\n\n")

            f.write("**Próximos pasos recomendados:**\n\n")
            f.write("1. Validar resultados con datos históricos no incluidos en este análisis\n")
            f.write("2. Realizar backtesting de la estrategia de filtrado propuesta\n")
            f.write("3. Implementar monitoreo en tiempo real de las correlaciones\n")
            f.write("4. Considerar análisis de regresión multivariante combinando drivers\n\n")

            f.write("---\n\n")
            f.write(f"*Informe generado automáticamente el {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        print(f"\n✓ Informe ejecutivo guardado en: {archivo_informe}")

        # Imprimir informe en consola
        with open(archivo_informe, 'r', encoding='utf-8') as f:
            print("\n" + f.read())

        return self

    def exportar_resultados(self):
        """Exportar todos los resultados a archivos CSV"""
        print("\n" + "=" * 80)
        print("11. EXPORTANDO RESULTADOS A CSV")
        print("=" * 80)

        directorio_salida = Path('analisis_resultados')

        # Exportar cada dataframe de resultados
        exportaciones = [
            ('estadisticas_descriptivas_drivers.csv', self.results['estadisticas_descriptivas']['drivers']),
            ('estadisticas_descriptivas_pnl.csv', self.results['estadisticas_descriptivas']['pnl']),
            ('correlaciones_pearson.csv', self.results['correlaciones']['pearson_corr']),
            ('correlaciones_pearson_pvalores.csv', self.results['correlaciones']['pearson_pval']),
            ('correlaciones_spearman.csv', self.results['correlaciones']['spearman_corr']),
            ('correlaciones_spearman_pvalores.csv', self.results['correlaciones']['spearman_pval']),
            ('ranking_drivers.csv', self.results['ranking']),
            ('analisis_rangos.csv', self.results['analisis_rangos']),
            ('analisis_cuartiles.csv', self.results['analisis_cuartiles']),
            ('analisis_top_bottom.csv', self.results['analisis_top_bottom']),
            ('escenarios_extremos.csv', self.results['escenarios_extremos']),
            ('recomendaciones_filtros.csv', self.results['recomendaciones_filtros'])
        ]

        for nombre_archivo, df in exportaciones:
            ruta_archivo = directorio_salida / nombre_archivo
            df.to_csv(ruta_archivo, index=True)
            print(f"  ✓ {nombre_archivo}")

        print(f"\n✓ Todos los resultados exportados a: {directorio_salida}/")

        return self

    def ejecutar_analisis_completo(self):
        """Ejecutar el pipeline completo de análisis"""
        print("\n" + "🔬" * 40)
        print("ANÁLISIS ESTADÍSTICO COMPLETO")
        print("Correlaciones de Etiquetas de Ventas con PnL Forward Points")
        print("🔬" * 40 + "\n")

        try:
            (self.cargar_datos()
             .estadisticas_descriptivas()
             .analisis_correlaciones()
             .ranking_drivers()
             .analisis_rangos_mejor_driver()
             .analisis_cuartiles()
             .analisis_top_bottom()
             .escenarios_extremos()
             .recomendaciones_filtros()
             .generar_visualizaciones()
             .generar_informe_ejecutivo()
             .exportar_resultados())

            print("\n" + "=" * 80)
            print("✓ ¡ANÁLISIS COMPLETADO!")
            print("=" * 80)
            print(f"\nTodos los resultados guardados en: analisis_resultados/")
            print("  - 6 visualizaciones (PNG, 300 DPI)")
            print("  - 12 archivos CSV con datos")
            print("  - 1 informe ejecutivo en Markdown")

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Ejecutar análisis
    analisis = AnalisisCorrelacionEtiquetasVentas('combined_mediana_labeled.csv')
    analisis.ejecutar_analisis_completo()
