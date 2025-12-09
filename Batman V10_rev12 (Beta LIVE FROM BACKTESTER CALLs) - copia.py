# -*- coding: utf-8 -*-
"""
BATMAN SPX V8 — SISTEMA FWD (Forward Testing) con Mediana Intraday 14×30min — LOCAL DATA
===========================================================================
VERSIÓN: Batman V8 — Mediana Intraday sobre 14 timestamps fijos de mercado US

NOVEDADES V8:
1) **FWD intradía por mediana (14×30min)**:
   - Para cada ventana FWD (01,05,15,25,50), evalúa estructura en 14 timestamps fijos US:
     ["09:30","10:00","10:30","11:00","11:30","12:00","12:30","13:00","13:30","14:00","14:30","15:00","15:30","16:00"]
   - Calcula PnL_fwd_pts y PnL_fwd_pct para cada timestamp
   - Agrega por MEDIANA (no media) → columnas: PnL_fwd_pts_*_mediana y PnL_fwd_pct_*_mediana
   - MANTIENE columnas originales sin _mediana (primer timestamp válido)
   - Optimización: lee archivo FWD UNA SOLA VEZ por ventana (no 14 veces)

NOVEDADES V5 (base):
1) **Uso de TODAS las expiraciones en RANGE_A/RANGE_B**:
   - Eliminados TARGET_E1/E2 y MAX_E1/E2
   - Se procesan todas las expiraciones dentro de cada rango, ordenadas por DTE ascendente
   - Sin priorización ni límites de expiraciones

2) **Sistema FWD simplificado**:
   - Selección exclusivamente por porcentaje (FWD_TOP_PCT)
   - Eliminados FWD_TOP_COUNT y FWD_UNIQUE_TOP_N
   - Filtro WINNERS/LOSERS basado en RANKING_MODE (BQI_ABS, PnLDV, EarScore)
   - Aplicación de FWD_TOP_PCT sobre estructuras ya filtradas
   - **DEDUPLICACIÓN por DTE1/DTE2**: elimina batmans con mismos DTEs antes de FWD
     (igual que Calendar V8_rev10, evita procesar múltiples estructuras con mismos vencimientos)

3) **Sistema FWD completo para backtest**:
   - Ventanas forward en % del DTE1: [1%, 5%, 15%, 25%, 50%]
   - Revalorización de estructura Batman en momentos futuros
   - Cálculo de P&L en puntos SPX y porcentaje
   - Tracking de SPX, bid/ask/mid, root checks por pata

4) **Paralelización con ProcessPoolExecutor**:
   - Worker function `_process_one_fwd_batman` para cálculo paralelo
   - Múltiples workers para maximizar performance
   - Progress tracking en tiempo real

5) **Análisis gráfico y correlaciones**:
   - Gráficos de promedios P&L% por ventana FWD
   - Filtrado de outliers (>±50%)
   - Correlaciones FWD vs métricas (BQI_ABS, PnLDV, EarScore, etc.)
   - Comparativa winners vs losers

6) **Adaptación a estructura Batman**:
   - Función `batman_value_from_df`: revaloriza -1C@k1 +2C@k2 -1C@k3
   - Cálculo de net_credit forward (3 patas, 2 expiraciones)
   - Metadata completa por pata (bid/ask/mid/root)

7) **Simplificaciones**:
   - N_CONTRACTS fijo en 1 (theta calculado por contrato)
   - Sistema de auditoría deprecado (funciones ahora son no-ops)

MANTIENE:
- Death Valley + PnLDV + BQI_ABS normalizado
- Canonical chain robusto
- Caché de greeks
- Filtros de liquidez y spread
"""

import os, math, calendar, re, random, urllib.request
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para guardar gráficos

from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor

BASE_URL = "https://optionstrat.com/build/custom/SPX/"
TZ_US = ZoneInfo("America/New_York")
TZ_ES = ZoneInfo("Europe/Madrid")

# ================== AUDIT LOG (DEPRECATED) ==================
# Sistema de auditoría eliminado para simplificar el código

def log_inc(tag: str, n: int = 1):
    pass  # No-op

def log_add(tag: str, n: int):
    pass  # No-op

def audit_dump(save_dir: Path | None = None, prefix: str = "AUDIT"):
    pass  # No-op

# ================== RUTAS Y CONFIGURACIÓN ==================
DATA_DIR = r"C:\Users\Administrator\Desktop\FINAL DATA\HIST AND STREAMING DATA\STREAMING"
DESKTOP  = Path.home() / "Desktop"

# ================== CONFIG SNAPSHOT ==================
# Configuración de timestamps para el snapshot inicial de estructuras Batman
TARGET_HHMMS_US = ["10:30"]               # Lista de horas US objetivo para snapshot (formato "HH:MM")
                                           # Ejemplo: ["10:30", "14:00"] busca estructuras a las 10:30 AM y 2:00 PM
NEAREST_MINUTE_TOLERANCE = 5              # Tolerancia en minutos para encontrar timestamp más cercano
                                           # Si el timestamp exacto no existe, busca dentro de ±5 minutos
IGNORE_TARGET_MINUTE = True              # Control de procesamiento completo del día
                                           # False: solo procesa los timestamps en TARGET_HHMMS_US
                                           # True: ignora TARGET_HHMMS_US y procesa TODOS los timestamps del CSV
                                           # ATENCIÓN: True puede generar miles de estructuras por día

# ================== CONFIG FWD (Forward Testing) ==================
# Ventanas de tiempo forward para evaluar evolución de las estructuras Batman
FORWARD_FRACS = [0.01, 0.05, 0.15, 0.25, 0.50]  # Fracciones del DTE1 para ventanas forward
                                                  # 0.01 = 1% del DTE1 (ej: si DTE1=50, evalúa en 0.5 días ≈ 12 horas)
                                                  # 0.05 = 5% del DTE1 (ej: si DTE1=50, evalúa en 2.5 días)
                                                  # 0.15 = 15% del DTE1 (ej: si DTE1=50, evalúa en 7.5 días)
                                                  # 0.25 = 25% del DTE1 (ej: si DTE1=50, evalúa en 12.5 días)
                                                  # 0.50 = 50% del DTE1 (ej: si DTE1=50, evalúa en 25 días)
FRAC_SUFFIXES = [f"{int(round(fr*100)):02d}" for fr in FORWARD_FRACS]  # Sufijos para nombrar columnas (01, 05, 15, 25, 50)

# Timestamps fijos de mercado US para evaluación intraday (mediana sobre múltiples snapshots)
FWD_INTRADAY_TIMESTAMPS = [
     "10:30", "11:00", "11:30", "12:00", "12:30",    # Mañana: 10:30 AM - 12:30 PM
    "13:00", "13:30", "14:00", "14:30", "15:00"      # Tarde: 1:00 PM - 3:00 PM
]
# PROPÓSITO: Para cada ventana FWD, evalúa la estructura en estos 10 timestamps y calcula la MEDIANA
# Esto reduce el ruido de snapshots únicos y captura mejor el comportamiento intradiario
# NOTA: Requiere que los archivos históricos tengan datos en estos timestamps (tolerancia ±5 min)

# Control de FWD sobre winners/losers (filtra qué estructuras se evalúan en forward)
FWD_ON_WINNERS = False    # True: evalúa FWD en los mejores batmans (mayor valor según RANKING_MODE)
                          # False: NO evalúa FWD en winners
FWD_ON_LOSERS = False    # True: evalúa FWD en los peores batmans (menor valor según RANKING_MODE)
                          # False: NO evalúa FWD en losers
                          # NOTA: Puedes activar ambos para evaluar extremos (winners Y losers)

# Selección por porcentaje (controla cuántos batmans se procesan en FWD)
FWD_TOP_PCT = 0.15      # Porcentaje de estructuras a evaluar (valor entre 0.0 y 1.0, o string "25%")
                        # 1.0 o "100%": evalúa TODAS las estructuras filtradas (winners/losers)
                        # 0.25 o "25%": evalúa solo el top/bottom 25% según RANKING_MODE
                        # 0.10 o "10%": evalúa solo el top/bottom 10%
                        # Ejemplo: Si tienes 1000 winners y FWD_TOP_PCT=0.25, evalúa los 250 mejores

# Gráficos de promedios FWD (análisis visual de performance por ventana)
FWD_PLOT_ENABLED = False  # True: genera gráficos PNG con promedios de PnL_fwd_pct por ventana (W01, W05, etc.)
                           # False: NO genera gráficos (ahorra tiempo y espacio)
                           # Los gráficos se guardan en Desktop con sufijo del batch

# Orden global pre-FWD (ranking de estructuras antes de aplicar filtros FWD)
ORDER_PRE_FWD_GLOBAL = True              # True: ordena TODAS las estructuras globalmente antes de FWD
                                          # False: ordena por archivo (no recomendado)
RANKING_MODE = "BQI_ABS"                 # Métrica de ranking para ordenar estructuras
                                          # "BQI_ABS": Batman Quality Index Absoluto (balance spread/riesgo)
                                          # "PnLDV": Profit & Loss en Death Valley (peor escenario)
                                          # "EarScore": Earnings Score (simetría de orejas del Batman)

# ================== CONFIG PROCESO ==================
NUM_RANDOM_FILES = 1       # Número de archivos CSV a procesar aleatoriamente del directorio DATA_DIR
                             # Útil para backtests rápidos sin procesar todo el histórico
                             # Ejemplo: 2 procesa 2 días aleatorios, 0 o None procesa TODOS los archivos
THETA_TO_DAILY = 1.0/365.0   # Factor de conversión theta anual a diario (1/365)
                             # Los greeks de options vienen en base anual, esto convierte a decay diario
RISK_FREE_R = 0.04           # Tasa libre de riesgo anual (4% = 0.04)
                             # Usado como fallback en cálculos Black-Scholes si no hay tasa específica

# ================== FILTROS DE LIQUIDEZ OI/VOLUMEN ==================
# Filtros para asegurar liquidez mínima en opciones (evitar strikes ilíquidos)
# Umbrales mínimos por pata:
# - FRONT (exp1): patas cortas k1 y k3 (vencimiento cercano)
# - BACK  (exp2): pata larga k2 (vencimiento lejano)
USE_LIQUIDITY_FILTERS = False  # True: aplica filtros de OI/Volumen | False: NO filtra por liquidez
MIN_OI_FRONT  = 1              # Open Interest mínimo para opciones FRONT (k1, k3)
MIN_VOL_FRONT = 1              # Volumen mínimo para opciones FRONT (k1, k3)
MIN_OI_BACK   = 1              # Open Interest mínimo para opciones BACK (k2)
MIN_VOL_BACK  = 1              # Volumen mínimo para opciones BACK (k2)
                               # NOTA: Valores bajos (1) efectivamente deshabilitan el filtro

# ================== CONFIGURACIÓN ESTRUCTURA BATMAN ==================
# Cobertura amplia de expiraciones y strikes para maximizar el espacio de búsqueda

# === DTE (Days To Expiration) - Rangos para expiraciones ===
# Se procesan TODAS las expiraciones dentro de cada rango (sin límites de cantidad)
RANGE_A = (200, 9999)       # Rango DTE para expiration FRONT (patas cortas k1 y k3)
                           # (40, 999): DTEs entre 40 días y 999 días (long-term)
                           # Ejemplos alternativos:
                           # (0, 20): Short DTE - estructuras de muy corto plazo
                           # (40, 180): Long DTE acotado - estructuras de mediano plazo

RANGE_B = (250, 9999)       # Rango DTE para expiration BACK (pata larga k2)
                           # (60, 999): DTEs entre 60 días y 999 días
                           # IMPORTANTE: RANGE_B debe ser >= RANGE_A para Batman válido
                           # Ejemplos alternativos:
                           # (0, 40): Short DTE - para calendarios cortos
                           # (60, 2000): Long DTE extendido - incluye LEAPS

# === K1 CANDIDATOS (Strike inicial - pata corta front) ===
K1_RANGE = 500              # Rango en puntos SPX alrededor del spot para buscar K1
                           # Ejemplo: Si SPX=5000, busca K1 entre 4940 y 5060
K1_STEP  = 10               # Paso entre strikes K1 candidatos (en puntos SPX)
                           # Ejemplo: genera K1 en [4940, 4945, 4950, ..., 5055, 5060]
                           # NOTA: Menor step = más cobertura pero más estructuras a evaluar

# === VENTANAS K2/K3 (Ratios relativos a K1) — granularidad 1% ===
P2_MIN, P2_MAX = 1.00, 1.30   # K2/K1
P3_MIN, P3_MAX = 1.01, 1.70   # K3/K1
STEP_PCT = 0.005               # paso del 0.1% que en 7000 puntos SPX son 7 puntos

K2_FACTORS = [round(P2_MIN + i*STEP_PCT, 3)
              for i in range(int((P2_MAX-P2_MIN)/STEP_PCT)+1)]
K3_FACTORS = [round(P3_MIN + i*STEP_PCT, 3)
              for i in range(int((P3_MAX-P3_MIN)/STEP_PCT)+1)]


# Ejemplo de restricción al combinar (opcional):
# - Asegurar Batman “convencional”: K1 < K2 <= K3
# combos = [(f2, f3) for f2 in K2_FACTORS for f3 in K3_FACTORS if f3 >= f2]


# === PREFILTRO NET CREDIT (Crédito inicial de la estructura) ===
# Filtra estructuras por crédito neto recibido al abrir (en puntos SPX, valores negativos = crédito)
PREFILTER_CREDIT_MIN = -40   # Crédito mínimo aceptable: -40 pts = recibir mínimo $4000 por contrato
PREFILTER_CREDIT_MAX = -2    # Crédito máximo aceptable: -2 pts = recibir máximo $200 por contrato
                              # RANGO TÍPICO: [-40, -2] filtra estructuras que dan entre $200 y $4000 de crédito
                              # NOTA: Valores negativos porque crédito = entrada de dinero

# === FILTROS FINALES (Greeks y métricas de riesgo) ===
DELTA_MIN, DELTA_MAX = 0, 1      # Rango de delta total permitido
                                       # Delta negativo: estructura bajista
                                       # (0.5, 100): acepta deltas desde ligeramente bajista hasta muy alcista
THETA_MIN, THETA_MAX = 0.00, 10000   # Theta diario permitido (decay temporal)
                                       # -2.00: pérdida máxima por theta de -$200/día
                                       # 10000: sin límite superior (acepta theta positivo alto)

# === FILTRO UEL_INF (Upper Earnings Limit Infinita - Pérdida máxima en T1 con spread infinito) ===
UEL_INF_MIN = 0                # UEL infinita mínima en USD (pérdida máxima plana oreja derecha en T1)
UEL_INF_MAX = 20000                # UEL infinita máxima en USD
                                       # Pérdida máxima cuando las cortas T1 vencen y la larga doble T2 sigue viva
                                       # Fórmula: (K1+K3) - 2·PVK2 - net_cost_pts, luego × 100 (multiplicador SPX)

# === FILTRO PnLDV (Profit & Loss en Death Valley) ===
FILTER_PNLDV_ENABLED = False          # True: aplica filtro por PnLDV | False: no filtra
PNLDV_MIN = -1000000                  # PnL mínimo en Death Valley (peor punto del gráfico)
PNLDV_MAX =  1000000                  # PnL máximo en Death Valley
                                       # DESHABILITADO: rango muy amplio [-1M, +1M]

# === FILTRO BQI_ABS (Batman Quality Index Absoluto) ===
FILTER_BQI_ABS_ENABLED = True         # True: aplica filtro por BQI_ABS | False: no filtra
BQI_ABS_MIN = 2                       # BQI_ABS mínimo aceptable (métrica de calidad estructura)
BQI_ABS_MAX = 10000                   # BQI_ABS máximo (sin límite superior práctico)
                                       # FILTRO ACTIVO: solo estructuras con BQI_ABS >= 2

# === FILTRO NET_CREDIT_DIFF (solo aplica en CSV Copia con mediana T+0) ===
# Filtra estructuras comparando net_credit vs net_credit_mediana (% diferencia)
FILTER_NET_CREDIT_DIFF_ENABLED = False  # True: aplica filtro | False: no filtra
NET_CREDIT_DIFF_MIN = -15              # Diferencia mínima: -15% (net_credit puede ser 15% menor que mediana)
NET_CREDIT_DIFF_MAX =  15              # Diferencia máxima: +15% (net_credit puede ser 15% mayor que mediana)
                                        # PROPÓSITO: Descarta estructuras con net_credit anómalo vs comportamiento intradiario
                                        # EJEMPLO: Si mediana=-10 pts y net_credit=-12 pts, diff=20% → RECHAZA

# === SPREAD MÁXIMO (Control de calidad de precios) ===
MAX_SPREAD_REL = 0.5                   # Spread bid-ask máximo relativo permitido (0.5 = 50% del mid)
                                        # Descarta opciones con spread muy amplio (ilíquidas o precios sospechosos)
                                        # EJEMPLO: Si mid=10, bid=7, ask=13, spread=6, spread_rel=60% → RECHAZA

# === SALIDA Y DISPLAY ===
SHOW_MAX = 0                           # Número máximo de estructuras a mostrar por consola (0 = no muestra)
S_PNL = 8000                           # Spot SPX para gráficos de P&L (valor de referencia)

# Caché greeks
GREEKS_CACHE = {}

# ================== HELPERS ==================
def safe_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '-', text)

def get_filter_stats(df, col_name, filter_min, filter_max):
    """
    Calcula estadísticas detalladas sobre una columna antes de aplicar un filtro.

    Args:
        df: DataFrame
        col_name: nombre de la columna a analizar
        filter_min: valor mínimo del filtro configurado
        filter_max: valor máximo del filtro configurado

    Returns:
        dict con: rango_real_min, rango_real_max, num_nan, num_inf, fuera_de_rango
    """
    if col_name not in df.columns:
        return {
            'rango_real_min': None,
            'rango_real_max': None,
            'num_nan': 0,
            'num_inf': 0,
            'fuera_de_rango': 0
        }

    col_data = pd.to_numeric(df[col_name], errors='coerce')

    # Contar NaN (incluye valores que no se pudieron convertir a numérico)
    num_nan = col_data.isna().sum()

    # Contar infinitos
    num_inf = np.isinf(col_data).sum()

    # Obtener valores válidos (ni NaN ni infinito)
    valid_data = col_data[np.isfinite(col_data)]

    if len(valid_data) > 0:
        rango_real_min = float(valid_data.min())
        rango_real_max = float(valid_data.max())

        # Contar cuántos valores válidos están fuera del rango del filtro
        fuera_de_rango = ((valid_data < filter_min) | (valid_data > filter_max)).sum()
    else:
        rango_real_min = None
        rango_real_max = None
        fuera_de_rango = 0

    return {
        'rango_real_min': rango_real_min,
        'rango_real_max': rango_real_max,
        'num_nan': int(num_nan),
        'num_inf': int(num_inf),
        'fuera_de_rango': int(fuera_de_rango)
    }

def format_filter_log(filter_name, filter_min, filter_max, filas_antes, filas_despues, stats, indent="    "):
    """
    Formatea el mensaje de log para un filtro con estadísticas detalladas.

    Args:
        filter_name: nombre del filtro
        filter_min: valor mínimo configurado
        filter_max: valor máximo configurado
        filas_antes: número de filas antes del filtro
        filas_despues: número de filas después del filtro
        stats: diccionario con estadísticas de get_filter_stats
        indent: sangría para el mensaje

    Returns:
        string formateado para imprimir
    """
    # Formatear rango del filtro
    rango_filtro = f"[{filter_min};{filter_max}]"

    # Formatear filas
    cambio_filas = f"filas {filas_antes:,} → {filas_despues:,}"
    if filas_antes == filas_despues:
        cambio_filas += " (sin cambios)"

    # Formatear rango real
    if stats['rango_real_min'] is not None and stats['rango_real_max'] is not None:
        # Formatear con 2 decimales y usar coma europea
        min_str = f"{stats['rango_real_min']:.2f}".replace('.', ',')
        max_str = f"{stats['rango_real_max']:.2f}".replace('.', ',')
        rango_real = f"rango real {filter_name}: [min={min_str};max={max_str}]"
    else:
        rango_real = f"rango real {filter_name}: [N/A]"

    # Formatear estadísticas
    estadisticas = f"NaN={stats['num_nan']:,};inf={stats['num_inf']};fuera_de_rango={stats['fuera_de_rango']:,}"

    # Construir mensaje completo
    msg = f"{indent}[FILTER] {filter_name} {rango_filtro} → {cambio_filas} | {rango_real} | {estadisticas}"

    return msg

def parse_percentage(value):
    """
    Convierte un porcentaje (string "15%" o float 0.15) a float.

    Args:
        value: None, string "15%", o float 0.15

    Returns:
        float entre 0 y 1, o None
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value.endswith('%'):
            return float(value.rstrip('%')) / 100.0
        return float(value)
    return float(value)

def is_third_friday(d):
    cal = calendar.monthcalendar(d.year, d.month)
    fridays = [wk[calendar.FRIDAY] for wk in cal if wk[calendar.FRIDAY]!=0]
    return len(fridays)>=3 and d.day==fridays[2]

def root_for_exp(exp_str):
    d = datetime.strptime(exp_str, "%Y-%m-%d").date()
    if d.weekday()==4 and is_third_friday(d): return "SPX"
    if d.weekday()==3:
        nd = d + timedelta(days=1)
        if nd.weekday()==4 and is_third_friday(nd): return "SPX"
    return "SPXW"

def yyyymmdd_to_yymmdd(d): return d.strftime("%y%m%d")
def round_to_nearest_10(x): return int(round(x/10.0)*10)
def normal_cdf(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def normal_pdf(x): return math.exp(-0.5*x*x)/math.sqrt(2*math.pi)

# ================== BLACK-SCHOLES ==================
def bs_call_price(S,K,T,r,sigma,q=0.0):
    if T<=0 or sigma<=0:
        return max(S-K,0.0)
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2=d1-sigma*math.sqrt(T)
    return S*math.exp(-q*T)*normal_cdf(d1) - K*math.exp(-r*T)*normal_cdf(d2)

def bs_delta_call(S,K,T,r,sigma,q=0.0):
    if T<=0 or sigma<=0:
        return 1.0 if S>K else 0.0
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    return math.exp(-q*T)*normal_cdf(d1)

def bs_theta_call_excel(S,K,T,r,sigma):
    if T<=0 or sigma<=0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    pdf = math.exp(-0.5*d1*d1)/math.sqrt(2*math.pi)
    term1 = -(S * pdf * sigma) / (2*sqrtT)
    term2 = - r * K * math.exp(-r*T) * normal_cdf(d2)
    return term1 + term2

def bs_vega_call(S,K,T,r,sigma,q=0.0):
    if T<=0 or sigma<=0:
        return 0.0
    d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    pdf=normal_pdf(d1)
    return S*math.exp(-q*T)*pdf*math.sqrt(T)

def implied_vol_call_from_price(S, K, T, r, q, target_price,
                                lo=1e-6, hi=5.0, tol=1e-7, max_iter=100):
    """IV inversa robusta: asegura precio ≥ intrínseco, bisección con fallback Newton."""
    intrinsic = max(S - K, 0.0)
    if target_price is None:
        return float('nan')
    try:
        tp = float(target_price)
    except:
        return float('nan')
    if math.isnan(tp) or tp <= 0:
        return float('nan')

    target = max(tp, intrinsic + 1e-6)
    if T <= 0:
        return float('nan')

    def f(sig):
        return bs_call_price(S, K, T, r, sig, q) - target

    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # sin bracket: intenta Newton con vega-guard
        sigma = 0.2
        for _ in range(max_iter):
            price = bs_call_price(S, K, T, r, sigma, q)
            vega  = bs_vega_call(S, K, T, r, sigma, q)
            if vega <= 1e-12:
                break
            diff = price - target
            sigma = max(lo, min(hi, sigma - diff / max(vega, 1e-12)))
            if abs(diff) < tol:
                return sigma
        return float('nan')

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) / 2 < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2.0

# ================== LECTURA DE ARCHIVOS LOCALES ==================
def list_local_files(folder: str) -> list[Path]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")
    files = sorted(p.glob("SNAP*.csv"))
    uniq = sorted({f.resolve() for f in files if f.is_file()})
    return list(uniq)

# ================== MINUTOS ==================
def hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":"); return int(hh)*60 + int(mm)

def minutes_to_hhmm(m: int) -> str:
    hh = m // 60; mm = m % 60; return f"{hh:02d}:{mm:02d}"

def series_hhmm(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        s = df["time"].astype(str)
        hhmm = s.str.extract(r'(\b\d{2}:\d{2}\b)', expand=False)
        mask_missing = hhmm.isna()
        if mask_missing.any():
            t = pd.to_datetime(s[mask_missing], errors="coerce", utc=False)
            hhmm.loc[mask_missing] = t.dt.strftime("%H:%M")
        return hhmm
    if "ms_of_day" in df.columns:
        col = df["ms_of_day"]
        if pd.api.types.is_string_dtype(col) or col.dtype == object:
            s = col.astype(str)
            return s.str.extract(r'(\b\d{2}:\d{2}\b)', expand=False)
        ms = pd.to_numeric(col, errors="coerce")
        hh = (ms // 3_600_000).astype("Int64")
        mm = ((ms % 3_600_000) // 60_000).astype("Int64")
        return (hh.astype("float").fillna(-1).astype(int).astype(str).str.zfill(2)
                + ":" +
                mm.astype("float").fillna(-1).astype(int).astype(str).str.zfill(2))
    raise KeyError("No existe 'time' ni 'ms_of_day' en el CSV.")

def filter_df_by_hhmm(df: pd.DataFrame, hhmm: str) -> pd.DataFrame:
    if "time" in df.columns:
        s = df["time"].astype(str)
        return df[s.str.startswith(hhmm + ":")].copy()
    if "ms_of_day" in df.columns:
        col = df["ms_of_day"]
        if pd.api.types.is_string_dtype(col) or col.dtype == object:
            s = col.astype(str)
            return df[s.str.startswith(hhmm + ":")].copy()
        ms = pd.to_numeric(col, errors="coerce")
        hh = (ms // 3_600_000).astype(int)
        mm = ((ms % 3_600_000) // 60_000).astype(int)
        mask = (pd.Series(hh, index=df.index).astype(str).str.zfill(2) + ":" +
                pd.Series(mm, index=df.index).astype(str).str.zfill(2)) == hhmm
        return df[mask].copy()
    raise KeyError("No existe 'time' ni 'ms_of_day' en el CSV.")

def get_slice_at_target_hhmm(df: pd.DataFrame, target_hhmm: str, tolerance_min: int = 0):
    hhmm_series = series_hhmm(df).dropna().astype(str)
    if (hhmm_series == target_hhmm).any():
        return filter_df_by_hhmm(df, target_hhmm), target_hhmm
    if tolerance_min and tolerance_min > 0:
        valid = [x for x in pd.unique(hhmm_series) if re.match(r"^\d{2}:\d{2}$", str(x))]
        if valid:
            target_m = hhmm_to_minutes(target_hhmm)
            candidates = sorted(valid, key=lambda x: abs(hhmm_to_minutes(str(x)) - target_m))
            best = candidates[0]
            if abs(hhmm_to_minutes(best) - target_m) <= tolerance_min:
                return filter_df_by_hhmm(df, best), best
    return pd.DataFrame(), target_hhmm

def resolve_used_hhmm(used_hhmm_us: str, df_one_min: pd.DataFrame) -> str:
    try:
        if used_hhmm_us != "ALL" and re.match(r"^\d{2}:\d{2}$", str(used_hhmm_us)):
            return str(used_hhmm_us)
        if df_one_min is not None and not df_one_min.empty:
            if "ms_of_day" in df_one_min.columns:
                s = df_one_min["ms_of_day"]
                if s.dtype == object:
                    m = re.search(r"\b(\d{2}:\d{2})\b", str(s.iloc[0]))
                    if m:
                        return m.group(1)
                try:
                    ms = int(pd.to_numeric(s.iloc[0], errors="coerce"))
                    hh = ms // 3_600_000
                    mm = (ms % 3_600_000) // 60_000
                    if 0 <= hh <= 23 and 0 <= mm <= 59:
                        return f"{int(hh):02d}:{int(mm):02d}"
                except Exception:
                    pass
            if "time" in df_one_min.columns:
                val = str(df_one_min["time"].iloc[0])
                m = re.search(r"\b(\d{2}:\d{2})\b", val)
                if m:
                    return m.group(1)
        return "12:00"
    except Exception:
        return "12:00"

# ================== ESCALA STRIKE ==================
def td_detect_scale_for_strike(df: pd.DataFrame) -> int:
    try:
        med = pd.to_numeric(df.get("strike", pd.Series(dtype=float)), errors="coerce").median()
        if med and med > 100000: return 1000
    except: pass
    return 1

def td_rescale_strike_only(df: pd.DataFrame) -> pd.DataFrame:
    if "strike" in df.columns:
        scale = td_detect_scale_for_strike(df)
        if scale != 1:
            df = df.copy()
            df["strike"] = pd.to_numeric(df["strike"], errors="coerce") / float(scale)
    return df

# ================== CANONICAL CHAIN (sin descartar por spread en selección) ==================
def pick_best_row(rows: pd.DataFrame) -> pd.Series | None:
    """
    Elige la 'mejor' fila dentro del trío, **sin** descartar por spread relativo ni mid inválido.
    Criterios de ordenación (desc): volumen, open_interest, ms_of_day.
    Si no existen estas columnas, devuelve la primera fila.
    """
    if rows is None or rows.empty:
        log_inc("pick_empty_rows")
        return None
    tmp = rows.copy()
    for c in ("volume","open_interest","ms_of_day"):
        if c in tmp.columns: tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    sort_cols = []; ascending = []
    for col in ("volume","open_interest","ms_of_day"):
        if col in tmp.columns: sort_cols.append(col); ascending.append(False)
    if sort_cols:
        tmp = tmp.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")
    log_add("pick_candidates_in_trio", len(tmp))
    if len(tmp) > 1:
        log_add("pick_dupes_resolved", len(tmp) - 1)
    log_inc("pick_kept_best")
    return tmp.iloc[0]

def build_canonical_chain(df_full: pd.DataFrame) -> pd.DataFrame:
    needed = {"right","expiration","strike","bid","ask"}
    if not needed.issubset(df_full.columns):
        log_inc("canonical_missing_cols")
        return pd.DataFrame()

    out = []
    for (r,e,k), sub in df_full.groupby(["right","expiration","strike"], dropna=False):
        log_inc("canonical_groups_total")
        best = pick_best_row(sub)
        if best is not None:
            out.append(best.to_dict())
        else:
            log_inc("canonical_discard_trio")

    can = pd.DataFrame(out)
    if not can.empty:
        pre = len(can)
        for c in ("strike","bid","ask"):
            if c in can.columns: can[c] = pd.to_numeric(can[c], errors="coerce")
        can = can.dropna(subset=["right","expiration","strike","bid","ask"]).reset_index(drop=True)
        log_add("canonical_dropna_removed", pre - len(can))

    log_add("canonical_kept", 0 if can is None else len(can))
    return can

# ================== CADENAS / DTE / MID ==================
def build_chains(df_one_minute: pd.DataFrame):
    expirations = sorted(pd.unique(df_one_minute["expiration"].astype(str)))
    precache = {}
    for exp in expirations:
        sub = df_one_minute[df_one_minute["expiration"].astype(str)==exp].copy()
        if "strike" in sub.columns:
            sub["strike"] = pd.to_numeric(sub["strike"], errors="coerce")
        calls = sub[sub["right"].str.upper()=="C"].copy()
        puts  = sub[sub["right"].str.upper()=="P"].copy()
        for dfp in (calls, puts):
            dfp.dropna(subset=["strike"], inplace=True)
            dfp.sort_values("strike", inplace=True)
            dfp.reset_index(drop=True, inplace=True)
        # Crear índices por strike para O(1) lookup
        calls_idx = calls.set_index("strike", drop=False) if not calls.empty else pd.DataFrame()
        puts_idx  = puts.set_index("strike", drop=False) if not puts.empty else pd.DataFrame()
        precache[exp] = (calls, puts, calls_idx, puts_idx)
    return expirations, precache

def compute_dte_map(expirations, date_et_str: str):
    date_et = datetime.strptime(date_et_str, "%Y-%m-%d").date()
    return {e: (datetime.strptime(e,"%Y-%m-%d").date() - date_et).days for e in expirations}

def get_mid_from_row(row):
    def f(x):
        try:
            v=float(x); return v if np.isfinite(v) else np.nan
        except: return np.nan
    mv = f(row.get("mid", np.nan))
    if not np.isnan(mv) and mv>0: return mv
    b = f(row.get("bid", np.nan)); a = f(row.get("ask", np.nan))
    if not np.isnan(b) and not np.isnan(a) and a>0: return (a+b)/2.0
    lp = f(row.get("last", np.nan))
    if np.isnan(lp): lp = f(row.get("lastPrice", np.nan))
    return lp

def get_root_from_series(row):
    if row is None:
        return None
    if hasattr(row, "get"):
        for key in ("root", "root_symbol", "rootSymbol", "Root", "ROOT"):
            try:
                val = row.get(key)
            except Exception:
                val = None
            if isinstance(val, str):
                val = val.strip()
                if val:
                    return val.upper()
    return None

def filter_df_by_root_strict(df: pd.DataFrame, expected_root: str):
    """
    Filtro ESTRICTO por root con preferencia, sin fallback silencioso.

    Returns:
        tuple: (filtered_df, root_pref_applied: bool, chosen_root: str|None)
        - Si hay match exacto con expected_root → (df_preferido, True, expected_root)
        - Si NO hay match → (df_original, False, root_encontrado_o_None)
    """
    if expected_root is None or df is None or df.empty:
        return df, False, None

    try:
        expected_norm = str(expected_root).strip().upper()
    except Exception:
        return df, False, None

    if not expected_norm:
        return df, False, None

    # Buscar columna de root
    root_col = None
    for key in ("root", "root_symbol", "rootSymbol", "Root", "ROOT"):
        if key in df.columns:
            root_col = key
            break

    if root_col is None:
        return df, False, None

    # Normalizar columna root
    df_work = df.copy()
    df_work["__root_norm"] = df_work[root_col].astype(str).str.strip().str.upper()

    # Filtrar por preferencia
    pref_mask = df_work["__root_norm"] == expected_norm
    df_pref = df_work.loc[pref_mask]

    if not df_pref.empty:
        # Match exacto → aplicar preferencia
        df_result = df_pref.drop(columns=["__root_norm"])
        return df_result, True, expected_norm
    else:
        # NO match → devolver original SIN filtrar, pero señalar anomalía
        chosen = df_work["__root_norm"].iloc[0] if len(df_work) > 0 else None
        df_result = df.copy()  # Sin __root_norm
        return df_result, False, chosen


def select_best_strike_row(df_strike: pd.DataFrame, base_mid: float = None):
    """
    Selecciona la mejor fila cuando hay múltiples con el mismo strike.

    Criterios de selección (en orden):
    1. Mejor liquidez: (ask-bid)/abs(mid) más pequeño, con bid>0 & ask>0
    2. Si empate: mid más cercano al base_mid (si se proporciona)
    3. Si aún empate: primera fila

    Returns:
        tuple: (row: pd.Series, tie_break: str)
        tie_break puede ser: "unique", "liquidity", "mid_proximity", "first"
    """
    if df_strike is None or df_strike.empty:
        return None, None

    if len(df_strike) == 1:
        return df_strike.iloc[0], "unique"

    # Calcular spread relativo para liquidez
    df_work = df_strike.copy()

    def calc_spread_rel(row):
        try:
            bid = float(row.get("bid", np.nan))
            ask = float(row.get("ask", np.nan))
            if not (np.isfinite(bid) and np.isfinite(ask)):
                return np.inf
            if bid <= 0 or ask <= 0:
                return np.inf
            mid = 0.5 * (bid + ask)
            if mid <= 0:
                return np.inf
            return (ask - bid) / abs(mid)
        except Exception:
            return np.inf

    df_work["__spread_rel"] = df_work.apply(calc_spread_rel, axis=1)

    # Filtrar solo filas con spread válido
    valid = df_work.loc[df_work["__spread_rel"] < np.inf]

    if valid.empty:
        # Ninguna tiene liquidez válida → tomar primera
        return df_strike.iloc[0], "first"

    # Ordenar por spread relativo (menor = mejor liquidez)
    valid = valid.sort_values("__spread_rel")

    if len(valid) == 1:
        row = df_strike.loc[df_strike.index == valid.index[0]].iloc[0]
        return row, "liquidity"

    # Empate en liquidez → desempatar por mid más cercano a base_mid
    best_spread = valid["__spread_rel"].iloc[0]
    tied = valid.loc[valid["__spread_rel"] == best_spread]

    if len(tied) == 1:
        row = df_strike.loc[df_strike.index == tied.index[0]].iloc[0]
        return row, "liquidity"

    if base_mid is not None and np.isfinite(base_mid):
        def calc_mid(row):
            try:
                bid = float(row.get("bid", np.nan))
                ask = float(row.get("ask", np.nan))
                if np.isfinite(bid) and np.isfinite(ask):
                    return 0.5 * (bid + ask)
                return np.nan
            except Exception:
                return np.nan

        tied["__mid"] = tied.apply(calc_mid, axis=1)
        tied = tied.loc[tied["__mid"].notna()]

        if not tied.empty:
            tied["__mid_diff"] = (tied["__mid"] - base_mid).abs()
            tied = tied.sort_values("__mid_diff")
            row = df_strike.loc[df_strike.index == tied.index[0]].iloc[0]
            return row, "mid_proximity"

    # Último recurso: primera fila de las empatadas
    row = df_strike.loc[df_strike.index == tied.index[0]].iloc[0]
    return row, "liquidity"

# ================== SPOT & r ==================
FRED_DGS1MO_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO"

def underlying_from_slice(df_slice: pd.DataFrame) -> float:
    if df_slice is None or df_slice.empty:
        return float('nan')
    candidates = ["underlying_price", "Underlying_Price", "underlyingPrice", "underlying", "spot", "SPX", "spx"]
    for col in candidates:
        if col in df_slice.columns:
            s = pd.to_numeric(df_slice[col], errors="coerce").dropna()
            if not s.empty:
                return float(s.median())
    return float('nan')

def fetch_risk_free_historical_fred(target_date: str, fallback: float=RISK_FREE_R) -> float:
    """
    Descarga el CSV histórico de FRED DGS1MO y devuelve el valor correspondiente
    a la fecha target_date (formato "YYYY-MM-DD"), tomando la última observación
    disponible <= target_date.

    Args:
        target_date: Fecha objetivo en formato "YYYY-MM-DD" (ej: "2023-06-15")
        fallback: Valor por defecto si no se encuentra dato válido

    Returns:
        Tasa libre de riesgo como decimal (ej: 0.05 para 5%)
    """
    try:
        with urllib.request.urlopen(FRED_DGS1MO_URL, timeout=10) as r:
            data = r.read()
        text = None
        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = data.decode(enc); break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = data.decode("utf-8", errors="ignore")

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pat = re.compile(r"^(\d{4}-\d{2}-\d{2}),\s*([+-]?\d+(?:\.\d+)?)\s*$")

        # Parsear target_date
        target_dt = datetime.strptime(target_date, "%Y-%m-%d").date()

        # Buscar todas las observaciones <= target_date, guardar la más reciente
        best_val = None
        best_date = None

        for ln in lines:
            m = pat.match(ln)
            if not m:
                continue
            date_str, val_str = m.groups()
            obs_date = datetime.strptime(date_str, "%Y-%m-%d").date()

            # Solo considerar fechas <= target_date
            if obs_date <= target_dt:
                val = float(val_str) / 100.0  # Convertir de % a decimal
                # Actualizar si es la primera o si encontramos fecha más reciente
                if best_date is None or obs_date > best_date:
                    best_date = obs_date
                    best_val = val

        # Validar resultado
        if best_val is None or not (best_val == best_val):  # None o NaN
            return float(fallback)
        if best_val < 0 or best_val > 1.0:  # Validación de rango razonable
            return float(fallback)

        return float(best_val)

    except Exception:
        return float(fallback)

# ================== FILTROS DE COTIZACIÓN ==================
def leg_quote_ok(df_chain: pd.DataFrame, K: float, max_spread_rel: float = MAX_SPREAD_REL) -> bool:
    if df_chain is None or df_chain.empty:
        log_inc("leg_df_empty"); return False
    Ks = pd.to_numeric(df_chain["strike"], errors="coerce")
    r = df_chain.loc[np.isclose(Ks, float(K), atol=1e-6)]
    if r.empty:
        log_inc("leg_strike_not_found"); return False

    # Usar selección inteligente en vez de iloc[0] ciego
    row, _ = select_best_strike_row(r, base_mid=None)
    if row is None:
        log_inc("leg_selection_failed"); return False

    try:
        bid = float(row.get("bid", np.nan))
        ask = float(row.get("ask", np.nan))
    except Exception:
        log_inc("leg_num_error"); return False
    if not (np.isfinite(bid) and np.isfinite(ask)):
        log_inc("leg_bidask_nonfinite"); return False
    if bid <= 0:
        log_inc("leg_bid_le_0"); return False
    if ask <= 0:
        log_inc("leg_ask_le_0"); return False
    if ask < bid:
        log_inc("leg_ask_lt_bid"); return False
    mid = 0.5*(bid + ask)
    if not np.isfinite(mid) or mid <= 0:
        log_inc("leg_mid_nonfinite"); return False
    spread_rel = (ask - bid) / mid
    if not np.isfinite(spread_rel) or spread_rel > float(max_spread_rel):
        log_inc("leg_spread_too_high"); return False
    log_inc("leg_ok"); return True

# ================== UTILS BÚSQUEDA ==================
def pick_nearest_dte(exp_list, dte_map, target, nmax):
    if nmax is None or nmax<=0: return exp_list
    exps = sorted(exp_list, key=lambda e: abs(dte_map[e]-target))
    return exps[:nmax]

def nearest(lst, x):
    return min(lst, key=lambda v: abs(v-x)) if lst else None

def sample_strikes(available_strikes, spot, factors):
    out=[]
    for f in factors:
        t = spot*f
        s = nearest(available_strikes, t)
        if s is not None and s not in out:
            out.append(s)
    return out

def gen_k1_candidates(base, rng=30, step=10):
    return [base + o for o in range(-rng, rng+1, step)]

def fetch_row(idx: pd.DataFrame, K: float):
    """Devuelve la fila de índice 'K' (Series). Si duplicado, toma la primera."""
    try:
        r = idx.loc[K]
    except KeyError:
        return None
    if isinstance(r, pd.DataFrame):
        if r.empty: return None
        return r.iloc[0]
    return r

def quick_mid_idx(idx, K):
    row = fetch_row(idx, K)
    if row is None: return np.nan
    return get_mid_from_row(row)

def leg_liquidity_ok_idx(idx: pd.DataFrame, K: float, min_oi: int, min_vol: int) -> bool:
    row = fetch_row(idx, K)
    if row is None: return False
    oi  = pd.to_numeric(row.get("openInterest", np.nan), errors="coerce")
    vol = pd.to_numeric(row.get("volume", np.nan), errors="coerce")
    oi  = 0 if (not np.isfinite(oi))  else float(oi)
    vol = 0 if (not np.isfinite(vol)) else float(vol)
    return (oi >= float(min_oi)) and (vol >= float(min_vol))

# ================== URL ==================
def make_url(exp1,k1,exp2,k2,k3):
    ymd1=yyyymmdd_to_yymmdd(datetime.strptime(exp1,"%Y-%m-%d"))
    ymd2=yyyymmdd_to_yymmdd(datetime.strptime(exp2,"%Y-%m-%d"))
    r1,r2=root_for_exp(exp1),root_for_exp(exp2)
    legs=[f".{r1}{ymd1}C{int(k1)}x-1", f".{r2}{ymd2}C{int(k2)}x2", f".{r1}{ymd1}C{int(k3)}x-1"]
    return BASE_URL + ",".join(legs)

# ================== REVALORIZACIÓN BATMAN (FWD) ==================
def normalize_root_value(val):
    """Normaliza valor de root a formato estándar."""
    if val is None:
        return None
    try:
        s = str(val).strip().upper()
        return s if s else None
    except Exception:
        return None

def compare_mid_values(mid_fwd, mid_base):
    """Compara mid forward vs mid base para validación."""
    if mid_fwd is None or mid_base is None:
        return "MISSING"
    if not (np.isfinite(mid_fwd) and np.isfinite(mid_base)):
        return "NONFINITE"
    try:
        diff_pct = abs(float(mid_fwd) - float(mid_base)) / max(abs(float(mid_base)), 1e-6) * 100.0
        if diff_pct < 5.0:
            return "CERCANO"
        elif diff_pct < 15.0:
            return "MODERADO"
        else:
            return "DISTANTE"
    except Exception:
        return "ERROR"

def batman_value_from_df(one_min_df: pd.DataFrame, exp1: str, k1: float, exp2: str, k2: float, k3: float, root1=None, root2=None):
    """
    Revaloriza estructura Batman: -1C@k1(exp1) +2C@k2(exp2) -1C@k3(exp1)

    Args:
        one_min_df: DataFrame con snapshot del mercado
        exp1: Expiración de k1 y k3 (front, short legs)
        k1: Strike de primera call corta
        exp2: Expiración de k2 (back, long leg)
        k2: Strike de call larga (×2)
        k3: Strike de segunda call corta
        root1: Root esperado para exp1 (SPX/SPXW)
        root2: Root esperado para exp2 (SPX/SPXW)

    Returns:
        dict con net_credit (puntos SPX), leg1/leg2/leg3 (bid/ask/mid/root/strike)
        o None si no se puede revalorizar
    """
    def _to_float(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        return v if np.isfinite(v) else None

    def _to_index(idx_val):
        try:
            return int(idx_val)
        except (TypeError, ValueError):
            return idx_val

    def _get_root(row):
        for key in ("root", "root_symbol", "rootSymbol", "Root", "ROOT"):
            val = row.get(key) if hasattr(row, "get") else None
            if isinstance(val, str) and val.strip():
                return val.strip().upper()
        return None

    try:
        # Filtrar por right=C (calls), expiración, y root
        sub1 = one_min_df[(one_min_df["right"].str.upper()=="C") & (one_min_df["expiration"].astype(str)==exp1)]
        sub2 = one_min_df[(one_min_df["right"].str.upper()=="C") & (one_min_df["expiration"].astype(str)==exp2)]

        # Aplicar filtro estricto de root con preferencia
        sub1_filt, root1_pref_applied, root1_chosen = filter_df_by_root_strict(sub1.copy(), root1)
        sub2_filt, root2_pref_applied, root2_chosen = filter_df_by_root_strict(sub2.copy(), root2)

        # Log de anomalías de root (solo si no hubo match exacto)
        if root1 is not None and not root1_pref_applied and root1_chosen is not None:
            expected_norm = normalize_root_value(root1)
            if expected_norm != root1_chosen:
                pass  # Anomalía detectada, metadata ya registrada en leg
        if root2 is not None and not root2_pref_applied and root2_chosen is not None:
            expected_norm = normalize_root_value(root2)
            if expected_norm != root2_chosen:
                pass  # Anomalía detectada, metadata ya registrada en leg

        sub1_filt["strike"] = pd.to_numeric(sub1_filt["strike"], errors="coerce")
        sub2_filt["strike"] = pd.to_numeric(sub2_filt["strike"], errors="coerce")

        # Buscar strikes k1 y k3 en exp1
        r1 = sub1_filt.loc[np.isclose(sub1_filt["strike"], float(k1), atol=1e-6)]
        r3 = sub1_filt.loc[np.isclose(sub1_filt["strike"], float(k3), atol=1e-6)]

        # Buscar strike k2 en exp2
        r2 = sub2_filt.loc[np.isclose(sub2_filt["strike"], float(k2), atol=1e-6)]

        if r1.empty or r2.empty or r3.empty:
            return None

        # Selección inteligente de la mejor fila (sin iloc[0] ciego)
        row1, tie1 = select_best_strike_row(r1, base_mid=None)
        row2, tie2 = select_best_strike_row(r2, base_mid=None)
        row3, tie3 = select_best_strike_row(r3, base_mid=None)

        if row1 is None or row2 is None or row3 is None:
            return None

        mid1 = get_mid_from_row(row1)
        mid2 = get_mid_from_row(row2)
        mid3 = get_mid_from_row(row3)

        if not np.isfinite(mid1) or not np.isfinite(mid2) or not np.isfinite(mid3):
            return None

        def _build_leg(row, idx, mid, tie_break, root_pref_applied, root_chosen):
            leg_root = _get_root(row)
            return {
                "row_index": _to_index(idx),
                "bid": _to_float(row.get("bid")),
                "ask": _to_float(row.get("ask")),
                "mid": float(mid),
                "root": leg_root,
                "strike": _to_float(row.get("strike")),
                "tie_break": tie_break,
                "root_pref_applied": root_pref_applied,
                "root_chosen": root_chosen,
                "root_expected": normalize_root_value(root1) if idx in r1.index or idx in r3.index else normalize_root_value(root2)
            }

        leg1 = _build_leg(row1, r1.index[0] if len(r1) > 0 else None, mid1, tie1, root1_pref_applied, root1_chosen)
        leg2 = _build_leg(row2, r2.index[0] if len(r2) > 0 else None, mid2, tie2, root2_pref_applied, root2_chosen)
        leg3 = _build_leg(row3, r3.index[0] if len(r3) > 0 else None, mid3, tie3, root1_pref_applied, root1_chosen)

        # net_credit = -mid1 + 2*mid2 - mid3 (puntos SPX)
        # CREDIT negativo = cobramos (favorable)
        net_credit = float(-mid1 + 2*mid2 - mid3)

        return {
            "net_credit": net_credit,
            "leg1": leg1,
            "leg2": leg2,
            "leg3": leg3
        }
    except Exception:
        return None

# ================== CÁLCULO ESTRATEGIA ==================
def compute_strategy_metrics(spot,r_base,exp1,dte1,k1,exp2,dte2,k2,k3,precache):
    calls1, _, calls1_idx, _ = precache[exp1]
    calls2, _, calls2_idx, _ = precache[exp2]

    T1 = max(dte1,1)/365.0
    T2 = max(dte2,1)/365.0
    tau = max(T2 - T1, 0.0)
    r1=r_base; r2=r_base; q1=q2=0.0

    def greeks_price(idx: pd.DataFrame, exp, K, T, r, q):
        key=("call",exp,float(K),round(T,6))
        if key in GREEKS_CACHE:
            return GREEKS_CACHE[key]
        row = fetch_row(idx, K)
        if row is None:
            GREEKS_CACHE[key]=(np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
            return GREEKS_CACHE[key]
        bid = float(row.get("bid",np.nan))
        ask = float(row.get("ask",np.nan))
        last = float(row.get("lastPrice",np.nan))
        if math.isnan(last):
            last = float(row.get("last",np.nan))
        mid = (bid+ask)/2.0 if (not math.isnan(bid) and not math.isnan(ask) and ask>0) else (last if not math.isnan(last) else np.nan)
        if math.isnan(mid) or T<=0:
            GREEKS_CACHE[key]=(np.nan,np.nan,mid,np.nan,bid,ask,last)
            return GREEKS_CACHE[key]
        iv = implied_vol_call_from_price(spot, K, T, r, q, mid)
        if (iv is None) or (isinstance(iv,float) and (math.isnan(iv) or iv<=0)):
            GREEKS_CACHE[key]=(np.nan,np.nan,mid,iv,bid,ask,last)
            return GREEKS_CACHE[key]
        delta = bs_delta_call(spot,K,T,r,iv,q)
        theta_annual_long = bs_theta_call_excel(spot,K,T,r,iv)
        theta_daily = theta_annual_long * THETA_TO_DAILY  # Por contrato
        GREEKS_CACHE[key]=(delta,theta_daily,mid,iv,bid,ask,last)
        return GREEKS_CACHE[key]

    d1,t1,p1,iv1,bid1,ask1,last1 = greeks_price(calls1_idx,exp1,k1,T1,r1,q1)   # -1C
    d2,t2,p2,iv2,bid2,ask2,last2 = greeks_price(calls2_idx,exp2,k2,T2,r2,q2)   # +2C
    d3,t3,p3,iv3,bid3,ask3,last3 = greeks_price(calls1_idx,exp1,k3,T1,r1,q1)   # -1C

    dz = lambda x: 0.0 if (x is None or (isinstance(x,float) and math.isnan(x))) else x
    delta_total = -1*dz(d1) + 2*dz(d2) - 1*dz(d3)
    theta_total = -1*dz(t1) + 2*dz(t2) - 1*dz(t3)

    p1z, p2z, p3z = dz(p1), dz(p2), dz(p3)
    net_credit = (-1*p1z) + (2*p2z) - (1*p3z)  # puntos SPX

    def clean_sigma(sig, default=0.20):
        try:
            return default if (sig is None or np.isnan(sig) or sig<=0) else float(sig)
        except:
            return default

    def bs_call_price_safe(S,K,T,r,sigma):
        return bs_call_price(S,K,T,r,sigma,q=0.0)

    def pnl_leg_at_S_T(q, K, P, S, T_rem, r, sigma):
        sigma = clean_sigma(sigma)
        if T_rem <= 1e-10:
            value = max(S - K, 0.0)
        else:
            value = bs_call_price_safe(S, K, T_rem, r, sigma)
        return q * 100.0 * (value - P)

    T_rem_long = max(T2 - T1, 0.0)
    pnl_short1 = pnl_leg_at_S_T(-1, k1, p1z, S_PNL, 0.0,        r1, iv1)
    pnl_long2  = pnl_leg_at_S_T(+2, k2, p2z, S_PNL, T_rem_long, r2, iv2)
    pnl_short3 = pnl_leg_at_S_T(-1, k3, p3z, S_PNL, 0.0,        r1, iv3)
    pnl_total  = pnl_short1 + pnl_long2 + pnl_short3

    # Death Valley + PnLDV (puntos SPX)
    death_valley = None
    pnl_dv_points = None
    tau = max(T2 - T1, 0.0)
    if tau > 0 and (iv2 is not None) and not (isinstance(iv2,float) and math.isnan(iv2)):
        sigma2 = float(iv2)
        S0 = float(k2) * math.exp(-(r2 + 0.5*sigma2*sigma2) * tau)
        k_lo, k_hi = (min(k1,k3), max(k1,k3))
        if (S0 >= k_lo) and (S0 < k_hi):
            val_short1 = -max(0.0, S0 - float(k1))
            val_short3 = -max(0.0, S0 - float(k3))
            val_long2  = 2.0 * bs_call_price_safe(S0, float(k2), tau, r2, sigma2)
            value_t1   = val_short1 + val_short3 + val_long2
            pnl_dv_points = value_t1 - net_credit
            death_valley  = S0
        else:
            value_lim = (float(k1) + float(k3)) - 2.0*float(k2)*math.exp(-r2*tau)
            pnl_dv_points = value_lim - net_credit
            death_valley  = float('nan')

    # Orejas (puntos SPX)
    iv2_clean = None if (iv2 is None or (isinstance(iv2,float) and math.isnan(iv2))) else float(iv2)
    EarL_pts = None; EarR_pts = None
    if tau > 0 and iv2_clean is not None:
        EarL_pts = 2.0*bs_call_price_safe(float(k1), float(k2), tau, r2, iv2_clean) - net_credit
        EarR_pts = (-(float(k3) - float(k1))) + 2.0*bs_call_price_safe(float(k3), float(k2), tau, r2, iv2_clean) - net_credit

    # UEL Infinita: pérdida máxima plana de la oreja derecha al llegar T1
    # Las dos cortas T1 vencen, la larga doble T2 sigue viva
    # Fórmula correcta: valor_limite - net_credit, donde valor_limite = (K1 + K3) - 2·PVK2
    # PVK2 = K2·exp(-r·ΔT) es el valor presente del strike K2 descontado de T2 a T1
    UEL_inf_USD = None
    if tau > 0:
        # Calcular el strike descontado de la larga (K2)
        PVK2 = float(k2) * math.exp(-r2 * tau)
        # Pérdida máxima plana en puntos: valor_limite - net_credit
        # Si net_credit < 0 (recibimos crédito), -net_credit suma a la UEL (peor pérdida)
        # Si net_credit > 0 (pagamos débito), -net_credit resta a la UEL (mejor pérdida)
        UEL_inf_pts = (float(k1) + float(k3)) - 2.0*PVK2 - net_credit
        # Convertir a USD (multiplicador 100 para SPX)
        UEL_inf_USD = round(100.0 * UEL_inf_pts, 2)
    elif tau <= 0:
        # Si no hay diferencia temporal, usar K2 sin descuento
        PVK2 = float(k2)
        UEL_inf_pts = (float(k1) + float(k3)) - 2.0*PVK2 - net_credit
        UEL_inf_USD = round(100.0 * UEL_inf_pts, 2)

    iv1_fmt = None if (iv1 is None or (isinstance(iv1,float) and math.isnan(iv1))) else round(float(iv1),6)
    iv2_fmt = None if (iv2 is None or (isinstance(iv2,float) and math.isnan(iv2))) else round(float(iv2),6)
    iv3_fmt = None if (iv3 is None or (isinstance(iv3,float) and math.isnan(iv3))) else round(float(iv3),6)

    out = {
        "delta_total": round(delta_total,6),
        "theta_total": round(theta_total,6),
        "net_credit": round(net_credit,4),

        "k1": int(k1), "k2": int(k2), "k3": int(k3),

        "iv_k1": iv1_fmt,
        "iv_k2": iv2_fmt,
        "iv_k3": iv3_fmt,

        "Death valley": None if (death_valley is None or (isinstance(death_valley,float) and math.isnan(death_valley))) else round(float(death_valley),2),
        "PnLDV": None if (pnl_dv_points is None or (isinstance(pnl_dv_points,float) and math.isnan(pnl_dv_points))) else round(float(pnl_dv_points),2),

        "EarL": None if (EarL_pts is None or (isinstance(EarL_pts,float) and math.isnan(EarL_pts))) else round(float(EarL_pts),2),
        "EarR": None if (EarR_pts is None or (isinstance(EarR_pts,float) and math.isnan(EarR_pts))) else round(float(EarR_pts),2),

        "DTE1/DTE2": f"{int(round(dte1))}/{int(round(dte2))}",

        "delta_k1": None if math.isnan(d1) else round(d1,6),
        "theta_k1": None if math.isnan(t1) else round(t1,6),
        "delta_k2": None if math.isnan(d2) else round(d2,6),
        "theta_k2": None if math.isnan(t2) else round(t2,6),
        "delta_k3": None if math.isnan(d3) else round(d3,6),
        "theta_k3": None if math.isnan(t3) else round(t3,6),

        "price_bid_short1": None if math.isnan(bid1) else round(bid1,4),
        "price_ask_short1": None if math.isnan(ask1) else round(ask1,4),
        "price_last_short1": None if math.isnan(last1) else round(last1,4),
        "price_mid_short1":  None if math.isnan(p1)   else round(p1,4),

        "price_bid_long2": None if math.isnan(bid2) else round(bid2,4),
        "price_ask_long2": None if math.isnan(ask2) else round(ask2,4),
        "price_last_long2": None if math.isnan(last2) else round(last2,4),
        "price_mid_long2":  None if math.isnan(p2)   else round(p2,4),

        "price_bid_short3": None if math.isnan(bid3) else round(bid3,4),
        "price_ask_short3": None if math.isnan(ask3) else round(ask3,4),
        "price_last_short3": None if math.isnan(last3) else round(last3,4),
        "price_mid_short3":  None if math.isnan(p3)   else round(p3,4),

        "pnl8000_short1": round(pnl_short1,2),
        "pnl8000_long2":  round(pnl_long2,2),
        "pnl8000_short3": round(pnl_short3,2),
        "pnl8000_total":  round(pnl_total,2),

        "UEL_inf_USD": UEL_inf_USD,
    }
    return out

# ================== PARALELIZACIÓN: WORKER PARA CALCULAR FWD (Mediana Intraday 14×30min) ==================
def _process_one_fwd_batman(args):
    """
    Worker function para calcular FWD de un batman en paralelo.
    Evalúa la estructura en 14 timestamps fijos de mercado US y calcula mediana de PnL.
    Debe ser top-level para ser serializable por ProcessPoolExecutor.

    Args:
        args: tupla (i, row_dict, files_sorted_list, forward_fracs, frac_suffixes,
                     tz_us_str, tz_es_str, nearest_minute_tol, intraday_timestamps)

    Returns:
        dict: resultados FWD para todas las ventanas de este batman (incluye medianas)
    """
    (i, row_dict, files_sorted_list, forward_fracs, frac_suffixes,
     tz_us_str, tz_es_str, nearest_minute_tol, intraday_timestamps) = args

    # Reconstruir ZoneInfo desde strings
    TZ_US_local = ZoneInfo(tz_us_str)
    TZ_ES_local = ZoneInfo(tz_es_str)

    # Función local para buscar archivo forward
    def forward_file_by_offset_local(row_base_idx: int, offset: int):
        idx = row_base_idx + offset
        if 0 <= idx < len(files_sorted_list):
            return files_sorted_list[idx]
        return None

    result = {
        'index': i,
        'success': False,
        'filled': 0,
        'fwd_data': {},
        'errors': []
    }

    try:
        # Parsear DTE1 desde "DTE1/DTE2"
        dte_str = str(row_dict.get("DTE1/DTE2", "0/0"))
        t1_days = int(dte_str.split("/")[0])
    except Exception as e:
        result['errors'].append(('fwd_dte_parse_error', str(e)))
        return result

    if not t1_days or t1_days <= 0:
        result['errors'].append(('fwd_invalid_dte', f'DTE1={t1_days}'))
        return result

    try:
        spot_base = float(row_dict.get("SPX"))
    except Exception:
        spot_base = float('nan')

    # Extraer root values del batman base (una sola vez)
    base_root_exp1_norm = normalize_root_value(row_dict.get("root_exp1"))
    base_root_exp2_norm = normalize_root_value(row_dict.get("root_exp2"))
    base_credit = float(row_dict["net_credit"])

    filled = 0
    for fr, suf in zip(forward_fracs, frac_suffixes):
        forward_steps = max(1, int(round(t1_days * fr)))
        if forward_steps >= t1_days:
            continue

        fwd_path = forward_file_by_offset_local(int(row_dict["__base_idx"]), forward_steps)
        if fwd_path is None or not fwd_path.exists():
            continue

        try:
            m2 = re.search(r"(\d{4}-\d{2}-\d{2})", fwd_path.name)
            if not m2:
                continue
            fwd_date_str_us = m2.group(1)

            # ============================================================
            # OPTIMIZACIÓN: Leer archivo UNA SOLA VEZ para los 14 timestamps
            # ============================================================
            fwd_df = pd.read_csv(fwd_path)
            if fwd_df.empty:
                continue
            fwd_df = td_rescale_strike_only(fwd_df)

            # Acumuladores para mediana sobre 14 timestamps
            pnl_pts_list = []
            pnl_pct_list = []

            # Variables para guardar el primer timestamp válido (columnas originales)
            first_valid_timestamp = None
            first_net_credit = None
            first_pnl_pts = None
            first_pnl_pct = None
            first_spx_chg = None
            first_dia_fwd = None
            first_hora_fwd = None
            first_leg_data = {}

            # ============================================================
            # LOOP: Evaluar estructura en los 14 timestamps fijos
            # ============================================================
            for ts_hhmm in intraday_timestamps:
                fwd_one_min, _ = get_slice_at_target_hhmm(fwd_df, ts_hhmm, nearest_minute_tol)
                if fwd_one_min.empty:
                    continue

                # Revalorizar batman en este timestamp
                bv_details = batman_value_from_df(
                    fwd_one_min,
                    str(row_dict["exp1"]), float(row_dict["k1"]),
                    str(row_dict["exp2"]), float(row_dict["k2"]),
                    float(row_dict["k3"]),
                    root1=base_root_exp1_norm,
                    root2=base_root_exp2_norm
                )

                if not bv_details or not np.isfinite(bv_details.get("net_credit", np.nan)):
                    continue

                net_credit_val = float(bv_details["net_credit"])

                # P&L en puntos
                pnl_pts = (net_credit_val - base_credit)
                pnl_pts_list.append(pnl_pts)

                # P&L en %
                denom_pts = abs(base_credit)
                if denom_pts and np.isfinite(denom_pts) and denom_pts > 0:
                    pct = (pnl_pts / denom_pts) * 100.0
                    pnl_pct_list.append(pct)

                # Guardar datos del primer timestamp válido (para columnas originales sin _mediana)
                if first_valid_timestamp is None:
                    first_valid_timestamp = ts_hhmm
                    first_net_credit = net_credit_val
                    first_pnl_pts = pnl_pts
                    first_pnl_pct = pct if denom_pts and np.isfinite(denom_pts) and denom_pts > 0 else None

                    # Timestamp forward
                    dt_us_fwd = datetime.strptime(f"{fwd_date_str_us} {ts_hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ_US_local)
                    dt_es_fwd = dt_us_fwd.astimezone(TZ_ES_local)
                    first_dia_fwd = dt_es_fwd.date().isoformat()
                    first_hora_fwd = dt_es_fwd.strftime("%H:%M")

                    # SPX change %
                    spx_fwd = underlying_from_slice(fwd_one_min)
                    if np.isfinite(spx_fwd) and np.isfinite(spot_base) and spot_base > 0:
                        first_spx_chg = round(float((spx_fwd / float(spot_base) - 1.0) * 100.0), 4)

                    # Metadata de las 3 patas
                    for leg_label, leg_info in (("k1", bv_details.get("leg1")),
                                                ("k2", bv_details.get("leg2")),
                                                ("k3", bv_details.get("leg3"))):
                        if not leg_info:
                            continue
                        bid_v, ask_v, mid_v = leg_info.get("bid"), leg_info.get("ask"), leg_info.get("mid")
                        root_v = normalize_root_value(leg_info.get("root"))

                        # Validación de mid
                        if leg_label == "k1":
                            expected_mid = row_dict.get("price_mid_short1")
                        elif leg_label == "k2":
                            expected_mid = row_dict.get("price_mid_long2")
                        else:  # k3
                            expected_mid = row_dict.get("price_mid_short3")
                        mid_status = compare_mid_values(mid_v, expected_mid)

                        # Validación de root
                        if leg_label in ("k1", "k3"):
                            base_root_expected = base_root_exp1_norm
                        else:  # k2
                            base_root_expected = base_root_exp2_norm
                        root_status = "CORRECTO" if (not base_root_expected or root_v==base_root_expected) else "INCORRECTO"

                        first_leg_data[leg_label] = {
                            'file': fwd_path.name,
                            'row': leg_info.get("row_index"),
                            'bid': round(float(bid_v),4) if bid_v is not None else None,
                            'ask': round(float(ask_v),4) if ask_v is not None else None,
                            'mid': round(float(mid_v),4) if mid_v is not None else None,
                            'root': root_v,
                            'check': mid_status,
                            'root_check': root_status
                        }

            # ============================================================
            # CALCULAR MEDIANAS (si tenemos datos)
            # ============================================================
            if pnl_pts_list:
                median_pnl_pts = round(float(np.median(pnl_pts_list)), 4)
                result['fwd_data'][f"PnL_fwd_pts_{suf}_mediana"] = median_pnl_pts

            if pnl_pct_list:
                median_pnl_pct = round(float(np.median(pnl_pct_list)), 2)
                result['fwd_data'][f"PnL_fwd_pct_{suf}_mediana"] = median_pnl_pct

            # ============================================================
            # GUARDAR COLUMNAS ORIGINALES (primer timestamp válido)
            # ============================================================
            if first_valid_timestamp is not None:
                result['fwd_data'][f"net_credit_fwd_{suf}"] = round(first_net_credit, 4)
                result['fwd_data'][f"PnL_fwd_pts_{suf}"] = round(first_pnl_pts, 4)
                result['fwd_data'][f"PnL_fwd_pct_{suf}"] = round(first_pnl_pct, 2) if first_pnl_pct is not None else None
                result['fwd_data'][f"dia_fwd_{suf}"] = first_dia_fwd
                result['fwd_data'][f"hora_fwd_{suf}"] = first_hora_fwd
                if first_spx_chg is not None:
                    result['fwd_data'][f"SPX_chg_pct_{suf}"] = first_spx_chg

                # Metadata de las 3 patas
                for leg_label in ("k1", "k2", "k3"):
                    if leg_label in first_leg_data:
                        ld = first_leg_data[leg_label]
                        result['fwd_data'][f"fwd_file_{leg_label}_{suf}"] = ld['file']
                        result['fwd_data'][f"fwd_row_{leg_label}_{suf}"] = ld['row']
                        result['fwd_data'][f"fwd_bid_{leg_label}_{suf}"] = ld['bid']
                        result['fwd_data'][f"fwd_ask_{leg_label}_{suf}"] = ld['ask']
                        result['fwd_data'][f"fwd_mid_{leg_label}_{suf}"] = ld['mid']
                        result['fwd_data'][f"fwd_root_{leg_label}_{suf}"] = ld['root']
                        result['fwd_data'][f"fwd_check_{leg_label}_{suf}"] = ld['check']
                        result['fwd_data'][f"fwd_root_check_{leg_label}_{suf}"] = ld['root_check']

                filled += 1
                result['errors'].append(('fwd_window_calculated_mediana', f'{suf} (14 timestamps)'))

        except Exception as e:
            result['errors'].append(('fwd_window_error', f'{suf}: {str(e)}'))

    result['filled'] = filled
    result['success'] = filled > 0
    return result

# ================== OPTIMIZACIÓN DE MEMORIA ==================
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Castea columnas numéricas a tipos más eficientes (float32, int32) para reducir memoria.
    """
    if df.empty:
        return df

    df_opt = df.copy()

    # Float64 → Float32
    float_cols = df_opt.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df_opt[col] = df_opt[col].astype('float32')

    # Int64 → Int32 (si cabe en rango)
    int_cols = df_opt.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df_opt[col].min() if not df_opt[col].isna().all() else 0
        col_max = df_opt[col].max() if not df_opt[col].isna().all() else 0
        # Rango int32: -2,147,483,648 a 2,147,483,647
        if col_min >= -2147483648 and col_max <= 2147483647:
            df_opt[col] = df_opt[col].astype('Int32')

    return df_opt


# ================== WORKER PARA PARALELIZACIÓN DE CANDIDATOS ==================
def _process_k1_candidate(args):
    """
    Worker para procesar un candidato k1 en paralelo.
    Retorna lista de candidatos Batman generados para este k1.
    """
    (k1, s2_list, s3_list, e1, e2, dte_map, spot, r_base, date_es_str, hhmm_es, _hhmm_us,
     base_idx, calls1_data, calls2_data, FRAC_SUFFIXES, USE_LIQUIDITY_FILTERS,
     MIN_OI_FRONT, MIN_VOL_FRONT, MIN_OI_BACK, MIN_VOL_BACK,
     PREFILTER_CREDIT_MIN, PREFILTER_CREDIT_MAX) = args

    # Reconstruir índices (necesario en proceso separado)
    calls1_idx = calls1_data.set_index("strike", drop=False)
    calls2_idx = calls2_data.set_index("strike", drop=False)

    rows = []

    # Validar liquidez k1
    if USE_LIQUIDITY_FILTERS and not leg_liquidity_ok_idx(calls1_idx, k1, MIN_OI_FRONT, MIN_VOL_FRONT):
        return rows

    mid1 = quick_mid_idx(calls1_idx, k1)
    if math.isnan(mid1):
        return rows

    # Loop k2
    for k2 in s2_list:
        if USE_LIQUIDITY_FILTERS and not leg_liquidity_ok_idx(calls2_idx, k2, MIN_OI_BACK, MIN_VOL_BACK):
            continue
        mid2 = quick_mid_idx(calls2_idx, k2)
        if math.isnan(mid2):
            continue

        # Loop k3
        for k3 in s3_list:
            if USE_LIQUIDITY_FILTERS and not leg_liquidity_ok_idx(calls1_idx, k3, MIN_OI_FRONT, MIN_VOL_FRONT):
                continue
            mid3 = quick_mid_idx(calls1_idx, k3)
            if math.isnan(mid3):
                continue

            # Prefiltro de crédito
            pre_net_credit = (-mid1) + (2*mid2) - mid3
            if not (PREFILTER_CREDIT_MIN <= pre_net_credit <= PREFILTER_CREDIT_MAX):
                continue

            # Validar orden de strikes
            if not (k1 < k2 < k3):
                continue

            # Generar URL
            url = make_url(e1, k1, e2, k2, k3)

            # Crear precache local para compute_strategy_metrics
            # (necesario porque compute_strategy_metrics usa precache global)
            precache_local = {
                e1: (calls1_data, None, calls1_idx, None),
                e2: (calls2_data, None, calls2_idx, None)
            }

            # Computar métricas
            met = compute_strategy_metrics(
                spot, r_base, e1, dte_map[e1], k1, e2, dte_map[e2], k2, k3, precache_local
            )

            # Metadata
            root_exp1 = root_for_exp(e1)
            root_exp2 = root_for_exp(e2)

            # Inicializar columnas FWD vacías (incluye medianas)
            fwd_cols = {}
            for suf in FRAC_SUFFIXES:
                fwd_cols[f"net_credit_fwd_{suf}"] = None
                fwd_cols[f"PnL_fwd_pts_{suf}"] = None
                fwd_cols[f"PnL_fwd_pct_{suf}"] = None
                fwd_cols[f"PnL_fwd_pts_{suf}_mediana"] = None
                fwd_cols[f"PnL_fwd_pct_{suf}_mediana"] = None
                fwd_cols[f"SPX_chg_pct_{suf}"] = None
                fwd_cols[f"dia_fwd_{suf}"] = None
                fwd_cols[f"hora_fwd_{suf}"] = None
                for leg in ("k1", "k2", "k3"):
                    fwd_cols[f"fwd_file_{leg}_{suf}"] = None
                    fwd_cols[f"fwd_row_{leg}_{suf}"] = None
                    fwd_cols[f"fwd_bid_{leg}_{suf}"] = None
                    fwd_cols[f"fwd_ask_{leg}_{suf}"] = None
                    fwd_cols[f"fwd_mid_{leg}_{suf}"] = None
                    fwd_cols[f"fwd_check_{leg}_{suf}"] = None
                    fwd_cols[f"fwd_root_{leg}_{suf}"] = None
                    fwd_cols[f"fwd_root_check_{leg}_{suf}"] = None

            rows.append({
                "url": url,
                "dia": date_es_str,
                "hora": hhmm_es,
                "SPX": spot,
                "r": r_base,
                "__base_idx": base_idx,
                "exp1": e1,
                "exp2": e2,
                "hora_us": _hhmm_us,
                "root_exp1": root_exp1,
                "root_exp2": root_exp2,
                **met,
                **fwd_cols
            })

    return rows


# ================== MAIN ==================
def main():
    files_sorted = list_local_files(DATA_DIR)
    if not files_sorted:
        print(f"[×] No se encontraron 30MIN*.csv en: {DATA_DIR}")
        return

    k = min(NUM_RANDOM_FILES, len(files_sorted))
    chosen_files = random.sample(files_sorted, k=k)

    ts_batch = datetime.now(TZ_ES).strftime("%Y%m%d_%H%M%S")
    batch_out_name = f"Batman_V18_LIVE_BETA_{ts_batch}.csv"
    batch_out_path = DESKTOP / safe_filename(batch_out_name)

    # Directorio temporal para Parquet incrementales
    temp_dir = DESKTOP / f"temp_batman_{ts_batch}"
    temp_dir.mkdir(exist_ok=True)
    parquet_files = []

    for pick_idx, chosen_file in enumerate(chosen_files, start=1):
        base_idx = files_sorted.index(chosen_file)
        print(f"[i] Archivo elegido al azar {pick_idx}/{k}: {chosen_file.name} (índice {base_idx} de {len(files_sorted)})")

        m = re.search(r"(\d{4}-\d{2}-\d{2})", chosen_file.name)
        if not m:
            print(f"[!] {chosen_file.name}: no pude extraer fecha, salto.")
            continue
        date_str_us = m.group(1)
        print(f"===== Archivo: {chosen_file.name} | Fecha US: {date_str_us} =====")

        df = pd.read_csv(chosen_file)
        if df.empty:
            print("[×] Archivo de opciones vacío."); continue
        df = td_rescale_strike_only(df)

        # Cadena canónica (día completo, sin descartar por spread en selección)
        df_canon = build_canonical_chain(df)
        if df_canon.empty:
            print("     [!] Cadena canónica vacía tras limpieza."); continue

        # Slicing por minuto
        for t_idx, target_hhmm_us in enumerate(TARGET_HHMMS_US if not IGNORE_TARGET_MINUTE else ["ALL"], start=1):
            if IGNORE_TARGET_MINUTE:
                one_min, used_hhmm_us = df.copy(), "ALL"
            else:
                one_min, used_hhmm_us = get_slice_at_target_hhmm(df, target_hhmm_us, NEAREST_MINUTE_TOLERANCE)
            status = "OK" if not one_min.empty else "NO-DATA"
            print(f"  - [{t_idx:02d}/{len(TARGET_HHMMS_US if not IGNORE_TARGET_MINUTE else ['ALL'])}] HH:MM US objetivo {target_hhmm_us} -> usado {used_hhmm_us} -> {status}")
            if one_min.empty:
                continue

            # Cache de greeks por snapshot
            global GREEKS_CACHE
            GREEKS_CACHE = {}

            # Hora ES + SPOT + r
            _hhmm_us = resolve_used_hhmm(used_hhmm_us, one_min)
            try:
                dt_us = datetime.strptime(f"{date_str_us} {_hhmm_us}", "%Y-%m-%d %H:%M").replace(tzinfo=TZ_US)
            except Exception:
                # fallback robusto
                dt_us = datetime.strptime(f"{date_str_us} 12:00", "%Y-%m-%d %H:%M").replace(tzinfo=TZ_US)
            dt_es = dt_us.astimezone(TZ_ES)
            hhmm_es = dt_es.strftime("%H:%M")
            date_es_str = dt_es.date().isoformat()

            spot = underlying_from_slice(one_min)
            r_base = fetch_risk_free_historical_fred(target_date=date_str_us, fallback=RISK_FREE_R)
            print(f"     SPOT @ {hhmm_es} ES: {spot:.2f} | r(DGS1MO @ {date_str_us})={r_base:.6f}")

            expirations, precache = build_chains(one_min)
            if not expirations:
                print("     [!] Sin expiraciones en este minuto."); continue
            dte_map = compute_dte_map(expirations, date_str_us)

            # Usar TODAS las expiraciones en RANGE_A y RANGE_B, ordenadas por DTE ascendente
            exp_A_all=[e for e in expirations if RANGE_A[0]<=dte_map[e]<=RANGE_A[1]]
            exp_B_all=[e for e in expirations if RANGE_B[0]<=dte_map[e]<=RANGE_B[1]]
            exp_A = sorted(exp_A_all, key=lambda e: dte_map[e])
            exp_B = sorted(exp_B_all, key=lambda e: dte_map[e])

            print(f"     Expiraciones únicas cargadas: {len(set(exp_A)|set(exp_B))}")
            print(f"     Combos brutos aprox.: {len(exp_A)}*{len(exp_B)}*|K1|*|K2|*|K3| (cómputo local)")

            base_p1 = round_to_nearest_10(spot*1.05)
            p1_raw = gen_k1_candidates(base_p1, rng=K1_RANGE, step=K1_STEP)

            rows=[]

            # ========== PARALELIZACIÓN DE SELECCIÓN DE CANDIDATOS ==========
            # Preparar todas las tareas de k1 para procesamiento paralelo
            tasks = []
            for e1 in exp_A:
                calls1, _, _, _ = precache[e1]
                strikes1 = sorted(set(calls1["strike"].tolist()))
                s1_list = [s for s in p1_raw if s in strikes1]

                p3_min, p3_max = spot*P3_MIN, spot*P3_MAX
                k3_all = [float(s) for s in strikes1 if p3_min<=s<=p3_max]
                s3_list = sample_strikes(k3_all, spot, K3_FACTORS)
                if not s1_list or not s3_list:
                    continue

                for e2 in exp_B:
                    calls2, _, _, _ = precache[e2]
                    strikes2 = sorted(set(calls2["strike"].tolist()))
                    p2_min, p2_max = spot*P2_MIN, spot*P2_MAX
                    k2_all = [float(s) for s in strikes2 if p2_min<=s<=p2_max]
                    s2_list = sample_strikes(k2_all, spot, K2_FACTORS)
                    if not s2_list:
                        continue

                    # Crear tareas para cada k1
                    for k1 in s1_list:
                        task = (
                            k1, s2_list, s3_list, e1, e2, dte_map, spot, r_base,
                            date_es_str, hhmm_es, _hhmm_us, base_idx,
                            calls1.reset_index(drop=True),  # DataFrame serializable
                            calls2.reset_index(drop=True),  # DataFrame serializable
                            FRAC_SUFFIXES, USE_LIQUIDITY_FILTERS,
                            MIN_OI_FRONT, MIN_VOL_FRONT, MIN_OI_BACK, MIN_VOL_BACK,
                            PREFILTER_CREDIT_MIN, PREFILTER_CREDIT_MAX
                        )
                        tasks.append(task)

            # Ejecutar en paralelo si hay tareas - STREAMING LAZY (sin list())
            if tasks:
                import os as os_module
                num_workers = os_module.cpu_count() or 1
                print(f"     [PARALLEL] Procesando {len(tasks)} tareas k1 con {num_workers} workers...")

                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Iteración lazy sobre resultados (no list())
                    for result in executor.map(_process_k1_candidate, tasks):
                        rows.extend(result)

                print(f"     [PARALLEL] Completado. {len(rows)} candidatos generados.")

            log_add("candidates_generated_file", len(rows))
            print(f"     Candidatos generados: {len(rows)}")

        # ============================================================
        # VOLCADO INCREMENTAL POR DÍA (stream to disk)
        # ============================================================
        if rows:
            df_day = pd.DataFrame(rows)

            # Optimizar dtypes antes de volcar
            df_day = optimize_dtypes(df_day)

            # Guardar Parquet incremental
            parquet_path = temp_dir / f"day_{pick_idx:03d}_{date_str_us.replace('-', '')}.parquet"
            df_day.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
            parquet_files.append(parquet_path)

            mem_mb = df_day.memory_usage(deep=True).sum() / 1024**2
            print(f"     [STREAM] Volcado incremental: {parquet_path.name} ({len(df_day)} filas, {mem_mb:.1f} MB)")

            # Liberar memoria
            del df_day
            del rows
        else:
            print(f"     [STREAM] Sin candidatos para este día, no se genera Parquet.")

    # ============================================================
    # CONSOLIDACIÓN: Leer todos los Parquet y combinar
    # ============================================================
    if not parquet_files:
        print("[×] No se generaron candidatos en ningún día.")
        audit_dump(DESKTOP, prefix="AUDIT_Batman_V18")
        return

    print(f"\n{'='*70}")
    print(f"CONSOLIDANDO {len(parquet_files)} archivos Parquet...")
    print(f"APLICANDO PREFILTROS durante la lectura para optimizar memoria...")
    print(f"{'='*70}")

    df_list = []
    total_filas_leidas = 0
    total_filas_filtradas = 0

    for pq_file in parquet_files:
        df_chunk = pd.read_parquet(pq_file)
        filas_antes = len(df_chunk)
        total_filas_leidas += filas_antes

        # Aplicar TODOS los prefiltros al chunk antes de agregarlo (evita consolidar basura)
        mask = pd.Series(True, index=df_chunk.index)

        # Prefiltro NET_CREDIT
        if "net_credit" in df_chunk.columns:
            mask &= (df_chunk["net_credit"] <= PREFILTER_CREDIT_MAX) & (df_chunk["net_credit"] >= PREFILTER_CREDIT_MIN)

        # Prefiltro DELTA_TOTAL
        if "delta_total" in df_chunk.columns:
            mask &= (df_chunk["delta_total"] >= DELTA_MIN) & (df_chunk["delta_total"] <= DELTA_MAX)

        # Prefiltro THETA_TOTAL
        if "theta_total" in df_chunk.columns:
            mask &= (df_chunk["theta_total"] >= THETA_MIN) & (df_chunk["theta_total"] <= THETA_MAX)

        # Prefiltro UEL_inf_USD
        if "UEL_inf_USD" in df_chunk.columns:
            mask &= (df_chunk["UEL_inf_USD"] >= UEL_INF_MIN) & (df_chunk["UEL_inf_USD"] <= UEL_INF_MAX)

        # Aplicar máscara combinada
        df_chunk = df_chunk[mask]
        filas_despues = len(df_chunk)
        total_filas_filtradas += filas_despues

        if not df_chunk.empty:
            df_list.append(df_chunk)

        print(f"  - Leído: {pq_file.name} ({filas_antes} → {filas_despues} filas tras prefiltros)")

    if not df_list:
        print("\n[!] ADVERTENCIA: No quedan filas después de aplicar prefiltros")
        return

    df = pd.concat(df_list, ignore_index=True)
    del df_list

    print(f"\n[✓] DataFrame consolidado: {total_filas_leidas} filas originales → {len(df)} filas tras prefiltros")
    print(f"[✓] Reducción de memoria: {100*(1-total_filas_filtradas/total_filas_leidas):.1f}% de filas eliminadas antes del concat")

    # Verificar que los filtros ya aplicados están OK (no deberían filtrar nada adicional)
    if not df.empty:
        # Verificar NET_CREDIT (ya filtrado)
        stats = get_filter_stats(df, "net_credit", PREFILTER_CREDIT_MIN, PREFILTER_CREDIT_MAX)
        print(f"\n[✓] NET_CREDIT ya prefiltrado: {format_filter_log('NET_CREDIT', PREFILTER_CREDIT_MIN, PREFILTER_CREDIT_MAX, len(df), len(df), stats)}")

        # Verificar DELTA_TOTAL (ya filtrado)
        stats = get_filter_stats(df, "delta_total", DELTA_MIN, DELTA_MAX)
        print(f"[✓] DELTA_TOTAL ya prefiltrado: {format_filter_log('DELTA_TOTAL', DELTA_MIN, DELTA_MAX, len(df), len(df), stats)}")

        # Verificar THETA_TOTAL (ya filtrado)
        stats = get_filter_stats(df, "theta_total", THETA_MIN, THETA_MAX)
        print(f"[✓] THETA_TOTAL ya prefiltrado: {format_filter_log('THETA_TOTAL', THETA_MIN, THETA_MAX, len(df), len(df), stats)}")

        # Verificar UEL_inf_USD (ya filtrado)
        stats = get_filter_stats(df, "UEL_inf_USD", UEL_INF_MIN, UEL_INF_MAX)
        print(f"[✓] UEL_inf_USD ya prefiltrado: {format_filter_log('UEL_inf_USD', UEL_INF_MIN, UEL_INF_MAX, len(df), len(df), stats)}")

        # Filtro PnLDV (condicional)
        if "PnLDV" in df.columns and not df.empty:
            if FILTER_PNLDV_ENABLED:
                filas_antes = len(df)
                stats = get_filter_stats(df, "PnLDV", PNLDV_MIN, PNLDV_MAX)
                df = df[
                    df["PnLDV"].notna() &
                    (df["PnLDV"] >= PNLDV_MIN) &
                    (df["PnLDV"] <= PNLDV_MAX)
                ]
                filas_despues = len(df)
                print(format_filter_log("PnLDV", PNLDV_MIN, PNLDV_MAX,
                                        filas_antes, filas_despues, stats))
            else:
                # Filtro desactivado, pero mostramos las estadísticas
                stats = get_filter_stats(df, "PnLDV", PNLDV_MIN, PNLDV_MAX)
                print(format_filter_log("PnLDV [DESACTIVADO]", PNLDV_MIN, PNLDV_MAX,
                                        len(df), len(df), stats))
        elif not df.empty:
            print(f"    [FILTER] PnLDV OMITIDO (columna no disponible) → filas {len(df)} (sin cambios)")

    # ---- BQI_ABS NORMALIZADO (indicador único con escala consistente) ----
    if not df.empty and {"PnLDV","EarL","EarR"}.issubset(df.columns):
        # Constantes para normalización
        EPS = 1e-6
        Wv  = 1.0          # Peso para profundidad del valle
        Wa  = 0.35         # Peso para asimetría
        OFFSET_BASE = 1000.0   # Base para valles positivos
        SCALE_FACTOR = 10.0    # Escala para PnLDV positivo

        earL  = pd.to_numeric(df["EarL"], errors="coerce")
        earR  = pd.to_numeric(df["EarR"], errors="coerce")
        pnldv = pd.to_numeric(df["PnLDV"], errors="coerce")

        # Componentes auxiliares
        EL = np.clip(earL, 0, None)
        ER = np.clip(earR, 0, None)
        EarScore = np.sqrt(EL * ER)
        ValleyDepth = np.clip(-pnldv, 0, None)
        Asym = np.abs(EL - ER)

        # Cálculo de BQI_ABS con normalización dimensional
        # Para PnLDV >= 0: mapeo lineal desde OFFSET_BASE
        # Para PnLDV < 0: ratio penalizado, capped a OFFSET_BASE - 1
        BQI_ABS_pos = OFFSET_BASE + (pnldv / SCALE_FACTOR)
        BQI_ABS_neg = EarScore / (EPS + Wv*ValleyDepth + Wa*Asym)
        
        # Cap negativo para garantizar separación
        BQI_ABS_neg = np.minimum(BQI_ABS_neg, OFFSET_BASE - 1.0)
        
        # Combinar según signo de PnLDV
        BQI_ABS = np.where(pnldv >= 0, BQI_ABS_pos, BQI_ABS_neg)

        # Conservar float (no redondear) y versión escalada ×1000
        df["BQI_ABS"] = pd.to_numeric(BQI_ABS, errors="coerce")
        df["BQR_1000"] = np.rint(df["BQI_ABS"] * 1000.0).astype("Int64")

        # Auxiliares para desempate/diagnóstico
        df["EarScore"] = EarScore
        df["Asym"]     = Asym

    # Filtro BQI_ABS (aplicado después de su cálculo, antes de FWD)
    if not df.empty and ("BQI_ABS" in df.columns):
        if FILTER_BQI_ABS_ENABLED:
            filas_antes = len(df)
            stats = get_filter_stats(df, "BQI_ABS", BQI_ABS_MIN, BQI_ABS_MAX)
            mask_bqi = df["BQI_ABS"].notna() & (df["BQI_ABS"] >= BQI_ABS_MIN) & (df["BQI_ABS"] <= BQI_ABS_MAX)
            df = df[mask_bqi]
            filas_despues = len(df)
            print(format_filter_log("BQI_ABS", BQI_ABS_MIN, BQI_ABS_MAX,
                                    filas_antes, filas_despues, stats))
        else:
            # Filtro desactivado, pero mostramos las estadísticas
            stats = get_filter_stats(df, "BQI_ABS", BQI_ABS_MIN, BQI_ABS_MAX)
            print(format_filter_log("BQI_ABS [DESACTIVADO]", BQI_ABS_MIN, BQI_ABS_MAX,
                                    len(df), len(df), stats))
    elif not df.empty:
        print(f"    [FILTER] BQI_ABS OMITIDO (columna no disponible) → filas {len(df)} (sin cambios)")

    # Orden interactivo con tie-breakers
    if df.empty:
        print("\nNo hay filas tras filtros; no se aplica orden. CSV vacío.")
    else:
        choice = "bqi"  # Elige entre: ["bqi", "bqr", "pnldv", "delta", "theta", "dv"]

        if choice in ("theta", "t", "θ", "theta_total"):
            df = df.sort_values(by=["theta_total","BQI_ABS","PnLDV"], ascending=[False,False,False])
            used = "theta_total → BQI_ABS → PnLDV"
        elif choice in ("delta", "d", "delta_total"):
            df = df.sort_values(by=["delta_total","BQI_ABS","PnLDV"], ascending=[False,False,False])
            used = "delta_total → BQI_ABS → PnLDV"
        elif choice in ("dv","death","valley"):
            df = df.sort_values(by=["Death valley","BQI_ABS","PnLDV"], ascending=[True,False,False])
            used = "Death valley → BQI_ABS → PnLDV"
        elif choice in ("pnldv","pnl","pnl_dv"):
            df = df.sort_values(by=["PnLDV","BQI_ABS","EarScore","Asym"], ascending=[False,False,False,True])
            used = "PnLDV → BQI_ABS → EarScore → Asym"
        elif choice in ("bqr","bqi1000"):
            df = df.sort_values(by=["BQR_1000","PnLDV","EarScore","Asym"], ascending=[False,False,False,True])
            used = "BQR_1000 → PnLDV → EarScore → Asym"
        else:
            df = df.sort_values(by=["BQI_ABS","PnLDV","EarScore","Asym"], ascending=[False,False,False,True])
            used = "BQI_ABS → PnLDV → EarScore → Asym"
        df = df.reset_index(drop=True)
        print(f"Orden aplicado: {used}")

    # ============================================================
    # SISTEMA FWD (FORWARD TESTING) — POST-RANKING
    # ============================================================
    fwd_indices_to_process = []

    if not df.empty and (FWD_ON_WINNERS or FWD_ON_LOSERS):
        print(f"\n{'='*70}")
        print("INICIANDO CÁLCULO FWD (Forward Testing)")
        print(f"{'='*70}")

        # Inicializar listas de índices
        winner_indices = []
        loser_indices = []

        # Parsear FWD_TOP_PCT
        pct_value = parse_percentage(FWD_TOP_PCT)
        if pct_value is None:
            print(f"\n[FWD] ⚠️ FWD_TOP_PCT no está definido. Abortando FWD.")
        else:
            print(f"\n[FWD] Modo de selección por PORCENTAJE activado: {pct_value*100:.1f}%")
            print(f"[FWD] Métrica de ranking: {RANKING_MODE} (descendente)")

            # Determinar qué batmans procesar
            if FWD_ON_WINNERS and FWD_ON_LOSERS:
                # Ambas activas: hacer FWD sobre winners Y losers
                df_valid = df[df[RANKING_MODE].notna()].copy()
                total_valid = len(df_valid)

                # Calcular N = max(1, floor(p × total_filas_validas))
                N = max(1, int(np.floor(pct_value * total_valid)))

                if N == 0 or total_valid == 0:
                    print(f"\n[FWD] ⚠️ No hay filas válidas o N=0. Abortando FWD.")
                else:
                    # Orden descendente: mejores son los primeros, peores son los últimos
                    # Winners: primeras N filas
                    # Losers: últimas N filas
                    df_winners_selected = df_valid.iloc[:N].copy()
                    df_losers_selected = df_valid.iloc[-N:].copy()

                    # Deduplicar por DTE1/DTE2 (keep="first" preserva orden)
                    num_winners_before = len(df_winners_selected)
                    df_winners_unique = df_winners_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                    winner_indices = list(df_winners_unique.index)

                    num_losers_before = len(df_losers_selected)
                    df_losers_unique = df_losers_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                    loser_indices = list(df_losers_unique.index)

                    print(f"\n[FWD] WINNERS - Selección por porcentaje posicional {pct_value*100:.1f}%:")
                    print(f"  - Métrica: {RANKING_MODE} (desc)")
                    print(f"  - Total filas válidas: {total_valid}")
                    print(f"  - N calculado: {N} (floor({{pct_value}} × {{total_valid}}))")
                    print(f"  - Batmans seleccionados (primeras {N} filas): {num_winners_before}")
                    print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(winner_indices)}")

                    print(f"\n[FWD] LOSERS - Selección por porcentaje posicional {pct_value*100:.1f}%:")
                    print(f"  - Métrica: {RANKING_MODE} (desc)")
                    print(f"  - Total filas válidas: {total_valid}")
                    print(f"  - N calculado: {N} (floor({{pct_value}} × {{total_valid}}))")
                    print(f"  - Batmans seleccionados (últimas {N} filas): {num_losers_before}")
                    print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(loser_indices)}")

                fwd_indices_to_process = winner_indices + loser_indices
                print(f"[FWD] Total batmans para FWD: {len(fwd_indices_to_process)} ({len(winner_indices)} winners + {len(loser_indices)} losers)")

            elif FWD_ON_WINNERS:
                # Solo winners
                df_valid = df[df[RANKING_MODE].notna()].copy()
                total_valid = len(df_valid)
                N = max(1, int(np.floor(pct_value * total_valid)))

                if N == 0 or total_valid == 0:
                    print(f"\n[FWD] ⚠️ No hay filas válidas o N=0. Abortando FWD.")
                else:
                    # Winners: primeras N filas (mejores valores altos)
                    df_winners_selected = df_valid.iloc[:N].copy()

                    # Deduplicar por DTE1/DTE2 (keep="first" preserva orden)
                    num_before = len(df_winners_selected)
                    df_winners_unique = df_winners_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                    winner_indices = list(df_winners_unique.index)

                    print(f"\n[FWD] WINNERS - Selección por porcentaje posicional {pct_value*100:.1f}%:")
                    print(f"  - Métrica: {RANKING_MODE} (desc)")
                    print(f"  - Total filas válidas: {total_valid}")
                    print(f"  - N calculado: {N} (floor({{pct_value}} × {{total_valid}}))")
                    print(f"  - Batmans seleccionados (primeras {N} filas): {num_before}")
                    print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(winner_indices)}")

                fwd_indices_to_process = winner_indices
                print(f"[FWD] Calculando forward para {len(fwd_indices_to_process)} batmans...")

            elif FWD_ON_LOSERS:
                # Solo losers
                df_valid = df[df[RANKING_MODE].notna()].copy()
                total_valid = len(df_valid)
                N = max(1, int(np.floor(pct_value * total_valid)))

                if N == 0 or total_valid == 0:
                    print(f"\n[FWD] ⚠️ No hay filas válidas o N=0. Abortando FWD.")
                else:
                    # Losers: últimas N filas (peores valores bajos)
                    df_losers_selected = df_valid.iloc[-N:].copy()

                    # Deduplicar por DTE1/DTE2 (keep="first" preserva orden)
                    num_before = len(df_losers_selected)
                    df_losers_unique = df_losers_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                    loser_indices = list(df_losers_unique.index)

                    print(f"\n[FWD] LOSERS - Selección por porcentaje posicional {pct_value*100:.1f}%:")
                    print(f"  - Métrica: {RANKING_MODE} (desc)")
                    print(f"  - Total filas válidas: {total_valid}")
                    print(f"  - N calculado: {N} (floor({{pct_value}} × {{total_valid}}))")
                    print(f"  - Batmans seleccionados (últimas {N} filas): {num_before}")
                    print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(loser_indices)}")

                fwd_indices_to_process = loser_indices
                print(f"[FWD] Calculando forward para {len(fwd_indices_to_process)} batmans...")

        # Eliminar duplicados manteniendo orden
        fwd_indices_to_process = list(dict.fromkeys(fwd_indices_to_process))

        print(f"Total a procesar: {len(fwd_indices_to_process)} batmans")
        print(f"Ventanas FWD: {FRAC_SUFFIXES} (% de DTE1)")
        print(f"{'='*70}\n")

        if fwd_indices_to_process:
            # Preparar argumentos para workers
            worker_args = []
            for idx in fwd_indices_to_process:
                row = df.loc[idx]
                row_dict = row.to_dict()

                args_tuple = (
                    idx,
                    row_dict,
                    files_sorted,
                    FORWARD_FRACS,
                    FRAC_SUFFIXES,
                    str(TZ_US),
                    str(TZ_ES),
                    NEAREST_MINUTE_TOLERANCE,
                    FWD_INTRADAY_TIMESTAMPS
                )
                worker_args.append(args_tuple)

            # Paralelización con ProcessPoolExecutor
            print(f"[FWD INTRADAY MEDIANA 14×30min] Lanzando {len(worker_args)} tareas en paralelo...")
            print(f"[FWD] Evaluando estructura en {len(FWD_INTRADAY_TIMESTAMPS)} timestamps por ventana: {FWD_INTRADAY_TIMESTAMPS}")
            completed_count = 0
            failed_count = 0

            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(_process_one_fwd_batman, arg): arg[0] for arg in worker_args}

                for future in futures:
                    idx = futures[future]
                    try:
                        result = future.result(timeout=60)

                        if result['success']:
                            # Actualizar DataFrame con resultados FWD
                            for col_name, col_value in result['fwd_data'].items():
                                df.at[idx, col_name] = col_value
                            completed_count += 1

                            if completed_count % 10 == 0:
                                print(f"[FWD] Progreso: {completed_count}/{len(worker_args)} completados...")
                        else:
                            failed_count += 1

                    except Exception as e:
                        print(f"[FWD] Error procesando índice {idx}: {e}")
                        failed_count += 1

            print(f"\n[FWD INTRADAY MEDIANA 14×30min] Cálculo completado:")
            print(f"  ✓ Exitosos: {completed_count}")
            print(f"  ✗ Fallidos: {failed_count}")
            print(f"  Columnas generadas: PnL_fwd_pts_*_mediana, PnL_fwd_pct_*_mediana")

            # ============================================================
            # GRÁFICOS DE PROMEDIOS FWD (Winners vs Losers)
            # ============================================================
            if FWD_PLOT_ENABLED and completed_count > 0:
                print(f"\n{'='*70}")
                print("GENERANDO GRÁFICOS DE PROMEDIOS FWD")
                print(f"{'='*70}")

                try:
                    # Preparar datos para gráficos usando índices calculados
                    winners_mask = df.index.isin(winner_indices) if FWD_ON_WINNERS else pd.Series([False]*len(df))
                    losers_mask = df.index.isin(loser_indices) if FWD_ON_LOSERS else pd.Series([False]*len(df))

                    # Recopilar datos para todas las ventanas
                    ventanas_x = [int(round(fr*100)) for fr in FORWARD_FRACS]  # % de DTE1

                    winners_means = []
                    losers_means = []
                    all_means = []

                    for suf in FRAC_SUFFIXES:
                        pnl_col = f"PnL_fwd_pct_{suf}"

                        if pnl_col not in df.columns:
                            winners_means.append(None)
                            losers_means.append(None)
                            all_means.append(None)
                            continue

                        # Obtener datos válidos
                        pnl_data = pd.to_numeric(df[pnl_col], errors='coerce')
                        valid_mask = pnl_data.notna()

                        if FWD_ON_WINNERS:
                            winners_pnl = pnl_data[winners_mask & valid_mask]
                            winners_means.append(winners_pnl.mean() if not winners_pnl.empty else None)
                        else:
                            winners_means.append(None)

                        if FWD_ON_LOSERS:
                            losers_pnl = pnl_data[losers_mask & valid_mask]
                            losers_means.append(losers_pnl.mean() if not losers_pnl.empty else None)
                        else:
                            losers_means.append(None)

                        if not FWD_ON_WINNERS and not FWD_ON_LOSERS:
                            all_pnl = pnl_data[valid_mask]
                            all_means.append(all_pnl.mean() if not all_pnl.empty else None)
                        else:
                            all_means.append(None)

                    # Crear gráfico de líneas
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Graficar líneas según configuración
                    if FWD_ON_WINNERS and any(m is not None for m in winners_means):
                        # Filtrar None values para la línea
                        x_winners = [x for x, m in zip(ventanas_x, winners_means) if m is not None]
                        y_winners = [m for m in winners_means if m is not None]
                        if x_winners:
                            ax.plot(x_winners, y_winners, marker='o', linewidth=2.5, markersize=8,
                                   color='#2ecc71', label=f'Winners (n={len(winner_indices)})', alpha=0.8)

                    if FWD_ON_LOSERS and any(m is not None for m in losers_means):
                        x_losers = [x for x, m in zip(ventanas_x, losers_means) if m is not None]
                        y_losers = [m for m in losers_means if m is not None]
                        if x_losers:
                            ax.plot(x_losers, y_losers, marker='s', linewidth=2.5, markersize=8,
                                   color='#e74c3c', label=f'Losers (n={len(loser_indices)})', alpha=0.8)

                    if not FWD_ON_WINNERS and not FWD_ON_LOSERS and any(m is not None for m in all_means):
                        x_all = [x for x, m in zip(ventanas_x, all_means) if m is not None]
                        y_all = [m for m in all_means if m is not None]
                        if x_all:
                            ax.plot(x_all, y_all, marker='o', linewidth=2.5, markersize=8,
                                   color='#3498db', label='All', alpha=0.8)

                    # Configuración del gráfico
                    ax.set_xlabel('Ventana Forward (% de DTE1)', fontsize=11, fontweight='bold')
                    ax.set_ylabel('P&L% promedio', fontsize=11, fontweight='bold')
                    ax.set_title('Promedio P&L Forward por Ventana', fontsize=13, fontweight='bold')
                    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(loc='best', fontsize=10, framealpha=0.9)

                    # Formato del eje X
                    ax.set_xticks(ventanas_x)
                    ax.set_xticklabels([f'{x}%' for x in ventanas_x])

                    plt.tight_layout()
                    plot_filename = f"FWD_Averages_{ts_batch}.png"
                    plot_path = DESKTOP / plot_filename
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Gráfico guardado: {plot_path}")

                except Exception as e:
                    print(f"✗ Error generando gráficos: {e}")

            print(f"{'='*70}\n")

    # Reordenación de columnas (mismo orden que CSV Copia)
    if not df.empty:
        preferred_order = [
            # Identificadores temporales
            "dia", "hora", "hora_us", "root_exp1", "root_exp2",
            # Métricas de calidad
            "net_credit_diff", "BQI_ABS", "DTE1/DTE2",
            # Strikes
            "k1", "k2", "k3",
            # Precios mid
            "price_mid_short1", "price_mid_long2", "price_mid_short3",
            # Greeks y valores de entrada
            "delta_total", "theta_total", "net_credit", "net_credit_mediana", "net_credit_mediana_n", "SPX",
            # Ventanas FWD - 01
            "dia_fwd_01", "hora_fwd_01", "PnL_fwd_pts_01", "PnL_fwd_pts_01_mediana",
            # Ventanas FWD - 05
            "dia_fwd_05", "hora_fwd_05", "PnL_fwd_pts_05", "PnL_fwd_pts_05_mediana",
            # Ventanas FWD - 15
            "dia_fwd_15", "hora_fwd_15", "PnL_fwd_pts_15", "PnL_fwd_pts_15_mediana",
            # Ventanas FWD - 25
            "dia_fwd_25", "hora_fwd_25", "PnL_fwd_pts_25", "PnL_fwd_pts_25_mediana",
            # Ventanas FWD - 50
            "dia_fwd_50", "hora_fwd_50", "PnL_fwd_pts_50", "PnL_fwd_pts_50_mediana",
        ]
        cols = [c for c in preferred_order if c in df.columns] + \
               [c for c in df.columns if c not in set(preferred_order)]
        df = df[cols]

    print(f"\nTotal enlaces tras filtro: {len(df)}")
    print(df.head(SHOW_MAX).to_string(index=False))

    # Exportación (numérico limpio para ordenar bien en CSV)
    num_cols = ["BQI_ABS","BQR_1000","PnLDV","EarL","EarR","Death valley","delta_total","theta_total","net_credit"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df.to_csv(batch_out_path, index=False, encoding="utf-8-sig", na_rep="")
    print(f"\nCSV guardado en: {batch_out_path} | Filas: {len(df)}")

    # ============================================================
    # PROCESO ADICIONAL: CSV COPIA CON net_credit_mediana (T+0) y net_credit_diff
    # ============================================================
    # PROPÓSITO: Calcular la mediana del net_credit en T+0 usando múltiples timestamps intradiarios
    #            para comparar con el net_credit original y detectar anomalías
    #
    # METODOLOGÍA:
    #   1. Para cada estructura Batman, evalúa net_credit en N timestamps del mismo día (T+0)
    #   2. Calcula la MEDIANA de estos net_credits (más robusta que la media ante outliers)
    #   3. Compara net_credit original vs net_credit_mediana → genera net_credit_diff (%)
    #   4. Filtra estructuras con net_credit_diff fuera de rango aceptable
    #
    # OPTIMIZACIÓN (VECTORIZADA):
    #   - Agrupa filas por día para cargar cada CSV UNA SOLA VEZ
    #   - Pre-carga todos los slices de timestamps necesarios en memoria
    #   - Reduce I/O de disco significativamente (factor 10x-100x más rápido)
    #   - Usa operaciones vectorizadas de NumPy/Pandas donde es posible
    # ============================================================
    print("\n" + "="*80)
    print("INICIANDO PROCESO ADICIONAL: Cálculo de net_credit_mediana (T+0) y net_credit_diff")
    print("="*80)

    # Crear copia del DataFrame
    df_copy = df.copy()

    # Filtrar solo filas con PnL_fwd_pct_01 no nulo
    col_filter = "PnL_fwd_pct_01_mediana" if "PnL_fwd_pct_01_mediana" in df_copy.columns else "PnL_fwd_pct_01"
    if col_filter not in df_copy.columns:
        print(f"[WARNING] Columna {col_filter} no encontrada. No se puede procesar CSV adicional.")
    else:
        df_copy = df_copy[df_copy[col_filter].notna()].copy()
        print(f"Filas tras filtro por {col_filter} no nulo: {len(df_copy)}")

        if len(df_copy) > 0:
            # Caché para DataFrames del día base
            _base_day_cache = {}

            def get_base_day_df_cached(dia_str):
                """
                Obtiene el DataFrame del día base desde caché o lo carga.
                """
                if not dia_str or dia_str in _base_day_cache:
                    return _base_day_cache.get(dia_str)

                try:
                    # Buscar archivo correspondiente a dia_str
                    matching_files = [f for f in files_sorted if dia_str in f.name]
                    if not matching_files:
                        print(f"  [WARNING] No se encontró archivo para fecha {dia_str}")
                        _base_day_cache[dia_str] = None
                        return None

                    base_path = matching_files[0]
                    base_df = pd.read_csv(base_path)
                    base_df = td_rescale_strike_only(base_df)
                    _base_day_cache[dia_str] = base_df
                    return base_df

                except Exception as e:
                    print(f"  [ERROR] Cargando archivo para {dia_str}: {e}")
                    _base_day_cache[dia_str] = None
                    return None

            # ============================================================
            # VERSIÓN VECTORIZADA: Cálculo de net_credit_mediana (T+0)
            # ============================================================
            # Agrupa filas por día para aprovechar caché de DataFrames y reducir I/O
            print(f"\nCalculando net_credit_mediana (T+0) para {len(df_copy)} filas... [VECTORIZADO]")

            # Agrupar por día para procesar en batch
            df_copy_with_idx = df_copy.reset_index(drop=False)
            grouped_by_day = df_copy_with_idx.groupby('dia', sort=False)

            # Inicializar arrays para resultados
            medianas = np.full(len(df_copy_with_idx), np.nan, dtype=float)
            n_valids = np.zeros(len(df_copy_with_idx), dtype=int)

            total_days = len(grouped_by_day)
            processed_rows = 0

            # Procesar cada día (todas sus filas juntas)
            for day_idx, (dia_str, day_group) in enumerate(grouped_by_day, 1):
                # Cargar DataFrame del día UNA SOLA VEZ
                base_df = get_base_day_df_cached(dia_str)

                if base_df is None or base_df.empty:
                    # Dejar NaN para todas las filas de este día
                    processed_rows += len(day_group)
                    print(f"  [{day_idx}/{total_days}] Día {dia_str}: {len(day_group)} filas - SIN DATOS")
                    continue

                # Pre-cargar TODOS los slices de timestamps necesarios para este día
                slice_cache = {}
                for ts_hhmm in FWD_INTRADAY_TIMESTAMPS:
                    slice_df, _ = get_slice_at_target_hhmm(base_df, ts_hhmm, NEAREST_MINUTE_TOLERANCE)
                    slice_cache[ts_hhmm] = slice_df

                # Procesar cada fila del día
                for row_idx, row in day_group.iterrows():
                    try:
                        # Extraer parámetros del Batman
                        exp1 = str(row.get("exp1", ""))
                        exp2 = str(row.get("exp2", ""))
                        k1 = float(row.get("k1", 0))
                        k2 = float(row.get("k2", 0))
                        k3 = float(row.get("k3", 0))
                        root_exp1_norm = normalize_root_value(row.get("root_exp1"))
                        root_exp2_norm = normalize_root_value(row.get("root_exp2"))

                        # Lista para acumular net_credit válidos
                        net_credit_list = []

                        # Evaluar en los timestamps intradía (usando caché)
                        for ts_hhmm in FWD_INTRADAY_TIMESTAMPS:
                            slice_df = slice_cache.get(ts_hhmm)
                            if slice_df is None or slice_df.empty:
                                continue

                            # Revalorizar Batman en este timestamp
                            bv_details = batman_value_from_df(
                                slice_df,
                                exp1, k1,
                                exp2, k2,
                                k3,
                                root1=root_exp1_norm,
                                root2=root_exp2_norm
                            )

                            if bv_details and np.isfinite(bv_details.get("net_credit", np.nan)):
                                net_credit_list.append(float(bv_details["net_credit"]))

                        # Calcular mediana si hay suficientes observaciones
                        n_valid = len(net_credit_list)
                        if n_valid >= 4:
                            mediana = float(np.median(net_credit_list))
                            medianas[row_idx] = round(mediana, 4)
                            n_valids[row_idx] = n_valid
                        else:
                            n_valids[row_idx] = n_valid

                    except Exception as e:
                        print(f"  [ERROR] Fila índice {row_idx}: {e}")
                        continue

                processed_rows += len(day_group)
                print(f"  [{day_idx}/{total_days}] Día {dia_str}: {len(day_group)} filas procesadas | Total: {processed_rows}/{len(df_copy_with_idx)}")

            # Insertar columnas net_credit_mediana y net_credit_mediana_n
            df_copy["net_credit_mediana"] = medianas
            df_copy["net_credit_mediana_n"] = n_valids

            # Calcular net_credit_diff (VECTORIZADO - sin apply)
            # Fórmula: 100 * (net_credit - net_credit_mediana) / abs(net_credit_mediana)
            nc = pd.to_numeric(df_copy["net_credit"], errors="coerce")
            nc_med = pd.to_numeric(df_copy["net_credit_mediana"], errors="coerce")

            # Calcular diferencia porcentual solo donde nc_med es válido y no cero
            mask_valid = nc_med.notna() & (nc_med != 0)
            df_copy["net_credit_diff"] = np.where(
                mask_valid,
                np.round(100.0 * (nc - nc_med) / np.abs(nc_med), 2),
                np.nan
            )

            # Asegurar tipos numéricos
            df_copy["net_credit"] = pd.to_numeric(df_copy["net_credit"], errors="coerce")
            df_copy["net_credit_mediana"] = pd.to_numeric(df_copy["net_credit_mediana"], errors="coerce")
            df_copy["net_credit_diff"] = pd.to_numeric(df_copy["net_credit_diff"], errors="coerce")
            df_copy["net_credit_mediana_n"] = pd.to_numeric(df_copy["net_credit_mediana_n"], errors="coerce")

            # Aplicar filtro de net_credit_diff si está habilitado
            if FILTER_NET_CREDIT_DIFF_ENABLED:
                filas_antes = len(df_copy)
                stats = get_filter_stats(df_copy, "net_credit_diff", NET_CREDIT_DIFF_MIN, NET_CREDIT_DIFF_MAX)
                df_copy = df_copy[
                    (df_copy["net_credit_diff"] >= NET_CREDIT_DIFF_MIN) &
                    (df_copy["net_credit_diff"] <= NET_CREDIT_DIFF_MAX)
                ].copy()
                filas_despues = len(df_copy)
                print(format_filter_log("NET_CREDIT_DIFF", NET_CREDIT_DIFF_MIN, NET_CREDIT_DIFF_MAX,
                                        filas_antes, filas_despues, stats, indent="                "))
            else:
                # Filtro desactivado, pero mostramos las estadísticas
                stats = get_filter_stats(df_copy, "net_credit_diff", NET_CREDIT_DIFF_MIN, NET_CREDIT_DIFF_MAX)
                print(format_filter_log("NET_CREDIT_DIFF [DESACTIVADO]", NET_CREDIT_DIFF_MIN, NET_CREDIT_DIFF_MAX,
                                        len(df_copy), len(df_copy), stats, indent="                "))

            # ============================================================
            # ANÁLISIS WINNERS/LOSERS (CSV COPIA FILTRADO)
            # ============================================================
            winner_indices_copy = []
            loser_indices_copy = []

            if len(df_copy) > 0 and (FWD_ON_WINNERS or FWD_ON_LOSERS):
                print(f"\n{'='*80}")
                print("ANÁLISIS WINNERS/LOSERS (CSV Copia Filtrado)")
                print(f"{'='*80}")

                # Parsear FWD_TOP_PCT
                pct_value = parse_percentage(FWD_TOP_PCT)
                if pct_value is None:
                    print(f"[WINNERS/LOSERS] ⚠️ FWD_TOP_PCT no está definido. Saltando análisis.")
                else:
                    print(f"[WINNERS/LOSERS] Porcentaje de selección: {pct_value*100:.1f}%")
                    print(f"[WINNERS/LOSERS] Métrica de ranking: {RANKING_MODE} (descendente)")

                    # Filtrar filas válidas según RANKING_MODE
                    df_copy_valid = df_copy[df_copy[RANKING_MODE].notna()].copy()

                    # Ordenar por RANKING_MODE descendente (mejores primero)
                    df_copy_valid = df_copy_valid.sort_values(by=RANKING_MODE, ascending=False)

                    total_valid = len(df_copy_valid)

                    if total_valid == 0:
                        print(f"[WINNERS/LOSERS] ⚠️ No hay filas válidas con {RANKING_MODE}. Saltando análisis.")
                    else:
                        # Calcular N = max(1, floor(p × total_filas_validas))
                        N = max(1, int(np.floor(pct_value * total_valid)))

                        print(f"[WINNERS/LOSERS] Total filas válidas: {total_valid}")
                        print(f"[WINNERS/LOSERS] N calculado: {N} (floor({pct_value} × {total_valid}))")

                        # Seleccionar winners y/o losers según configuración
                        if FWD_ON_WINNERS and FWD_ON_LOSERS:
                            # Ambos activados
                            df_winners_selected = df_copy_valid.iloc[:N].copy()
                            df_losers_selected = df_copy_valid.iloc[-N:].copy()

                            # Deduplicar por DTE1/DTE2
                            num_winners_before = len(df_winners_selected)
                            df_winners_unique = df_winners_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                            winner_indices_copy = list(df_winners_unique.index)

                            num_losers_before = len(df_losers_selected)
                            df_losers_unique = df_losers_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                            loser_indices_copy = list(df_losers_unique.index)

                            print(f"\n[WINNERS] Selección por porcentaje {pct_value*100:.1f}%:")
                            print(f"  - Batmans seleccionados (primeras {N} filas): {num_winners_before}")
                            print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(winner_indices_copy)}")

                            print(f"\n[LOSERS] Selección por porcentaje {pct_value*100:.1f}%:")
                            print(f"  - Batmans seleccionados (últimas {N} filas): {num_losers_before}")
                            print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(loser_indices_copy)}")

                            print(f"\n[TOTAL] Winners: {len(winner_indices_copy)} | Losers: {len(loser_indices_copy)}")

                        elif FWD_ON_WINNERS:
                            # Solo winners
                            df_winners_selected = df_copy_valid.iloc[:N].copy()
                            num_winners_before = len(df_winners_selected)
                            df_winners_unique = df_winners_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                            winner_indices_copy = list(df_winners_unique.index)

                            print(f"\n[WINNERS] Selección por porcentaje {pct_value*100:.1f}%:")
                            print(f"  - Batmans seleccionados (primeras {N} filas): {num_winners_before}")
                            print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(winner_indices_copy)}")

                        elif FWD_ON_LOSERS:
                            # Solo losers
                            df_losers_selected = df_copy_valid.iloc[-N:].copy()
                            num_losers_before = len(df_losers_selected)
                            df_losers_unique = df_losers_selected.drop_duplicates(subset=["DTE1/DTE2"], keep="first")
                            loser_indices_copy = list(df_losers_unique.index)

                            print(f"\n[LOSERS] Selección por porcentaje {pct_value*100:.1f}%:")
                            print(f"  - Batmans seleccionados (últimas {N} filas): {num_losers_before}")
                            print(f"  - Batmans tras deduplicación DTE1/DTE2: {len(loser_indices_copy)}")

                # Calcular estadísticas de promedios Winners vs Losers
                pnl_cols_mediana_sorted = sorted([col for col in df_copy.columns if col.startswith("PnL_fwd_pct_") and col.endswith("_mediana")])
                if len(pnl_cols_mediana_sorted) > 0 and (len(winner_indices_copy) > 0 or len(loser_indices_copy) > 0):
                    print("\n" + "-"*80)
                    print("ESTADÍSTICAS WINNERS vs LOSERS (columnas _mediana)")
                    print("-"*80)

                    for col in pnl_cols_mediana_sorted:
                        sufijo = col.replace("PnL_fwd_pct_", "").replace("_mediana", "")
                        pnl_data = pd.to_numeric(df_copy[col], errors='coerce')

                        if FWD_ON_WINNERS and len(winner_indices_copy) > 0:
                            winners_data = pnl_data[df_copy.index.isin(winner_indices_copy)]
                            winners_data_valid = winners_data.dropna()
                            if len(winners_data_valid) > 0:
                                print(f"\nVentana W{sufijo} - WINNERS (n={len(winners_data_valid)}):")
                                print(f"  • Promedio: {winners_data_valid.mean():.2f}%")
                                print(f"  • Mediana:  {winners_data_valid.median():.2f}%")
                                print(f"  • Std Dev:  {winners_data_valid.std():.2f}%")
                                print(f"  • Min/Max:  {winners_data_valid.min():.2f}% / {winners_data_valid.max():.2f}%")

                        if FWD_ON_LOSERS and len(loser_indices_copy) > 0:
                            losers_data = pnl_data[df_copy.index.isin(loser_indices_copy)]
                            losers_data_valid = losers_data.dropna()
                            if len(losers_data_valid) > 0:
                                print(f"\nVentana W{sufijo} - LOSERS (n={len(losers_data_valid)}):")
                                print(f"  • Promedio: {losers_data_valid.mean():.2f}%")
                                print(f"  • Mediana:  {losers_data_valid.median():.2f}%")
                                print(f"  • Std Dev:  {losers_data_valid.std():.2f}%")
                                print(f"  • Min/Max:  {losers_data_valid.min():.2f}% / {losers_data_valid.max():.2f}%")

                        if FWD_ON_WINNERS and FWD_ON_LOSERS and len(winner_indices_copy) > 0 and len(loser_indices_copy) > 0:
                            winners_data_valid = pnl_data[df_copy.index.isin(winner_indices_copy)].dropna()
                            losers_data_valid = pnl_data[df_copy.index.isin(loser_indices_copy)].dropna()
                            if len(winners_data_valid) > 0 and len(losers_data_valid) > 0:
                                diff_promedio = winners_data_valid.mean() - losers_data_valid.mean()
                                print(f"\n  → DIFERENCIA (Winners - Losers) Promedio: {diff_promedio:+.2f}%")

                print(f"{'='*80}\n")

            # ============================================================
            # ANÁLISIS DE CORRELACIONES PEARSON (CSV COPIA FILTRADO)
            # ============================================================
            if len(df_copy) > 0:
                print(f"\n{'='*80}")
                print("ANÁLISIS DE CORRELACIONES PEARSON (CSV Copia Filtrado)")
                print(f"{'='*80}")
                print(f"Filas disponibles: {len(df_copy)}")

                try:
                    # Métricas a correlacionar
                    metrics_to_correlate = [
                        'BQI_ABS', 'PnLDV', 'EarScore', 'Asym',
                        'delta_total', 'theta_total',
                        'net_credit', 'net_credit_mediana'
                    ]

                    # Verificar qué métricas están disponibles
                    available_metrics = [m for m in metrics_to_correlate if m in df_copy.columns]

                    # Ventanas de PnL_fwd_pct_*_mediana (W01, W05, W15, W25, W50)
                    pnl_mediana_cols = [
                        'PnL_fwd_pct_01_mediana',
                        'PnL_fwd_pct_05_mediana',
                        'PnL_fwd_pct_15_mediana',
                        'PnL_fwd_pct_25_mediana',
                        'PnL_fwd_pct_50_mediana'
                    ]

                    # Verificar qué columnas de PnL existen
                    available_pnl_cols = [col for col in pnl_mediana_cols if col in df_copy.columns]

                    if len(available_pnl_cols) == 0:
                        print("[WARNING] No se encontraron columnas PnL_fwd_pct_*_mediana")
                    elif len(available_metrics) == 0:
                        print("[WARNING] No se encontraron métricas para correlacionar")
                    else:
                        print(f"\nColumnas PnL disponibles: {len(available_pnl_cols)}")
                        print(f"Métricas disponibles: {len(available_metrics)}")
                        print("\nCorrelaciones Pearson entre PnL_fwd_pct_*_mediana y métricas:\n")

                        for pnl_col in available_pnl_cols:
                            # Extraer sufijo (01, 05, 15, 25, 50)
                            sufijo = pnl_col.replace('PnL_fwd_pct_', '').replace('_mediana', '')

                            # Convertir columna PnL a numérica
                            pnl_data = pd.to_numeric(df_copy[pnl_col], errors='coerce')
                            valid_pnl_mask = pnl_data.notna()

                            if valid_pnl_mask.sum() < 2:
                                print(f"Ventana W{sufijo}: Datos insuficientes (n={valid_pnl_mask.sum()})")
                                continue

                            print(f"Ventana W{sufijo} (n={valid_pnl_mask.sum()}):")

                            for metric in available_metrics:
                                # Convertir métrica a numérica
                                metric_data = pd.to_numeric(df_copy[metric], errors='coerce')

                                # Filtrar solo filas donde ambos valores son válidos
                                both_valid = valid_pnl_mask & metric_data.notna()

                                if both_valid.sum() >= 2:
                                    corr = pnl_data[both_valid].corr(metric_data[both_valid])
                                    print(f"  • {metric:22s}: {corr:+.4f}")

                            print()

                except Exception as e:
                    print(f"✗ Error calculando correlaciones: {e}")
                    import traceback
                    traceback.print_exc()

                print(f"{'='*80}\n")

            # ============================================================
            # REORDENAMIENTO DE COLUMNAS según especificación
            # ============================================================
            # Definir orden deseado
            columnas_ordenadas = [
                # Identificadores temporales
                "dia", "hora", "hora_us", "root_exp1", "root_exp2",
                # Métricas de calidad
                "net_credit_diff", "BQI_ABS", "DTE1/DTE2",
                # Strikes
                "k1", "k2", "k3",
                # Precios mid
                "price_mid_short1", "price_mid_long2", "price_mid_short3",
                # Greeks y valores de entrada
                "delta_total", "theta_total", "net_credit", "net_credit_mediana", "net_credit_mediana_n", "SPX",
                # Ventanas FWD - 01
                "dia_fwd_01", "hora_fwd_01", "PnL_fwd_pts_01", "PnL_fwd_pts_01_mediana",
                # Ventanas FWD - 05
                "dia_fwd_05", "hora_fwd_05", "PnL_fwd_pts_05", "PnL_fwd_pts_05_mediana",
                # Ventanas FWD - 15
                "dia_fwd_15", "hora_fwd_15", "PnL_fwd_pts_15", "PnL_fwd_pts_15_mediana",
                # Ventanas FWD - 25
                "dia_fwd_25", "hora_fwd_25", "PnL_fwd_pts_25", "PnL_fwd_pts_25_mediana",
                # Ventanas FWD - 50
                "dia_fwd_50", "hora_fwd_50", "PnL_fwd_pts_50", "PnL_fwd_pts_50_mediana",
            ]

            # Agregar columnas que existen pero no están en la lista (el resto...)
            columnas_existentes = set(df_copy.columns)
            columnas_ya_ordenadas = set(columnas_ordenadas)
            columnas_restantes = sorted(columnas_existentes - columnas_ya_ordenadas)

            # Construir lista final: ordenadas + resto
            columnas_finales = [c for c in columnas_ordenadas if c in columnas_existentes] + columnas_restantes

            # Aplicar reordenamiento
            df_copy = df_copy[columnas_finales]

            # Generar nombre para CSV copia
            batch_out_name_copy = batch_out_name.replace(".csv", "_T0_mediana.csv")
            batch_out_path_copy = DESKTOP / safe_filename(batch_out_name_copy)

            # Guardar CSV copia
            df_copy.to_csv(batch_out_path_copy, index=False, encoding="utf-8-sig", na_rep="")
            print(f"\n[CSV COPIA] Guardado en: {batch_out_path_copy}")
            print(f"[CSV COPIA] Filas: {len(df_copy)}")
            print(f"[CSV COPIA] Columnas nuevas: net_credit_mediana, net_credit_mediana_n, net_credit_diff")

            # Mostrar estadísticas
            valid_medianas = df_copy["net_credit_mediana"].notna().sum()
            print(f"[STATS] Filas con net_credit_mediana válida: {valid_medianas}/{len(df_copy)}")
            if valid_medianas > 0:
                avg_n = df_copy[df_copy["net_credit_mediana"].notna()]["net_credit_mediana_n"].mean()
                print(f"[STATS] Promedio de timestamps válidos por fila: {avg_n:.1f}")

            # ============================================================
            # GRÁFICO: Promedio de PnL_fwd_pct_*_mediana por periodo W
            # ============================================================
            print("\n" + "="*80)
            print("GENERANDO GRÁFICO: Promedio PnL_fwd_pct por periodo W")
            print("="*80)

            # Identificar columnas PnL_fwd_pct_*_mediana (5 periodos W)
            pnl_cols = [col for col in df_copy.columns if col.startswith("PnL_fwd_pct_") and col.endswith("_mediana")]
            pnl_cols_sorted = sorted(pnl_cols)  # Ordenar para mantener secuencia W1, W2, W3, W4, W5

            if len(pnl_cols_sorted) >= 1 and len(df_copy) > 0:
                # Calcular promedio para cada periodo W
                promedios = []
                labels_w = []

                for col in pnl_cols_sorted:
                    # Extraer sufijo (ej: "01", "05", "15", "25", "50")
                    sufijo = col.replace("PnL_fwd_pct_", "").replace("_mediana", "")
                    # Calcular promedio (ignorando NaN)
                    promedio = df_copy[col].mean()

                    if np.isfinite(promedio):
                        promedios.append(promedio)
                        labels_w.append(f"W{sufijo}")
                    else:
                        print(f"  [WARNING] Columna {col} no tiene datos válidos para promediar")

                # Generar gráfico si hay datos válidos
                if len(promedios) >= 1:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Gráfico lineal
                    ax.plot(range(len(promedios)), promedios, marker='o', linewidth=2, markersize=8, color='steelblue')

                    # Configuración del gráfico
                    ax.set_xlabel("Periodo W (Fracción del DTE1)", fontsize=12, fontweight='bold')
                    ax.set_ylabel("Promedio PnL_fwd_pct (%)", fontsize=12, fontweight='bold')
                    ax.set_title("Promedio de PnL_fwd_pct_mediana por Periodo W", fontsize=14, fontweight='bold')
                    ax.set_xticks(range(len(labels_w)))
                    ax.set_xticklabels(labels_w, fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

                    # Ajustar layout
                    plt.tight_layout()

                    # Guardar gráfico
                    plot_name = batch_out_name.replace(".csv", "_PnL_pct_W_promedio.png")
                    plot_path = DESKTOP / safe_filename(plot_name)
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"[GRÁFICO] Guardado en: {plot_path}")
                    print(f"[GRÁFICO] Periodos graficados: {len(promedios)}")
                    print(f"[GRÁFICO] Valores: {', '.join([f'{labels_w[i]}={promedios[i]:.2f}%' for i in range(len(promedios))])}")
                else:
                    print("[INFO] No hay suficientes datos válidos para generar el gráfico")
            else:
                print("[INFO] No se encontraron columnas PnL_fwd_pct_*_mediana para graficar")

            print("="*80 + "\n")

            # ============================================================
            # GRÁFICO: Promedio de PnL_fwd_pts_*_mediana por periodo W
            # ============================================================
            print("\n" + "="*80)
            print("GENERANDO GRÁFICO: Promedio PnL_fwd_pts por periodo W")
            print("="*80)

            # Identificar columnas PnL_fwd_pts_*_mediana (5 periodos W)
            pnl_pts_cols = [col for col in df_copy.columns if col.startswith("PnL_fwd_pts_") and col.endswith("_mediana")]
            pnl_pts_cols_sorted = sorted(pnl_pts_cols)  # Ordenar para mantener secuencia W1, W2, W3, W4, W5

            if len(pnl_pts_cols_sorted) >= 1 and len(df_copy) > 0:
                # Calcular promedio para cada periodo W
                promedios_pts = []
                labels_w_pts = []

                for col in pnl_pts_cols_sorted:
                    # Extraer sufijo (ej: "01", "05", "15", "25", "50")
                    sufijo = col.replace("PnL_fwd_pts_", "").replace("_mediana", "")
                    # Calcular promedio (ignorando NaN)
                    promedio = df_copy[col].mean()

                    if np.isfinite(promedio):
                        promedios_pts.append(promedio)
                        labels_w_pts.append(f"W{sufijo}")
                    else:
                        print(f"  [WARNING] Columna {col} no tiene datos válidos para promediar")

                # Generar gráfico si hay datos válidos
                if len(promedios_pts) >= 1:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Gráfico lineal
                    ax.plot(range(len(promedios_pts)), promedios_pts, marker='o', linewidth=2, markersize=8, color='darkorange')

                    # Configuración del gráfico
                    ax.set_xlabel("Periodo W (Fracción del DTE1)", fontsize=12, fontweight='bold')
                    ax.set_ylabel("Promedio PnL_fwd_pts (puntos)", fontsize=12, fontweight='bold')
                    ax.set_title("Promedio de PnL_fwd_pts_mediana por Periodo W", fontsize=14, fontweight='bold')
                    ax.set_xticks(range(len(labels_w_pts)))
                    ax.set_xticklabels(labels_w_pts, fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

                    # Ajustar layout
                    plt.tight_layout()

                    # Guardar gráfico
                    plot_name_pts = batch_out_name.replace(".csv", "_PnL_pts_W_promedio.png")
                    plot_path_pts = DESKTOP / safe_filename(plot_name_pts)
                    plt.savefig(plot_path_pts, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"[GRÁFICO] Guardado en: {plot_path_pts}")
                    print(f"[GRÁFICO] Periodos graficados: {len(promedios_pts)}")
                    print(f"[GRÁFICO] Valores: {', '.join([f'{labels_w_pts[i]}={promedios_pts[i]:.2f} pts' for i in range(len(promedios_pts))])}")
                else:
                    print("[INFO] No hay suficientes datos válidos para generar el gráfico")
            else:
                print("[INFO] No se encontraron columnas PnL_fwd_pts_*_mediana para graficar")

            print("="*80 + "\n")

            # ============================================================
            # GRÁFICOS COMPARATIVOS: WINNERS vs LOSERS (usando columnas _mediana)
            # ============================================================
            if (FWD_ON_WINNERS or FWD_ON_LOSERS) and (len(winner_indices_copy) > 0 or len(loser_indices_copy) > 0):
                print("\n" + "="*80)
                print("GENERANDO GRÁFICOS COMPARATIVOS: WINNERS vs LOSERS")
                print("="*80)

                # Identificar columnas PnL_fwd_pct_*_mediana
                pnl_cols_mediana = [col for col in df_copy.columns if col.startswith("PnL_fwd_pct_") and col.endswith("_mediana")]
                pnl_cols_mediana_sorted = sorted(pnl_cols_mediana)

                if len(pnl_cols_mediana_sorted) >= 1:
                    # Crear máscaras para winners y losers
                    winners_mask = df_copy.index.isin(winner_indices_copy) if FWD_ON_WINNERS else pd.Series([False]*len(df_copy), index=df_copy.index)
                    losers_mask = df_copy.index.isin(loser_indices_copy) if FWD_ON_LOSERS else pd.Series([False]*len(df_copy), index=df_copy.index)

                    # Calcular promedios para cada ventana
                    ventanas_x = []
                    winners_means = []
                    losers_means = []

                    for col in pnl_cols_mediana_sorted:
                        # Extraer sufijo (01, 05, 15, 25, 50)
                        sufijo = col.replace("PnL_fwd_pct_", "").replace("_mediana", "")
                        ventanas_x.append(f"W{sufijo}")

                        # Convertir a numérico
                        pnl_data = pd.to_numeric(df_copy[col], errors='coerce')
                        valid_mask = pnl_data.notna()

                        # Calcular promedio para winners
                        if FWD_ON_WINNERS:
                            winners_pnl = pnl_data[winners_mask & valid_mask]
                            winners_means.append(winners_pnl.mean() if not winners_pnl.empty else None)
                        else:
                            winners_means.append(None)

                        # Calcular promedio para losers
                        if FWD_ON_LOSERS:
                            losers_pnl = pnl_data[losers_mask & valid_mask]
                            losers_means.append(losers_pnl.mean() if not losers_pnl.empty else None)
                        else:
                            losers_means.append(None)

                    # Generar gráfico comparativo
                    fig, ax = plt.subplots(figsize=(12, 7))

                    # Plot winners (verde)
                    if FWD_ON_WINNERS and any(m is not None for m in winners_means):
                        x_winners = [i for i, m in enumerate(winners_means) if m is not None]
                        y_winners = [m for m in winners_means if m is not None]
                        if x_winners:
                            ax.plot(x_winners, y_winners, marker='o', linewidth=2.5, markersize=8,
                                   color='#2ecc71', label=f'Winners (n={len(winner_indices_copy)})', alpha=0.9)

                    # Plot losers (rojo)
                    if FWD_ON_LOSERS and any(m is not None for m in losers_means):
                        x_losers = [i for i, m in enumerate(losers_means) if m is not None]
                        y_losers = [m for m in losers_means if m is not None]
                        if x_losers:
                            ax.plot(x_losers, y_losers, marker='s', linewidth=2.5, markersize=8,
                                   color='#e74c3c', label=f'Losers (n={len(loser_indices_copy)})', alpha=0.9)

                    # Configuración del gráfico
                    ax.set_xlabel("Periodo W (Fracción del DTE1)", fontsize=12, fontweight='bold')
                    ax.set_ylabel("Promedio PnL_fwd_pct_mediana (%)", fontsize=12, fontweight='bold')
                    ax.set_title(f"Comparativa Winners vs Losers - PnL_fwd_pct_mediana\n(Ranking: {RANKING_MODE}, Top {pct_value*100:.1f}%)",
                                fontsize=14, fontweight='bold')
                    ax.set_xticks(range(len(ventanas_x)))
                    ax.set_xticklabels(ventanas_x, fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                    ax.legend(fontsize=11, loc='best')

                    # Ajustar layout
                    plt.tight_layout()

                    # Guardar gráfico
                    plot_name_wl = batch_out_name.replace(".csv", "_Winners_vs_Losers_pct.png")
                    plot_path_wl = DESKTOP / safe_filename(plot_name_wl)
                    plt.savefig(plot_path_wl, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"[GRÁFICO W/L] Guardado en: {plot_path_wl}")
                    print(f"[GRÁFICO W/L] Periodos: {len(ventanas_x)}")
                    if FWD_ON_WINNERS:
                        print(f"[GRÁFICO W/L] Winners (n={len(winner_indices_copy)}): {', '.join([f'{ventanas_x[i]}={winners_means[i]:.2f}%' if winners_means[i] is not None else f'{ventanas_x[i]}=N/A' for i in range(len(ventanas_x))])}")
                    if FWD_ON_LOSERS:
                        print(f"[GRÁFICO W/L] Losers (n={len(loser_indices_copy)}): {', '.join([f'{ventanas_x[i]}={losers_means[i]:.2f}%' if losers_means[i] is not None else f'{ventanas_x[i]}=N/A' for i in range(len(ventanas_x))])}")

                    # ============================================================
                    # GRÁFICO COMPARATIVO: WINNERS vs LOSERS (PnL_fwd_pts_mediana)
                    # ============================================================
                    print("\n" + "-"*80)

                    # Identificar columnas PnL_fwd_pts_*_mediana
                    pnl_pts_cols_mediana = [col for col in df_copy.columns if col.startswith("PnL_fwd_pts_") and col.endswith("_mediana")]
                    pnl_pts_cols_mediana_sorted = sorted(pnl_pts_cols_mediana)

                    if len(pnl_pts_cols_mediana_sorted) >= 1:
                        # Calcular promedios para cada ventana
                        ventanas_pts_x = []
                        winners_pts_means = []
                        losers_pts_means = []

                        for col in pnl_pts_cols_mediana_sorted:
                            # Extraer sufijo (01, 05, 15, 25, 50)
                            sufijo = col.replace("PnL_fwd_pts_", "").replace("_mediana", "")
                            ventanas_pts_x.append(f"W{sufijo}")

                            # Convertir a numérico
                            pnl_pts_data = pd.to_numeric(df_copy[col], errors='coerce')
                            valid_pts_mask = pnl_pts_data.notna()

                            # Calcular promedio para winners
                            if FWD_ON_WINNERS:
                                winners_pts_pnl = pnl_pts_data[winners_mask & valid_pts_mask]
                                winners_pts_means.append(winners_pts_pnl.mean() if not winners_pts_pnl.empty else None)
                            else:
                                winners_pts_means.append(None)

                            # Calcular promedio para losers
                            if FWD_ON_LOSERS:
                                losers_pts_pnl = pnl_pts_data[losers_mask & valid_pts_mask]
                                losers_pts_means.append(losers_pts_pnl.mean() if not losers_pts_pnl.empty else None)
                            else:
                                losers_pts_means.append(None)

                        # Generar gráfico comparativo
                        fig, ax = plt.subplots(figsize=(12, 7))

                        # Plot winners (verde)
                        if FWD_ON_WINNERS and any(m is not None for m in winners_pts_means):
                            x_winners_pts = [i for i, m in enumerate(winners_pts_means) if m is not None]
                            y_winners_pts = [m for m in winners_pts_means if m is not None]
                            if x_winners_pts:
                                ax.plot(x_winners_pts, y_winners_pts, marker='o', linewidth=2.5, markersize=8,
                                       color='#2ecc71', label=f'Winners (n={len(winner_indices_copy)})', alpha=0.9)

                        # Plot losers (rojo)
                        if FWD_ON_LOSERS and any(m is not None for m in losers_pts_means):
                            x_losers_pts = [i for i, m in enumerate(losers_pts_means) if m is not None]
                            y_losers_pts = [m for m in losers_pts_means if m is not None]
                            if x_losers_pts:
                                ax.plot(x_losers_pts, y_losers_pts, marker='s', linewidth=2.5, markersize=8,
                                       color='#e74c3c', label=f'Losers (n={len(loser_indices_copy)})', alpha=0.9)

                        # Configuración del gráfico
                        ax.set_xlabel("Periodo W (Fracción del DTE1)", fontsize=12, fontweight='bold')
                        ax.set_ylabel("Promedio PnL_fwd_pts_mediana (puntos)", fontsize=12, fontweight='bold')
                        ax.set_title(f"Comparativa Winners vs Losers - PnL_fwd_pts_mediana\n(Ranking: {RANKING_MODE}, Top {pct_value*100:.1f}%)",
                                    fontsize=14, fontweight='bold')
                        ax.set_xticks(range(len(ventanas_pts_x)))
                        ax.set_xticklabels(ventanas_pts_x, fontsize=10)
                        ax.grid(True, alpha=0.3, linestyle='--')
                        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                        ax.legend(fontsize=11, loc='best')

                        # Ajustar layout
                        plt.tight_layout()

                        # Guardar gráfico
                        plot_name_wl_pts = batch_out_name.replace(".csv", "_Winners_vs_Losers_pts.png")
                        plot_path_wl_pts = DESKTOP / safe_filename(plot_name_wl_pts)
                        plt.savefig(plot_path_wl_pts, dpi=150, bbox_inches='tight')
                        plt.close()

                        print(f"[GRÁFICO W/L] Guardado en: {plot_path_wl_pts}")
                        print(f"[GRÁFICO W/L] Periodos: {len(ventanas_pts_x)}")
                        if FWD_ON_WINNERS:
                            print(f"[GRÁFICO W/L] Winners (n={len(winner_indices_copy)}): {', '.join([f'{ventanas_pts_x[i]}={winners_pts_means[i]:.2f}pts' if winners_pts_means[i] is not None else f'{ventanas_pts_x[i]}=N/A' for i in range(len(ventanas_pts_x))])}")
                        if FWD_ON_LOSERS:
                            print(f"[GRÁFICO W/L] Losers (n={len(loser_indices_copy)}): {', '.join([f'{ventanas_pts_x[i]}={losers_pts_means[i]:.2f}pts' if losers_pts_means[i] is not None else f'{ventanas_pts_x[i]}=N/A' for i in range(len(ventanas_pts_x))])}")

                    print("="*80 + "\n")
                else:
                    print("[INFO] No se encontraron columnas PnL_fwd_pct_*_mediana para gráficos W/L")
                    print("="*80 + "\n")

            print("="*80 + "\n")
        else:
            print("[INFO] No hay filas válidas tras el filtro. No se genera CSV copia.")

    print("="*80)
    print("FIN PROCESO ADICIONAL")
    print("="*80 + "\n")

    # Limpieza de archivos temporales Parquet
    print(f"\n[CLEANUP] Eliminando {len(parquet_files)} archivos temporales Parquet...")
    for pq_file in parquet_files:
        try:
            pq_file.unlink()
        except Exception as e:
            print(f"  - Error eliminando {pq_file.name}: {e}")
    try:
        temp_dir.rmdir()
        print(f"[CLEANUP] Directorio temporal eliminado: {temp_dir.name}")
    except Exception as e:
        print(f"[CLEANUP] No se pudo eliminar directorio temporal: {e}")

    # Dump de auditoría
    audit_dump(DESKTOP, prefix="AUDIT_Batman_V18")

if __name__=="__main__":
    main()