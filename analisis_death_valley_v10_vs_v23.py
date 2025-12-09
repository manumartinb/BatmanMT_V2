#!/usr/bin/env python3
"""
Análisis comparativo del cálculo de Death Valley entre Batman V10 y V23

PROBLEMA IDENTIFICADO:
- V10 usa un método analítico simplificado que asume que el Death Valley está en un punto específico S0
- V23 usa un método numérico de búsqueda en grilla para encontrar el mínimo real

Este script compara ambos métodos y determina cuál es más preciso.
"""

import math
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

# ==================== BLACK-SCHOLES ====================
def normal_cdf(x):
    return norm.cdf(x)

def bs_call_price(S, K, T, r, sigma, q=0.0):
    """Precio de una call europea usando Black-Scholes"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return S*math.exp(-q*T)*normal_cdf(d1) - K*math.exp(-r*T)*normal_cdf(d2)

# ==================== MÉTODOS DE CÁLCULO ====================

def death_valley_v10(k1, k2, k3, r2, sigma2, tau, net_credit):
    """
    Método V10: Analítico simplificado
    Asume que el Death Valley está en S0 = K2 * exp(-(r + 0.5*σ²)*τ)
    """
    S0 = float(k2) * math.exp(-(r2 + 0.5*sigma2*sigma2) * tau)
    k_lo, k_hi = (min(k1, k3), max(k1, k3))

    if (S0 >= k_lo) and (S0 < k_hi):
        val_short1 = -max(0.0, S0 - float(k1))
        val_short3 = -max(0.0, S0 - float(k3))
        val_long2 = 2.0 * bs_call_price(S0, float(k2), tau, r2, sigma2)
        value_t1 = val_short1 + val_short3 + val_long2
        pnl_dv_points = value_t1 - net_credit
        death_valley = S0
    else:
        value_lim = (float(k1) + float(k3)) - 2.0*float(k2)*math.exp(-r2*tau)
        pnl_dv_points = value_lim - net_credit
        death_valley = float('nan')

    return death_valley, pnl_dv_points, S0

def death_valley_v23(k1, k2, k3, r2, sigma2, tau, net_credit, S_PNL):
    """
    Método V23: Búsqueda numérica en grilla
    Busca el mínimo del valor del Batman en una grilla de 300 puntos
    """
    def batman_value_at_S(S):
        """Valor teórico en T1 (shorts intrínseco, long valorada a T2)."""
        val_short1 = -max(0.0, S - float(k1))
        val_short3 = -max(0.0, S - float(k3))
        val_long2 = 2.0 * bs_call_price(S, float(k2), tau, r2, sigma2)
        return val_short1 + val_short3 + val_long2

    k_lo, k_hi = (min(k1, k3), max(k1, k3))
    spot_ref = float(S_PNL)
    lower = min(k_lo, spot_ref) * 0.6
    upper = max(k_hi, spot_ref) * 1.4
    S_grid = np.linspace(lower, upper, 300)
    vals = np.array([batman_value_at_S(s) for s in S_grid])
    idx = int(np.argmin(vals))
    death_valley = float(S_grid[idx])
    pnl_dv_points = float(vals[idx] - net_credit)

    return death_valley, pnl_dv_points, batman_value_at_S

def death_valley_optimizado(k1, k2, k3, r2, sigma2, tau, net_credit):
    """
    Método OPTIMIZADO: Usa scipy.optimize para encontrar el mínimo exacto
    Este es el método más preciso y debería coincidir con OptionStrat
    """
    def batman_value_at_S(S):
        """Valor teórico en T1 (shorts intrínseco, long valorada a T2)."""
        val_short1 = -max(0.0, S - float(k1))
        val_short3 = -max(0.0, S - float(k3))
        val_long2 = 2.0 * bs_call_price(S, float(k2), tau, r2, sigma2)
        return val_short1 + val_short3 + val_long2

    k_lo, k_hi = (min(k1, k3), max(k1, k3))

    # Buscar el mínimo usando optimización numérica
    # Expandir el rango de búsqueda para estar seguros
    result = minimize_scalar(
        batman_value_at_S,
        bounds=(k_lo * 0.5, k_hi * 1.5),
        method='bounded'
    )

    death_valley = result.x
    min_value = result.fun
    pnl_dv_points = min_value - net_credit

    return death_valley, pnl_dv_points, batman_value_at_S

# ==================== ANÁLISIS Y COMPARACIÓN ====================

def analizar_caso(k1, k2, k3, r2, sigma2, tau, net_credit, S_PNL, nombre_caso):
    """Compara los tres métodos para un caso específico"""

    print(f"\n{'='*80}")
    print(f"CASO: {nombre_caso}")
    print(f"{'='*80}")
    print(f"Parámetros:")
    print(f"  K1={k1}, K2={k2}, K3={k3}")
    print(f"  r={r2:.4f}, σ={sigma2:.4f}, τ={tau:.4f} años")
    print(f"  net_credit={net_credit:.2f} puntos")
    print(f"  S_PNL (spot referencia)={S_PNL:.2f}")
    print()

    # V10: Método analítico
    dv_v10, pnl_v10, s0_v10 = death_valley_v10(k1, k2, k3, r2, sigma2, tau, net_credit)
    print(f"V10 (Analítico):")
    print(f"  S0 calculado = {s0_v10:.2f}")
    dv_str = f"{dv_v10:.2f}" if not math.isnan(dv_v10) else "NaN"
    print(f"  Death Valley = {dv_str}")
    print(f"  PnLDV = {pnl_v10:.2f}")
    print()

    # V23: Búsqueda en grilla
    dv_v23, pnl_v23, func_v23 = death_valley_v23(k1, k2, k3, r2, sigma2, tau, net_credit, S_PNL)
    print(f"V23 (Grilla 300 pts):")
    print(f"  Death Valley = {dv_v23:.2f}")
    print(f"  PnLDV = {pnl_v23:.2f}")
    print()

    # Optimizado: Minimización numérica precisa
    dv_opt, pnl_opt, func_opt = death_valley_optimizado(k1, k2, k3, r2, sigma2, tau, net_credit)
    print(f"OPTIMIZADO (scipy.optimize):")
    print(f"  Death Valley = {dv_opt:.2f}")
    print(f"  PnLDV = {pnl_opt:.2f}")
    print()

    # Comparación
    print(f"DIFERENCIAS vs OPTIMIZADO (Referencia):")
    if not math.isnan(dv_v10):
        diff_dv_v10 = abs(dv_v10 - dv_opt)
        diff_pnl_v10 = abs(pnl_v10 - pnl_opt)
        print(f"  V10: ΔDV = {diff_dv_v10:.2f} pts, ΔPnL = {diff_pnl_v10:.2f} pts")
    else:
        print(f"  V10: Death Valley fuera de rango [k1, k3]")

    diff_dv_v23 = abs(dv_v23 - dv_opt)
    diff_pnl_v23 = abs(pnl_v23 - pnl_opt)
    print(f"  V23: ΔDV = {diff_dv_v23:.2f} pts, ΔPnL = {diff_pnl_v23:.2f} pts")
    print()

    # Validación: verificar que V23 encontró un mínimo en la grilla
    if diff_pnl_v23 > 0.5:  # Tolerancia de 0.5 puntos
        print(f"⚠️  PROBLEMA EN V23: La grilla no capturó el mínimo exacto")
        print(f"    Error: {diff_pnl_v23:.2f} puntos SPX = ${diff_pnl_v23*100:.2f} USD")

    # Verificar si V10 es preciso
    if not math.isnan(dv_v10) and diff_pnl_v10 > 1.0:
        print(f"⚠️  PROBLEMA EN V10: El método analítico no es preciso")
        print(f"    Error: {diff_pnl_v10:.2f} puntos SPX = ${diff_pnl_v10*100:.2f} USD")

    return {
        'v10': (dv_v10, pnl_v10),
        'v23': (dv_v23, pnl_v23),
        'opt': (dv_opt, pnl_opt)
    }

# ==================== CASOS DE PRUEBA ====================

if __name__ == "__main__":
    print("="*80)
    print("ANÁLISIS COMPARATIVO: Death Valley V10 vs V23")
    print("="*80)

    # Caso 1: Batman típico con DTE largo (>250)
    analizar_caso(
        k1=5600, k2=5900, k3=6200,
        r2=0.045, sigma2=0.18, tau=250/365,
        net_credit=-15.5,
        S_PNL=5900,
        nombre_caso="Batman típico 250 DTE (ATM)"
    )

    # Caso 2: Batman OTM (spot por debajo)
    analizar_caso(
        k1=5800, k2=6100, k3=6400,
        r2=0.045, sigma2=0.20, tau=300/365,
        net_credit=-18.2,
        S_PNL=5700,
        nombre_caso="Batman OTM 300 DTE (spot bajo)"
    )

    # Caso 3: Batman ITM (spot por encima)
    analizar_caso(
        k1=5400, k2=5700, k3=6000,
        r2=0.045, sigma2=0.16, tau=280/365,
        net_credit=-12.3,
        S_PNL=6100,
        nombre_caso="Batman ITM 280 DTE (spot alto)"
    )

    # Caso 4: Batman con alta volatilidad
    analizar_caso(
        k1=5500, k2=5850, k3=6200,
        r2=0.045, sigma2=0.35, tau=270/365,
        net_credit=-25.8,
        S_PNL=5850,
        nombre_caso="Batman alta volatilidad (σ=35%)"
    )

    # Caso 5: Batman con DTE corto
    analizar_caso(
        k1=5700, k2=5950, k3=6200,
        r2=0.045, sigma2=0.18, tau=60/365,
        net_credit=-8.5,
        S_PNL=5950,
        nombre_caso="Batman DTE corto (60 días)"
    )

    print("\n" + "="*80)
    print("CONCLUSIONES")
    print("="*80)
    print("""
1. V10 usa un método analítico simplificado que asume el Death Valley en S0
   - Fórmula: S0 = K2 * exp(-(r + 0.5*σ²)*τ)
   - Problema: Solo evalúa UN punto, puede no ser el mínimo real
   - Ventaja: Rápido y simple

2. V23 usa búsqueda numérica en grilla de 300 puntos
   - Problema: La grilla puede no capturar el mínimo exacto
   - Problema: El rango de búsqueda depende de S_PNL que puede ser inapropiado
   - Ventaja: Más flexible que V10

3. MÉTODO OPTIMIZADO (scipy.optimize)
   - Encuentra el mínimo exacto usando optimización numérica
   - Este debería coincidir con OptionStrat (que probablemente usa algo similar)
   - Recomendación: USAR ESTE MÉTODO EN V23

RECOMENDACIÓN:
Reemplazar el método de grilla en V23 por scipy.optimize.minimize_scalar
para encontrar el Death Valley exacto y coincidir con OptionStrat.
    """)
