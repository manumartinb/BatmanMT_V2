# Análisis W50: Glosario Aplicado a Ventana 50 Días
## Estudio Especial: Correlación BQI_ABS > 10 con PnL_50

**Archivo analizado:** `combined_BATMAN_mediana_w_stats_w_vix.csv`
**Target:** `PnL_fwd_pts_50_mediana`
**Fecha:** 2025-12-28

---

## Resumen Ejecutivo

### Hallazgo Principal: BQI_ABS > 10

| Métrica | Valor |
|---------|-------|
| **N total** | 2,565 trades |
| **N con BQI_ABS > 10** | 173 trades (6.7%) |
| **PnL_50 mediana (BQI > 10)** | **+23.90 pts** |
| **PnL_50 mediana (BQI ≤ 10)** | +11.18 pts |
| **Diferencia de medianas** | **+12.73 pts** |
| **Win Rate (BQI > 10)** | **91.91%** |
| **Win Rate (BQI ≤ 10)** | 63.96% |
| **p-value (Mann-Whitney)** | **0.000004** ✅ |
| **Cohen's d** | 0.41 (efecto medio) |
| **Cliff's delta** | 0.28 |
| **IC 95% Bootstrap** | [8.82, 16.91] ✅ |

### ✅ CONCLUSIÓN PRINCIPAL
**BQI_ABS > 10 es un predictor estadísticamente significativo de mejor PnL a 50 días.** Los trades con BQI_ABS > 10 tienen una mediana de PnL +12.73 pts superior al resto, con un win rate del 91.91% vs 63.96%.

---

## 1. Información del Dataset

- **Filas totales:** 2,609
- **Filas válidas (sin NaN en target):** 2,565
- **Columnas totales:** 271
- **Columnas excluidas (anti-leakage):** 182
- **Columnas numéricas analizadas:** 86

### Estadísticas del Target (PnL_fwd_pts_50_mediana)

| Estadística | Valor |
|-------------|-------|
| Media | 15.32 pts |
| Mediana | 12.58 pts |
| Std | 31.62 pts |
| Min | -55.58 pts |
| Max | 207.00 pts |
| % Positivos | 65.5% |

---

## 2. Top 20 Variables Correlacionadas con PnL_50 (Global)

| # | Variable | Spearman ρ | Pearson r | Sig |
|---|----------|------------|-----------|-----|
| 1 | SPX_minus_SMA100 | -0.1625 | -0.2345 | *** |
| 2 | SPX_ZScore50 | -0.1594 | -0.1960 | *** |
| 3 | **BQI_V2_ABS** | **+0.1333** | +0.1974 | *** |
| 4 | SPX_MACD_Line | -0.1292 | -0.2118 | *** |
| 5 | SPX_HV50 | +0.1267 | +0.1735 | *** |
| 6 | SPX_minus_SMA200 | -0.1262 | -0.1746 | *** |
| 7 | FF_BAT | +0.1255 | +0.0454 | *** |
| 8 | SPX_minus_SMA50 | -0.1249 | -0.2212 | *** |
| 9 | r | +0.1240 | +0.1197 | *** |
| 10 | SPX_ROC20 | -0.1210 | -0.1730 | *** |
| 11 | SPX_MACD_Signal | -0.1203 | -0.1975 | *** |
| 12 | SPX_minus_SMA20 | -0.1164 | -0.1760 | *** |
| 13 | SPX_RSI14 | -0.1125 | -0.1320 | *** |
| 14 | SPX_BB_Pct | -0.1089 | -0.1183 | *** |
| 15 | SPX_ZScore20 | -0.1089 | -0.1183 | *** |
| 16 | BQR_1000 | +0.1070 | +0.0657 | *** |
| 17 | **BQI_ABS** | **+0.1070** | +0.0657 | *** |
| 18 | PnLDV | +0.1063 | +0.0509 | *** |
| 19 | EarL | +0.1060 | +0.1195 | *** |
| 20 | SPX_Stoch_D | -0.1019 | -0.1064 | *** |

**Significancia:** *** p < 0.001, ** p < 0.01, * p < 0.05

---

## 3. Análisis Detallado de BQI_ABS

### 3.1 Distribución de BQI_ABS

| Percentil | Valor |
|-----------|-------|
| P1 | 0.56 |
| P5 | 0.65 |
| P10 | 0.73 |
| P25 | 0.92 |
| P50 (mediana) | 1.24 |
| P75 | 1.86 |
| P90 | 4.10 |
| **P95** | **20.59** |
| P99 | 1003.18 |

> **Nota:** BQI_ABS > 10 corresponde aproximadamente al top 7% de la distribución.

### 3.2 Comparación BQI > 10 vs BQI ≤ 10

| Métrica | BQI_ABS > 10 | BQI_ABS ≤ 10 | Diferencia |
|---------|--------------|--------------|------------|
| **N** | 173 | 2,392 | - |
| **Media PnL_50** | 25.54 | 14.58 | +10.96 |
| **Mediana PnL_50** | **23.90** | 11.18 | **+12.72** |
| **Std PnL_50** | 19.71 | 32.19 | -12.48 |
| **Min PnL_50** | -29.70 | -55.58 | - |
| **Max PnL_50** | 78.25 | 207.00 | - |
| **Win Rate** | **91.91%** | 63.96% | **+27.95%** |

### 3.3 Tests Estadísticos

| Test | Estadístico | p-value | Interpretación |
|------|-------------|---------|----------------|
| T-test | t = 4.42 | 0.000010 | ✅ Significativo |
| Mann-Whitney U | U = 265,784 | **0.000004** | ✅ Muy significativo |
| Cohen's d | 0.41 | - | Efecto medio |
| Cliff's delta | 0.28 | - | Efecto pequeño-medio |

### 3.4 IC 95% Bootstrap (1000 iteraciones)

- **Diferencia observada:** +12.73 pts
- **IC 95%:** [8.82, 16.91] pts
- **Media bootstrap:** 12.62 pts
- **Std bootstrap:** 2.12 pts

✅ **El intervalo de confianza NO incluye el cero**, confirmando la significancia.

---

## 4. Análisis por Deciles de BQI_ABS

| Decil | N | Rango BQI | Media PnL_50 | Mediana PnL_50 | Win Rate |
|-------|---|-----------|--------------|----------------|----------|
| 0 | 257 | 0.48 - 0.72 | 11.73 | 4.58 | 55.25% |
| 1 | 256 | 0.73 - 0.85 | 14.16 | 9.98 | 61.33% |
| 2 | 257 | 0.85 - 0.97 | 11.43 | 7.05 | 59.92% |
| 3 | 256 | 0.97 - 1.09 | 14.37 | 10.89 | 59.38% |
| 4 | 257 | 1.09 - 1.24 | 16.43 | 12.25 | 68.48% |
| 5 | 256 | 1.24 - 1.43 | 15.56 | 13.16 | 65.23% |
| 6 | 256 | 1.43 - 1.68 | 17.58 | 13.83 | 67.19% |
| 7 | 257 | 1.68 - 2.10 | 16.38 | 13.73 | 65.76% |
| 8 | 256 | 2.10 - 4.08 | 14.61 | 14.28 | 70.31% |
| **9** | **257** | **4.11 - 1009.50** | **20.90** | **20.10** | **85.60%** |

> **Observación:** El decil 9 (BQI_ABS > 4.11) muestra un salto significativo en win rate (85.60%) y mediana de PnL.

---

## 5. Análisis por Múltiples Umbrales de BQI_ABS

| Umbral | N (arriba) | % Total | Media PnL | Mediana PnL | Win Rate | Δ Mediana | p-value |
|--------|------------|---------|-----------|-------------|----------|-----------|---------|
| > 1 | 1,732 | 67.5% | 16.66 | 14.51 | 69.2% | +7.49 | 0.000009 |
| > 2 | 561 | 21.9% | 17.87 | 17.25 | 77.0% | +6.99 | 0.000043 |
| > 5 | 229 | 8.9% | 21.87 | 20.98 | 86.9% | +9.80 | 0.000000 |
| **> 10** | **173** | **6.7%** | **25.54** | **23.90** | **91.9%** | **+12.73** | **0.000000** |
| > 20 | 135 | 5.3% | 27.22 | 25.95 | 91.9% | +14.36 | 0.000000 |
| > 50 | 85 | 3.3% | 26.37 | 25.40 | 91.8% | +13.51 | 0.000003 |
| > 100 | 84 | 3.3% | 26.38 | 25.35 | 91.7% | +13.45 | 0.000004 |
| > 200 | 84 | 3.3% | 26.38 | 25.35 | 91.7% | +13.45 | 0.000004 |

> **Hallazgo:** El umbral BQI > 10 parece ser el punto óptimo de balance entre tamaño muestral (173) y efecto (91.9% win rate).

---

## 6. Correlación DENTRO del Grupo BQI_ABS > 10

**Pregunta:** Una vez filtrado por BQI > 10, ¿importa qué tan alto sea el BQI?

| Métrica | Valor | p-value |
|---------|-------|---------|
| Pearson r | 0.0433 | 0.571 |
| Spearman ρ | 0.1438 | 0.059 |

❌ **Conclusión:** Dentro del grupo BQI > 10, NO hay correlación significativa adicional. Es decir, una vez que BQI > 10, no importa si es 15 o 500.

### Top 15 Variables Correlacionadas con PnL_50 (Solo BQI > 10)

| # | Variable | Spearman ρ | Sig |
|---|----------|------------|-----|
| 1 | **EarR** | **+0.4290** | *** |
| 2 | **EarScore** | **+0.4009** | *** |
| 3 | **FF_ATM** | **+0.3851** | *** |
| 4 | BQI_V2_ABS | +0.3779 | *** |
| 5 | SPX_DailyChange | +0.2885 | *** |
| 6 | EarL | +0.2838 | *** |
| 7 | RATIO_UEL_EARS | +0.2712 | *** |
| 8 | net_credit_mediana | -0.2677 | *** |
| 9 | net_credit | -0.2631 | *** |
| 10 | UEL_inf_USD | +0.2627 | *** |
| 11 | price_ask_short1 | +0.2458 | ** |
| 12 | price_mid_short1 | +0.2441 | ** |
| 13 | price_bid_short1 | +0.2438 | ** |
| 14 | FF_BAT | +0.2361 | ** |
| 15 | SPX_MACD_Line | -0.2350 | ** |

> **Hallazgo clave:** Dentro del grupo BQI > 10, las variables más predictivas son **EarR** (ρ=0.43), **EarScore** (ρ=0.40) y **FF_ATM** (ρ=0.39).

---

## 7. Características Distintivas del Grupo BQI > 10

| Variable | Media (>10) | Media (≤10) | Mediana (>10) | Mediana (≤10) | p-value |
|----------|-------------|-------------|---------------|---------------|---------|
| **VIX_Close** | **39.93** | 22.66 | **37.13** | 21.38 | 0.000000 |
| **FF_ATM** | **0.315** | 0.128 | **0.252** | 0.101 | 0.000000 |
| theta_total | +0.222 | -0.172 | +0.186 | -0.174 | 0.000000 |
| delta_total | 0.040 | 0.077 | 0.040 | 0.087 | 0.000000 |
| iv_k1 | 0.307 | 0.201 | 0.277 | 0.190 | 0.000000 |
| SPX | 3,746 | 4,530 | 3,295 | 4,361 | 0.000000 |
| net_credit_mediana | -14.18 | -21.48 | -10.40 | -21.00 | 0.000000 |
| EarL | 88.23 | 56.13 | 83.54 | 55.51 | 0.000000 |
| EarR | 94.86 | 135.03 | 86.99 | 133.66 | 0.000000 |

> **Perfil típico de BQI > 10:**
> - VIX alto (~37 vs ~21)
> - FF_ATM más alto
> - Theta positivo
> - SPX más bajo (~3,300 vs ~4,400)
> - Mejor net_credit

---

## 8. Análisis Temporal (Por Año)

| Año | N Total | N BQI>10 | % BQI>10 | Med PnL (>10) | Med PnL (≤10) | Δ |
|-----|---------|----------|----------|---------------|---------------|---|
| 2019 | 119 | 3 | 2.5% | 19.95 | 15.08 | +4.88 |
| **2020** | **548** | **102** | **18.6%** | **28.59** | 17.45 | **+11.14** |
| 2021 | 366 | 1 | 0.3% | -5.35 | -1.28 | -4.08 |
| 2022 | 451 | 16 | 3.6% | 7.05 | 1.73 | +5.33 |
| 2023 | 215 | 3 | 1.4% | 16.45 | -2.65 | +19.10 |
| 2024 | 353 | 18 | 5.1% | 25.13 | 13.60 | +11.53 |
| 2025 | 513 | 30 | 5.9% | 29.00 | 24.80 | +4.20 |

> **Observación:** 2020 tuvo la mayor concentración de trades con BQI > 10 (18.6%), coincidiendo con alta volatilidad (COVID).

---

## 9. Interacción con VIX

| VIX Cuartil | N (BQI>10) | Med PnL (>10) | Win Rate (>10) | N (BQI≤10) | Med PnL (≤10) |
|-------------|------------|---------------|----------------|------------|---------------|
| Q1 (Bajo) | 3 | 14.78 | 100.0% | 641 | 9.53 |
| Q2 | 11 | 24.75 | 100.0% | 628 | 13.43 |
| Q3 | 21 | 23.90 | 100.0% | 620 | 11.48 |
| **Q4 (Alto)** | **138** | **24.55** | **89.9%** | 503 | 8.25 |

> **Hallazgo:** La mayoría de los trades con BQI > 10 (138/173 = 80%) ocurren cuando el VIX está alto (Q4). El win rate es ~90% en VIX alto y 100% en VIX bajo (aunque con muy pocas muestras).

---

## 10. DTE Típico del Grupo BQI > 10

| Estadística | DTE1 | DTE2 |
|-------------|------|------|
| Media | 76.77 | 134.46 |
| Mediana | 77.00 | 140.00 |
| Min | 70.00 | 100.00 |
| Max | 90.00 | 160.00 |

---

## 11. Reglas Accionables Encontradas (p < 0.05)

| Regla | N | Mediana PnL | Win Rate | Δ vs resto | p-value |
|-------|---|-------------|----------|------------|---------|
| **BQI_ABS > 10** | **173** | **+23.90** | **91.9%** | **+12.73** | **0.000004** |
| SPX_minus_SMA100 ≤ 117 | 1,284 | +16.35 | 70.9% | +8.05 | 0.000000 |
| SPX_ZScore50 ≤ 0.50 | 1,283 | +14.53 | 68.7% | +4.75 | 0.000006 |
| BQI_V2_ABS > 47.29 | 1,282 | +16.14 | 67.2% | +6.54 | 0.000029 |
| SPX_HV50 > 16.45 | 1,281 | +15.65 | 70.7% | +6.90 | 0.000000 |
| SPX_minus_SMA200 ≤ 216 | 1,283 | +16.28 | 70.1% | +7.68 | 0.000000 |
| FF_BAT > 0.60 | 1,149 | +16.95 | 69.1% | +8.13 | 0.000000 |

---

## 12. Reglas Combinadas (BQI > 10 + otras condiciones)

| Regla Combinada | N | Mediana PnL | Win Rate |
|-----------------|---|-------------|----------|
| BQI > 10 + VIX Alto | 159 | +24.45 | 91.2% |
| BQI > 10 + VIX Bajo | 14 | +19.38 | 100.0% |
| BQI > 10 + FF_ATM Alto | 170 | +24.46 | 91.8% |

---

## 13. Análisis del VIX

| Métrica | Valor | p-value |
|---------|-------|---------|
| Pearson r (VIX vs PnL_50) | +0.1178 | 0.000000 |
| Spearman ρ (VIX vs PnL_50) | +0.0335 | 0.089 (NS) |

| VIX Cuartil | N | Media PnL | Mediana PnL | Std |
|-------------|---|-----------|-------------|-----|
| Q1 (Bajo) | 644 | 14.26 | 9.58 | 26.77 |
| Q2 | 639 | 15.18 | 13.70 | 29.50 |
| Q3 | 641 | 13.00 | 12.38 | 31.34 |
| Q4 (Alto) | 641 | 18.83 | 14.65 | 37.64 |

---

## 14. Conclusiones y Recomendaciones

### ✅ Hallazgos Principales

1. **BQI_ABS > 10 es un predictor robusto:**
   - Win rate: 91.9% vs 64.0%
   - Mediana PnL: +23.9 vs +11.2 (+12.7 pts)
   - Estadísticamente significativo (p < 0.00001)
   - IC 95% Bootstrap: [8.8, 16.9] - no cruza cero

2. **Perfil del trade BQI > 10:**
   - VIX alto (mediana ~37)
   - FF_ATM elevado
   - Theta positivo
   - Ocurre principalmente en mercados volátiles (80% en Q4 de VIX)

3. **Una vez BQI > 10, el valor exacto no importa:**
   - No hay correlación significativa adicional dentro del grupo
   - La regla es binaria: >10 o ≤10

4. **Variables más predictivas dentro de BQI > 10:**
   - EarR (ρ = 0.43)
   - EarScore (ρ = 0.40)
   - FF_ATM (ρ = 0.39)

### 📋 Recomendaciones

1. **Filtro primario:** Considerar BQI_ABS > 10 como criterio de selección de trades con alta probabilidad de éxito (91.9% win rate).

2. **Limitación:** La muestra es pequeña (173 trades, 6.7% del total), concentrada en periodos de alta volatilidad.

3. **Refinamiento opcional:** Dentro del grupo BQI > 10, priorizar trades con EarR y FF_ATM altos.

4. **Precaución:** La mayoría de los casos BQI > 10 ocurren en VIX alto. El edge puede estar relacionado con el régimen de volatilidad más que con BQI per se.

---

*Análisis generado automáticamente siguiendo el glosario y metodología del documento "Glosario_y_Prompt_W50 y W90.docx"*
