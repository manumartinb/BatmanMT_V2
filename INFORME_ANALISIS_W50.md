# Informe de Análisis Predictivo - Ventana 50

**Target:** `PnL_fwd_pts_50_mediana`
**Dataset:** `combined_BATMAN_mediana_w_stats_w_vix.csv`
**N analizado:** 2,565 observaciones

---

## (a) Resumen Ejecutivo

### Hallazgos Accionables

1. **Distancia a SMA100 es el predictor más fuerte:** Cuando el SPX está muy por debajo de su SMA100 (percentil 10), la mediana del PnL es ~29 pts vs ~11 pts cuando está cerca o por encima. Correlación Spearman = -0.163.

2. **Volatilidad histórica alta favorece el trade:** SPX_HV50 > 28.21 (P85) muestra mediana de PnL de 34.05 vs 9.62. Diferencia = +24.42 pts.

3. **BQI_V2_ABS como indicador de calidad:** Valores altos (>57, P90) correlacionan con mejores resultados (mediana 32.05 vs 11.18).

4. **Momentum negativo favorece:** MACD_Line negativo (P10) genera mejor PnL mediana (28.76 vs 10.72).

5. **RSI extremos:** RSI14 bajo (<30) tiende a producir mejores resultados.

6. **VIX tiene señal débil:** Correlación Spearman = 0.034 (no significativa). Sin embargo, VIX alto (>P80) muestra mediana PnL ligeramente superior (14.85 vs 9.10 en VIX bajo).

7. **Theta_k1 extremo:** Valores muy negativos de theta_k1 (<P10) muestran PnL mediana de 29.50 vs 10.28.

8. **Capacidad predictiva OOS es débil:** Spearman OOS = 0.071 ± 0.148. Hay señal pero con alta variabilidad entre folds.

9. **FF_BAT como filtro:** Q5 vs Q1 muestra diferencia de 13.72 pts en mediana.

10. **EarL y EarScore como indicadores:** Valores altos correlacionan con mejor PnL.

11. **⚠️ GOLDEN CROSS NO FAVORECE - DEATH CROSS ES MEJOR:** Contraintuitivamente, cuando SMA50 < SMA200 (Death Cross), la mediana del PnL es 16.27 vs 11.64 en Golden Cross. Diferencia = -4.64 pts. **p=0.0007 (significativo)**.

12. **Ratio Theta K1/K2 es predictor fuerte:** Correlación Spearman r=-0.122. Ratios más negativos (theta_k1 más negativo respecto a theta_k2) → MEJOR PnL. Decil más bajo: mediana 18.65 vs 4.45 en decil más alto.

13. **SPX por debajo de SMAs es favorable:** Todos los filtros de cruce muestran que estar POR DEBAJO de las SMAs mejora el PnL:
    - SPX < SMA100: +9.24 pts de diferencia
    - SPX < SMA50: +7.21 pts
    - SPX < SMA7: +6.89 pts

14. **BQI_ABS valores altos mejoran PnL:** Correlación r=0.107. Percentil 95 (BQI_ABS ≥ 20.59) muestra +15.31 pts de mejora, aunque con N=129.

---

## (b) Ranking de Features

### Correlaciones Spearman (Top 20)

| Feature | N | Spearman r | p-adj | Pearson r | p-adj |
|---------|---|------------|-------|-----------|-------|
| SPX_minus_SMA100 | 2,565 | -0.1625 | <0.001*** | -0.2345 | <0.001*** |
| SPX_ZScore50 | 2,565 | -0.1594 | <0.001*** | -0.1960 | <0.001*** |
| BQI_V2_ABS | 2,565 | 0.1333 | <0.001*** | 0.1974 | <0.001*** |
| SPX_MACD_Line | 2,565 | -0.1292 | <0.001*** | -0.2118 | <0.001*** |
| SPX_HV50 | 2,565 | 0.1267 | <0.001*** | 0.1735 | <0.001*** |
| SPX_minus_SMA200 | 2,565 | -0.1262 | <0.001*** | -0.1746 | <0.001*** |
| FF_BAT | 2,298 | 0.1255 | <0.001*** | 0.0454 | 0.038* |
| SPX_minus_SMA50 | 2,565 | -0.1249 | <0.001*** | -0.2212 | <0.001*** |
| r | 2,565 | 0.1240 | <0.001*** | 0.1197 | <0.001*** |
| SPX_ROC20 | 2,565 | -0.1210 | <0.001*** | -0.1730 | <0.001*** |
| SPX_MACD_Signal | 2,565 | -0.1203 | <0.001*** | -0.1975 | <0.001*** |
| SPX_minus_SMA20 | 2,565 | -0.1164 | <0.001*** | -0.1760 | <0.001*** |
| SPX_RSI14 | 2,565 | -0.1125 | <0.001*** | -0.1320 | <0.001*** |
| SPX_BB_Pct | 2,565 | -0.1089 | <0.001*** | -0.1183 | <0.001*** |
| SPX_ZScore20 | 2,565 | -0.1089 | <0.001*** | -0.1183 | <0.001*** |
| BQR_1000 | 2,565 | 0.1070 | <0.001*** | 0.0657 | 0.002** |
| BQI_ABS | 2,565 | 0.1070 | <0.001*** | 0.0657 | 0.002** |
| PnLDV | 2,565 | 0.1063 | <0.001*** | 0.0509 | 0.015* |
| EarL | 2,565 | 0.1060 | <0.001*** | 0.1195 | <0.001*** |
| SPX_Stoch_D | 2,565 | -0.1019 | <0.001*** | -0.1064 | <0.001*** |

**Total features significativas (FDR < 0.05):** 44

---

## (c) Reglas por Umbrales

### Top 15 Reglas por Diferencia de Medianas

| Regla | N< | N≥ | Med< | Med≥ | Δ Mediana | p-value |
|-------|----|----|------|------|-----------|---------|
| SPX_HV50 ≥ 28.21 (P85) | 2,180 | 385 | 9.62 | 34.05 | +24.42 | <0.0001 |
| SPX_HV20 ≥ 28.09 (P85) | 2,180 | 385 | 9.88 | 31.15 | +21.27 | <0.0001 |
| BQI_V2_ABS ≥ 56.99 (P90) | 2,308 | 257 | 11.18 | 32.05 | +20.87 | <0.0001 |
| theta_k2 < -1.03 (P10) | 257 | 2,308 | 30.60 | 11.02 | -19.58 | <0.0001 |
| theta_k1 < -1.56 (P10) | 255 | 2,310 | 29.50 | 10.28 | -19.23 | <0.0001 |
| SPX_ATR14 ≥ 64.42 (P90) | 2,307 | 258 | 10.57 | 29.25 | +18.68 | <0.0001 |
| SPX_minus_SMA100 < -325.18 (P10) | 256 | 2,309 | 29.25 | 10.95 | -18.30 | <0.0001 |
| SPX_BB_Width ≥ 617.54 (P90) | 2,307 | 258 | 10.35 | 28.45 | +18.10 | <0.0001 |
| SPX_MACD_Signal < -70.05 (P10) | 250 | 2,315 | 28.76 | 10.72 | -18.04 | <0.0001 |
| VIX_Close ≥ 33.31 (P90) | 2,308 | 257 | 11.43 | 28.62 | +17.20 | <0.0001 |
| EarScore ≥ 108.36 (P90) | 2,308 | 257 | 11.43 | 28.30 | +16.88 | <0.0001 |
| SPX_minus_SMA200 < -355.35 (P10) | 255 | 2,310 | 28.45 | 11.59 | -16.86 | <0.0001 |

### Interpretación de las Reglas

1. **Reglas de Alta Volatilidad (favorables):**
   - `SPX_HV50 ≥ 28`: Entrar cuando volatilidad histórica 50d es alta
   - `VIX_Close ≥ 33`: VIX elevado mejora el PnL mediano

2. **Reglas de Tendencia Negativa (favorables):**
   - `SPX_minus_SMA100 < -325`: SPX muy por debajo de su media → mejor resultado
   - `MACD_Signal < -70`: Momentum claramente negativo

3. **Reglas de Calidad de Posición:**
   - `BQI_V2_ABS ≥ 57`: Posiciones de alta calidad según el índice

---

## (3.VIX) Análisis Especial VIX

### Estadísticas Descriptivas
- **N:** 2,565
- **Media:** 23.82
- **Mediana:** 21.89
- **Rango:** 12.02 - 82.69

### Correlación con Target
| Método | r | p-value |
|--------|---|---------|
| Spearman | 0.0335 | 0.0894 (no sig.) |
| Pearson | 0.1178 | <0.0001*** |

### Análisis por Régimen VIX

| Régimen | N | Mediana PnL | Media PnL | Std |
|---------|---|-------------|-----------|-----|
| VIX Bajo (<P20) | 511 | 9.10 | 14.02 | 27.96 |
| VIX Normal (P20-P80) | 1,540 | 13.00 | 14.27 | 29.41 |
| VIX Alto (>P80) | 512 | 14.85 | 19.75 | 40.09 |

**Conclusión VIX:** Hay una tendencia ligera a mejor PnL con VIX alto, pero la relación no es estadísticamente robusta (Spearman no significativo). La mayor dispersión en VIX alto indica mayor riesgo también.

---

## (3.BQI) Análisis BQI_ABS

### Estadísticas
- **N:** 2,565
- **Media:** 34.96 (sesgada por outliers)
- **Mediana:** 1.24
- **Rango:** 0.48 - 1009.50

### Correlación
| Método | r | p-value |
|--------|---|---------|
| Spearman | 0.1070 | <0.0001*** |
| Pearson | 0.0657 | 0.0009*** |

### Análisis por Percentiles BQI_ABS

| Percentil | Umbral | N≥ | Mediana≥ | Mediana< | Δ Mediana | p-value |
|-----------|--------|-----|----------|----------|-----------|---------|
| P70 | 1.68 | 770 | 16.38 | 9.78 | +6.60 | 0.0002 |
| P80 | 2.10 | 513 | 17.25 | 10.44 | +6.81 | 0.0001 |
| P90 | 4.10 | 257 | 20.10 | 11.15 | +8.95 | <0.0001 |
| P95 | 20.59 | 129 | 26.90 | 11.59 | +15.31 | <0.0001 |

**Conclusión BQI_ABS:** Valores altos de BQI_ABS correlacionan positivamente con mejor PnL. El corte en P70 (≥1.68) ofrece buen balance entre mejora (+6.60) y tamaño muestral (N=770).

---

## (3.Theta) Ratio Theta K1 / Theta K2

### Estadísticas del Ratio
- **Media:** -1.505
- **Mediana:** -1.484
- **Rango:** -2.39 a -1.20

### Correlación
| Variable | Spearman r | p-value |
|----------|------------|---------|
| theta_k1 | -0.0977 | <0.0001*** |
| theta_k2 | -0.0548 | 0.0055** |
| **Ratio θK1/θK2** | **-0.1223** | **<0.0001***** |

### Análisis por Deciles del Ratio

| Decil | N | Media PnL | Mediana PnL |
|-------|---|-----------|-------------|
| 0 (más negativo) | 257 | 19.38 | **18.65** |
| 1 | 256 | 15.77 | 14.78 |
| 2 | 257 | 18.37 | 14.35 |
| ... | ... | ... | ... |
| 8 | 256 | 13.86 | 10.65 |
| 9 (más cercano a -1) | 257 | 8.37 | **4.45** |

**Conclusión Ratio Theta:** Ratios más negativos (θK1 más grande en valor absoluto respecto a θK2) correlacionan con **MEJOR PnL**. El decil inferior muestra mediana de 18.65 vs 4.45 del decil superior. Diferencia = +14.20 pts.

**Interpretación:** Un ratio θK1/θK2 más negativo indica que la pata corta (k1) tiene mayor theta en valor absoluto que la pata larga (k2), lo que podría indicar estructuras con mejor captación de decaimiento temporal.

---

## (3.SMA) Golden Cross y Filtros SMA

### ⚠️ HALLAZGO CONTRAINTUITIVO: Death Cross es MEJOR

| Filtro | N True | N False | Med True | Med False | Δ | p-value | Sig |
|--------|--------|---------|----------|-----------|---|---------|-----|
| **Golden Cross (SMA50>SMA200)** | 1,896 | 669 | 11.64 | 16.27 | **-4.64** | 0.0007 | *** |

**Conclusión:** Operar en Death Cross (SMA50 < SMA200) produce **MEJOR** PnL mediano que en Golden Cross.

### Cruces de SMAs

| Filtro | N True | N False | Med True | Med False | Δ | p-value |
|--------|--------|---------|----------|-----------|---|---------|
| SMA7 > SMA20 | 1,474 | 1,091 | 10.03 | 15.10 | -5.07 | 0.0001*** |
| SMA20 > SMA50 | 1,643 | 922 | 12.47 | 12.69 | -0.21 | 0.1262 |
| SMA7 > SMA50 | 1,603 | 962 | 10.15 | 16.31 | -6.16 | <0.0001*** |
| SMA20 > SMA100 | 1,797 | 768 | 11.70 | 15.23 | -3.53 | 0.0020** |
| **SMA50 > SMA100** | **1,851** | **714** | **10.25** | **19.86** | **-9.61** | **<0.0001***** |
| SMA20 > SMA200 | 1,888 | 677 | 12.01 | 14.90 | -2.89 | 0.0089** |

### SPX vs SMA Individual

| Filtro | N Above | N Below | Med Above | Med Below | Δ | p-value |
|--------|---------|---------|-----------|-----------|---|---------|
| SPX > SMA7 | 1,389 | 1,176 | 8.95 | 15.84 | -6.89 | <0.0001*** |
| SPX > SMA20 | 1,463 | 1,102 | 9.80 | 15.06 | -5.26 | 0.0001*** |
| SPX > SMA50 | 1,552 | 1,013 | 9.19 | 16.40 | -7.21 | <0.0001*** |
| **SPX > SMA100** | **1,667** | **898** | **8.90** | **18.14** | **-9.24** | **<0.0001***** |
| SPX > SMA200 | 1,846 | 719 | 11.60 | 15.60 | -4.00 | 0.0004*** |

**Conclusión SMA:** TODOS los indicadores de tendencia muestran que estar **POR DEBAJO** de las medias móviles mejora el PnL. El filtro más potente es **SPX < SMA100** con +9.24 pts de mejora en mediana.

---

## (5) Combinaciones de Filtros Óptimas

### Baseline
- **N total:** 2,565
- **Mediana PnL:** 12.57

### Mejores Combinaciones

| Combinación | N | Mediana | Δ vs Baseline | Media | Std |
|-------------|---|---------|---------------|-------|-----|
| **SPX < SMA100 + HV50 Alto** | 342 | **30.52** | **+17.95** | 29.53 | 39.09 |
| Death Cross + HV50 Alto | 454 | 27.49 | +14.91 | 23.92 | 37.05 |
| Golden Cross + HV50 Alto | 47 | 29.30 | +16.73 | 34.82 | 37.88 |
| Death Cross + BQI Alto (P70) | 209 | 21.20 | +8.62 | 22.39 | 27.07 |
| SPX < SMA100 + BQI Alto (P70) | 414 | 20.23 | +7.65 | 21.21 | 26.13 |
| Golden Cross + BQI Alto (P70) | 561 | 14.78 | +2.20 | 15.41 | 25.19 |

**Mejor combinación:** `SPX < SMA100 AND SPX_HV50 ≥ 25` → Mediana = 30.52 (+17.95 vs baseline)

---

## (d) Validación Out-of-Sample (OOS)

### Resultados por Fold (TimeSeriesSplit, 5 folds)

| Fold | Spearman OOS | MAE | Top-Decile Lift |
|------|--------------|-----|-----------------|
| 1 | 0.0232 | 28.93 | +4.07 |
| 2 | 0.0111 | 27.41 | +0.38 |
| 3 | -0.0790 | 27.24 | -1.97 |
| 4 | 0.0442 | 23.37 | +13.34 |
| 5 | 0.3553 | 31.48 | +3.09 |

### Resumen OOS
- **Spearman OOS:** 0.071 ± 0.148
- **MAE OOS:** 27.69 ± 2.64
- **Top-decile lift:** +3.78 ± 5.23

### Importancia de Features (Ridge, último fold)

| Feature | Coeficiente |
|---------|-------------|
| SPX_MACD_Signal | +17.58 |
| SPX_MACD_Line | -16.43 |
| SPX_minus_SMA100 | -16.15 |
| PnLDV | +14.83 |
| VIX_zscore_robust | -13.72 |
| iv_spread_k1_k2 | -13.53 |
| SPX_HV50 | +12.66 |

---

## Conclusiones Finales

### Señal Detectada
Existe una señal predictiva **débil pero estadísticamente significativa** para `PnL_fwd_pts_50_mediana`.

### Predictores Clave (en T+0)
1. **Posición del SPX respecto a sus medias móviles** (SPX < SMA100 es el más fuerte)
2. **Volatilidad histórica** (HV50, HV20) - valores altos favorecen
3. **Ratio Theta K1/K2** (valores más negativos = mejor)
4. **Indicadores de calidad** (BQI_ABS, BQI_V2_ABS) - valores altos favorecen
5. **Momentum negativo** (MACD, RSI bajos)

### ⚠️ Hallazgo Contraintuitivo
**El Golden Cross NO es favorable.** Operar en Death Cross (SMA50 < SMA200) produce mejor PnL (+4.64 pts de mejora, p=0.0007).

### Reglas Simples Recomendadas

**Regla Principal (máxima mejora):**
```
SI (SPX < SMA100) Y (SPX_HV50 ≥ 25)
ENTONCES → PnL mediano esperado ~30.5 pts (vs 12.6 baseline)
           Mejora: +17.95 pts | N=342
```

**Regla Alternativa (mayor soporte muestral):**
```
SI (SPX < SMA100) Y (BQI_ABS ≥ 1.68)
ENTONCES → PnL mediano esperado ~20.2 pts
           Mejora: +7.65 pts | N=414
```

**Regla basada en Theta:**
```
SI (Ratio θK1/θK2 < -1.70)  [Percentil 10]
ENTONCES → PnL mediano esperado ~18.6 pts
           Mejora: +6.0 pts | N=257
```

### Resumen de Filtros SMA

| Filtro | Efecto | Recomendación |
|--------|--------|---------------|
| Golden Cross (SMA50>SMA200) | NEGATIVO (-4.6 pts) | NO usar como entrada |
| SPX < SMA100 | POSITIVO (+9.2 pts) | Usar como filtro de entrada |
| SMA50 < SMA100 | POSITIVO (+9.6 pts) | Usar como filtro de entrada |

### Limitaciones
- Alta variabilidad OOS (Spearman varía de -0.08 a +0.36 entre folds)
- Señal podría ser no estacionaria (mejor en periodos de alta volatilidad)
- Los filtros con HV50 alto reducen significativamente el N disponible
- Recomendación: usar como filtro/complemento, no como predictor único

---

*Análisis generado automáticamente. Archivos adicionales: `analysis_w50_correlations.csv`, `analysis_w50_thresholds.csv`, `analysis_w50_extended.py`*
