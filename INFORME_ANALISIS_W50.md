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
1. **Posición del SPX respecto a sus medias móviles** (SMA100, SMA200, SMA50)
2. **Volatilidad histórica** (HV50, HV20)
3. **Indicadores de calidad de la posición** (BQI_V2_ABS, FF_BAT)
4. **Momentum** (MACD, RSI)

### Reglas Simples Recomendadas
```
SI (SPX_HV50 > 28 OR SPX_minus_SMA100 < -325)
   Y BQI_V2_ABS > 50
ENTONCES → PnL mediano esperado ~25-30 pts
```

### Limitaciones
- Alta variabilidad OOS (Spearman varía de -0.08 a +0.36 entre folds)
- Señal podría ser no estacionaria (mejor en periodos de alta volatilidad)
- Recomendación: usar como filtro/complemento, no como predictor único

---

*Análisis generado automáticamente. Archivos adicionales: `analysis_w50_correlations.csv`, `analysis_w50_thresholds.csv`*
