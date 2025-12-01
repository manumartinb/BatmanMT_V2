# 📊 ANÁLISIS ESTADÍSTICO COMPLETO
## Correlaciones entre Etiquetas de Ventas y PnL Forward Points

---

## 1. 📋 RESUMEN DEL DATASET

- **Total de Observaciones:** 2,609
- **Variables PnL Analizadas:** 5
- **Variables Driver Analizadas:** 6

### Variables PnL:
- `PnL_fwd_pts_01_mediana`
- `PnL_fwd_pts_05_mediana`
- `PnL_fwd_pts_25_mediana`
- `PnL_fwd_pts_50_mediana`
- `PnL_fwd_pts_90_mediana`

### Variables Driver:
- `LABEL_GENERAL_SCORE`
- `BQI_ABS`
- `FF_ATM`
- `delta_total`
- `theta_total`
- `FF_BAT`

---

## 2. 📐 ESCALA DE INTERPRETACIÓN DE CORRELACIONES

### Fuerza de la Correlación:

| Rango | Interpretación | Emoji |
|-------|----------------|-------|
| \|r\| < 0.20 | Muy Débil | 🔵 |
| 0.20 ≤ \|r\| < 0.40 | Débil | 🟢 |
| 0.40 ≤ \|r\| < 0.60 | Moderada | 🟡 |
| 0.60 ≤ \|r\| < 0.80 | Fuerte | 🟠 |
| \|r\| ≥ 0.80 | Muy Fuerte | 🔴 |

### Significancia Estadística:

| Símbolo | P-valor | Interpretación |
|---------|---------|----------------|
| *** | p < 0.001 | Altamente significativa |
| ** | p < 0.01 | Muy significativa |
| * | p < 0.05 | Significativa |
| ns | p ≥ 0.05 | No significativa |

---

## 3. 🏆 TOP 3 DRIVERS POR PODER PREDICTIVO

### #1. **FF_ATM** 🔵

- **Correlación Absoluta Promedio:** 0.0827
- **Calidad de Correlación:** Muy Débil
- **Pearson Promedio:** 0.0874
- **Spearman Promedio:** 0.0780

### #2. **BQI_ABS** 🔵

- **Correlación Absoluta Promedio:** 0.0679
- **Calidad de Correlación:** Muy Débil
- **Pearson Promedio:** 0.0438
- **Spearman Promedio:** 0.0920

### #3. **theta_total** 🔵

- **Correlación Absoluta Promedio:** 0.0588
- **Calidad de Correlación:** Muy Débil
- **Pearson Promedio:** 0.0417
- **Spearman Promedio:** 0.0759

---

## 4. 🔍 HALLAZGOS CLAVE

### ✅ MEJOR DRIVER: **FF_ATM**

- Muestra la correlación más fuerte con PnL (promedio: **0.0827**)
- Calidad de correlación: **Muy Débil**

### 📊 CALIDAD DE LAS CORRELACIONES (Análisis Detallado)

**Resumen por Calidad:**

- **Muy Débil:** 6 driver(s)
  - `FF_ATM` (r = 0.0827)
  - `BQI_ABS` (r = 0.0679)
  - `theta_total` (r = 0.0588)
  - `FF_BAT` (r = 0.0518)
  - `LABEL_GENERAL_SCORE` (r = 0.0487)
  - `delta_total` (r = 0.0292)

**Interpretación General:**

⚠️ **ADVERTENCIA:** Todas las correlaciones son **MUY DÉBILES**. Esto indica que:
- Los drivers analizados tienen un poder predictivo muy limitado sobre el PnL
- Pueden existir otros factores no capturados que influyen más en el rendimiento
- Se recomienda precaución al aplicar filtros basados en estos drivers

### 📈 RENDIMIENTO TOP 10% vs BOTTOM 10%

- **Spread Promedio:** 10.4683
- **Dirección:** POSITIVA ✅ (Mayor FF_ATM → Mayor PnL)

### ⚠️ PARADOJAS Y ANOMALÍAS DETECTADAS

- ⚠️ `BQI_ABS` muestra correlaciones MIXTAS entre ventanas (inconsistente)
- ⚠️ `delta_total` muestra correlaciones MIXTAS entre ventanas (inconsistente)
- ⚠️ `theta_total` muestra correlaciones MIXTAS entre ventanas (inconsistente)

---

## 5. 🎯 RECOMENDACIONES DE FILTROS

### 🛡️ FILTRO CONSERVADOR (P75)

- **Umbral:** `FF_ATM` ≥ 0.1940
- **Retención:** 653 operaciones (25.03%)
- **PnL Esperado Promedio:** 10.5641 puntos

**PnL Esperado por Ventana:**

- Ventana 01: 1.12 pts
- Ventana 05: 2.63 pts
- Ventana 25: 6.99 pts
- Ventana 50: 18.79 pts
- Ventana 90: 23.29 pts

### ⚖️ FILTRO EQUILIBRADO (P90)

- **Umbral:** `FF_ATM` ≥ 0.2981
- **Retención:** 261 operaciones (10.00%)
- **PnL Esperado Promedio:** 13.2942 puntos

**PnL Esperado por Ventana:**

- Ventana 01: 1.93 pts
- Ventana 05: 4.25 pts
- Ventana 25: 9.68 pts
- Ventana 50: 24.81 pts
- Ventana 90: 25.80 pts

### 🚀 FILTRO AGRESIVO (P95)

- **Umbral:** `FF_ATM` ≥ 0.3919
- **Retención:** 131 operaciones (5.02%)
- **PnL Esperado Promedio:** 15.0132 puntos

**PnL Esperado por Ventana:**

- Ventana 01: 3.11 pts
- Ventana 05: 4.85 pts
- Ventana 25: 12.45 pts
- Ventana 50: 26.54 pts
- Ventana 90: 28.12 pts

### 🚫 ANTI-FILTROS (ZONAS A EVITAR)

**ZONA BAJA:** `FF_ATM` ≤ 0.0496

- **Operaciones Afectadas:** 653 (25.03%)
- **Motivo:** Rendimiento significativamente inferior

**PnL Esperado (Zona Baja):**

- Ventana 01: 0.03 pts
- Ventana 05: 0.42 pts
- Ventana 25: 4.27 pts
- Ventana 50: 13.49 pts
- Ventana 90: 13.94 pts

---

## 6. 💡 RECOMENDACIONES FINALES

### 1. 🎯 FILTRO PRINCIPAL

**Usar `FF_ATM` como criterio de selección principal**

- Estrategia recomendada: **Filtro Equilibrado (P90)**
- Ofrece el mejor balance entre selectividad y retención
- Mejora sustancial del PnL esperado con riesgo controlado

### 2. 🔗 FILTROS SECUNDARIOS

Considerar combinar con:

- **`BQI_ABS`** (Rank #2)
- **`theta_total`** (Rank #3)

La combinación de múltiples drivers puede mejorar la robustez del sistema de filtrado.

### 3. 🚫 EXCLUSIONES

Evitar operaciones donde:

- `FF_ATM` < 0.0496 (25% inferior)
- Estas operaciones muestran rendimiento consistentemente bajo

### 4. 📊 MONITOREO Y VALIDACIÓN

- **Seguimiento continuo:** Rastrear estabilidad de correlaciones en el tiempo
- **Validación out-of-sample:** Testear filtros con datos no utilizados en este análisis
- **Adaptación:** Las relaciones pueden evolucionar con cambios en condiciones de mercado
- **Revisión periódica:** Re-ejecutar este análisis trimestral o semestralmente

### 5. ⚠️ ADVERTENCIAS IMPORTANTES

- ⚠️ **Correlaciones débiles:** El poder predictivo es limitado
- Los filtros pueden ofrecer mejoras modestas pero no garantizadas
- Considerar otros factores no capturados en este análisis
- Validación rigurosa es crítica antes de implementación en producción

---

## 7. 📈 TABLA RESUMEN DE RENDIMIENTO

### Comparación: Filtro Equilibrado (P90) vs Sin Filtro

| Ventana | PnL Sin Filtro | PnL Con Filtro | Mejora | Mejora % |
|---------|----------------|----------------|--------|----------|
| 01 días | 0.54 | 1.93 | +1.39 | +256.3% |
| 05 días | 1.40 | 4.25 | +2.85 | +203.0% |
| 25 días | 6.73 | 9.68 | +2.96 | +44.0% |
| 50 días | 15.32 | 24.81 | +9.49 | +62.0% |
| 90 días | 20.33 | 25.80 | +5.47 | +26.9% |

---

## 8. 📚 METODOLOGÍA

### Técnicas Estadísticas Aplicadas:

1. **Correlación de Pearson:** Mide relación lineal entre variables
2. **Correlación de Spearman:** Mide relación monotónica (robusta a outliers)
3. **Análisis por Percentiles:** Identifica umbrales óptimos de filtrado
4. **Análisis por Cuartiles:** Evalúa distribución de rendimiento
5. **Top/Bottom Analysis:** Compara extremos de distribución

### Datos Analizados:

- **Periodo:** Dataset completo disponible
- **N Observaciones:** 2,609
- **Variables:** 6 drivers × 5 ventanas PnL

---

## 📌 CONCLUSIÓN

Este análisis identifica **FF_ATM** como el driver con mayor poder predictivo, aunque las correlaciones son muy débil (0.0827). Se recomienda precaución al implementar filtros y considerar validación exhaustiva con datos out-of-sample antes de uso en producción.

**Próximos pasos recomendados:**

1. Validar resultados con datos históricos no incluidos en este análisis
2. Realizar backtesting de la estrategia de filtrado propuesta
3. Implementar monitoreo en tiempo real de las correlaciones
4. Considerar análisis de regresión multivariante combinando drivers

---

*Informe generado automáticamente el 2025-12-01 12:54:31*
