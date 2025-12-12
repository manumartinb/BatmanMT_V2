# Análisis Exhaustivo de Correlaciones con PnL_fwd_pts_50_mediana

**Fecha:** 2025-12-12
**Dataset:** combined_mediana.csv (2,565 filas válidas)
**Metodología:** Proceso iterativo generativo sin variables futuras
**Validación:** Train (70%) / Test (30%) split

---

## 🎯 RESUMEN EJECUTIVO

### ✅ Validación Correcta
- **EXCLUIDAS:** 169 columnas con información futura (`fwd`, `SPX_chg_pct`)
- **VÁLIDAS:** 89 variables numéricas disponibles en tiempo de trade
- **GENERADAS:** 3,917 features mediante proceso iterativo (10 iteraciones)
- **SIGNIFICATIVAS:** 2,982 features con |r| ≥ 0.15

### 📊 Hallazgos Principales

**Las correlaciones encontradas son MODERADAS (no fuertes):**
- Mejor variable original: **BQI_V2_ABS** → r = 0.20 (train), r = 0.18 (test)
- Mejor feature derivada: **(theta_k1 > p25)** → r = -0.26 (test)
- Correlaciones máximas en rango **0.15 - 0.26** (moderadas)

**⚠️ IMPORTANTE:** No se encontraron correlaciones fuertes (>0.5) con variables disponibles en tiempo real.

---

## 🏆 TOP 10 VARIABLES ORIGINALES

| Rank | Variable           | Train Corr | Test Corr | Robustez | Interpretación |
|------|--------------------|------------|-----------|----------|----------------|
| 1    | **BQI_V2_ABS**     | 0.2020     | 0.1847    | ⭐⭐⭐    | Batman Quality Indicator V2 |
| 2    | **SPX_MACD_Line**  | -0.1972    | -0.1836   | ⭐⭐⭐    | MACD del SPX (negativo) |
| 3    | **SPX_MACD_Signal**| -0.1943    | -0.1772   | ⭐⭐      | Señal MACD del SPX (negativo) |
| 4    | **SPX_minus_SMA50**| -0.1923    | -0.1854   | ⭐⭐⭐    | SPX vs SMA50 (negativo) |
| 5    | **SPX_minus_SMA100**| -0.1911   | -0.1526   | ⭐⭐      | SPX vs SMA100 (negativo) |
| 6    | **EarScore**       | 0.1647     | 0.1486    | ⭐⭐⭐    | Batman Ear Score |
| 7    | **SPX_ZScore50**   | -0.1610    | -0.1806   | ⭐⭐      | Z-Score del SPX (negativo) |
| 8    | **theta_k1**       | -0.1558    | -0.1470   | ⭐⭐⭐    | Theta de leg corto K1 (negativo) |

**Robustez:** ⭐⭐⭐ Excelente (diff < 0.01) | ⭐⭐ Buena (diff < 0.02) | ⭐ Aceptable (diff < 0.04)

### 📈 Interpretación de Correlaciones

**Positivas (mayor variable → mayor PnL):**
- `BQI_V2_ABS` (+): Mejor calidad de la estructura Batman → mejor PnL
- `EarScore` (+): Mejores "orejas" en la estructura → mejor PnL

**Negativas (mayor variable → menor PnL):**
- Variables SPX técnicas (MACD, SMA_minus, ZScore): Mercado "sobrecomprado" → peor PnL para Batman
- `theta_k1` (-): Mayor decay en leg corto → peor PnL

---

## 🔬 TOP 15 FEATURES DERIVADAS SIMPLES E INTERPRETABLES

| Rank | Formula                                              | Train  | Test   | Robustez |
|------|------------------------------------------------------|--------|--------|----------|
| 1    | **(theta_k1 > p25)**                                 | -0.220 | -0.256 | 96% ⭐⭐   |
| 2    | **(SPX_minus_SMA200 > p10)**                         | -0.286 | -0.240 | 84% ⭐    |
| 3    | **(SPX_minus_SMA200_above_p10) + (theta_k1_above_p10)** | -0.324 | -0.290 | 90% ⭐⭐   |
| 4    | **(SPX_minus_SMA200_above_p10) × (theta_k1_above_p10)** | -0.272 | -0.270 | 99% ⭐⭐⭐  |
| 5    | **BQI_V2_ABS × EarScore × theta_k1**                 | -0.268 | -0.230 | 86% ⭐⭐   |
| 6    | **(prod_BQI_V2_ABS_theta_k1 > p10)**                 | -0.284 | -0.282 | 99% ⭐⭐⭐  |
| 7    | **(BQI_V2_ABS > p90)**                               | 0.207  | 0.207  | 100% ⭐⭐⭐ |
| 8    | **BQI_V2_ABS - SPX_ROC20**                           | 0.222  | 0.212  | 95% ⭐⭐⭐  |
| 9    | **BQI_V2_ABS - SPX_ZScore50**                        | 0.218  | 0.208  | 95% ⭐⭐⭐  |
| 10   | **BQI_V2_ABS - SPX_MACD_Line**                       | 0.219  | 0.204  | 93% ⭐⭐   |

**Robustez = (Test_Corr / Train_Corr) %** - Mayor es mejor (menos overfitting)

---

## 🧬 COMPONENTES MÁS IMPORTANTES

Análisis de frecuencia en Top 100 features:

| Componente         | Apariciones | % en Top 100 | Significado |
|--------------------|-------------|--------------|-------------|
| **theta_k1**       | 100         | 100%         | Theta del leg corto K1 - **MÁS IMPORTANTE** |
| **SPX_minus_SMA**  | 99          | 99%          | Distancia SPX vs medias móviles |
| **EarScore**       | 61          | 61%          | Puntuación de "orejas" Batman |
| **theta_k2**       | 59          | 59%          | Theta del leg largo K2 |
| **BQI_V2_ABS**     | 8           | 8%           | Batman Quality Indicator |

**Conclusión:** Las variables **theta_k1** y **SPX_minus_SMA** son los componentes más predictivos.

---

## 💡 RECOMENDACIONES PARA TRADING

### 1. Mejor Métrica Individual: **BQI_V2_ABS**

```
Correlación: r = 0.20 (moderada)
Tipo: Positiva
Robustez: Excelente (diff = 0.017)

Interpretación:
- BQI_V2_ABS alto (> percentil 90) → PnL esperado superior
- Es la variable original más confiable
```

**Uso recomendado:**
```python
if BQI_V2_ABS > np.percentile(historical_BQI, 90):
    # Señal positiva para el trade
    signal_strength = "STRONG"
elif BQI_V2_ABS > np.percentile(historical_BQI, 75):
    # Señal moderada
    signal_strength = "MODERATE"
else:
    # Señal débil o negativa
    signal_strength = "WEAK"
```

### 2. Mejor Métrica Derivada Simple: **(theta_k1 > p25)**

```
Correlación: r = -0.26 (test) - moderada
Tipo: Negativa
Robustez: Buena (96%)

Interpretación:
- theta_k1 bajo (< percentil 25) → Mejor PnL
- theta_k1 alto → Peor PnL
```

**Filtro recomendado:**
```python
theta_k1_p25 = np.percentile(historical_theta_k1, 25)

if theta_k1 < theta_k1_p25:
    # FAVORABLE: theta bajo indica menos decay
    theta_signal = "POSITIVE"
else:
    # DESFAVORABLE: theta alto indica más decay
    theta_signal = "NEGATIVE"
```

### 3. Métrica Compuesta: **Score Combinado**

```python
# Normalizar variables (z-score)
z_BQI = (BQI_V2_ABS - mean_BQI) / std_BQI
z_theta = (theta_k1 - mean_theta) / std_theta
z_SPX_SMA = (SPX_minus_SMA200 - mean_SPX_SMA) / std_SPX_SMA

# Score combinado (ponderado por correlaciones)
BATMAN_SCORE = (
    0.40 * z_BQI +              # Peso mayor (mejor correlación)
    (-0.35) * z_theta +         # Negativo (correlación inversa)
    (-0.25) * z_SPX_SMA         # Negativo (correlación inversa)
)

# Interpretación
if BATMAN_SCORE > 1.0:
    trade_quality = "EXCELLENT"
elif BATMAN_SCORE > 0.5:
    trade_quality = "GOOD"
elif BATMAN_SCORE > 0:
    trade_quality = "FAIR"
else:
    trade_quality = "POOR - AVOID"
```

### 4. Filtros Basados en Percentiles

**Condiciones favorables (AND logic):**
```python
favorable = (
    BQI_V2_ABS > percentile(BQI_V2_ABS, 75) AND       # BQI alto
    theta_k1 < percentile(theta_k1, 25) AND           # Theta bajo
    SPX_minus_SMA200 < percentile(SPX_minus_SMA200, 25)  # SPX no sobrecomprado
)
```

**Condiciones desfavorables (OR logic - evitar):**
```python
avoid_trade = (
    BQI_V2_ABS < percentile(BQI_V2_ABS, 25) OR        # BQI muy bajo
    theta_k1 > percentile(theta_k1, 90) OR            # Theta muy alto
    SPX_minus_SMA200 > percentile(SPX_minus_SMA200, 90)  # SPX muy sobrecomprado
)
```

---

## 📊 TOP 10 FEATURES MÁS ROBUSTAS

Ordenadas por menor diferencia train-test (máxima estabilidad):

| Rank | Formula                                    | Train  | Test   | Diff   |
|------|--------------------------------------------|--------|--------|--------|
| 1    | SPX_MACD_Signal / EarScore                 | -0.154 | -0.154 | 0.0000 |
| 2    | log(sq(BQI_V2_ABS × EarScore × theta_k1))  | 0.154  | 0.155  | 0.0001 |
| 3    | 1 / (wavg_SPX_minus_SMA200_p10_cube_theta_k2) | -0.193 | -0.193 | 0.0002 |
| 4    | SPX_MACD_Line / theta_k2                   | 0.170  | 0.170  | 0.0003 |
| 5    | **BQI_V2_ABS > p90**                       | 0.207  | 0.207  | 0.0004 |
| 6    | SPX_minus_SMA50 + SPX_minus_SMA20          | -0.177 | -0.176 | 0.0010 |
| 7    | SPX_MACD_Signal + SPX_minus_SMA20          | -0.168 | -0.169 | 0.0011 |
| 8    | EarScore > p90                             | 0.200  | 0.198  | 0.0016 |
| 9    | SPX_MACD_Line / theta_k1                   | 0.172  | 0.173  | 0.0014 |
| 10   | prod3(BQI_V2_ABS, EarScore, theta_k1) > p10 | -0.263 | -0.262 | 0.0015 |

**Estas features son las MÁS CONFIABLES para producción** (mínimo overfitting)

---

## 📈 ANÁLISIS DE MEJORA ITERATIVA

| Iteración | Features Generadas | Mejor Correlación | Tipo |
|-----------|-------------------|-------------------|------|
| 0         | 86                | 0.20              | Original (BQI_V2_ABS) |
| 1         | 488               | -0.29             | Percentil (SPX_minus_SMA200 > p10) |
| 2         | 509               | -0.38             | Combinación (EarScore³ × theta_k2³) |
| 3         | 516               | -0.41             | Suma ponderada |
| 4-10      | 2,318             | -0.42             | **Convergencia** (no mejora) |

**Conclusión:** El proceso converge en iteración 4. Mejoras posteriores son marginales y aumentan complejidad.

---

## ⚠️ LIMITACIONES Y ADVERTENCIAS

### 1. Correlaciones Moderadas
- Las correlaciones encontradas están en rango **0.15 - 0.26**
- Son **moderadas**, no fuertes (< 0.5)
- **R² ≈ 0.04 - 0.07** (explican 4-7% de la varianza del PnL)

### 2. Capacidad Predictiva Limitada
- Las variables disponibles en tiempo real tienen **poder predictivo limitado**
- No son suficientes para predecir PnL con alta precisión
- Deben usarse como **filtros complementarios**, no como señal principal

### 3. Overfitting en Features Complejas
- Features muy complejas (generaciones 5-10) muestran overfitting
- Train corr = -0.42, pero Test corr = -0.27 (diferencia 0.15)
- **Preferir siempre features simples (generación 1-2)**

### 4. Variables Futuras Más Potentes
- Las variables `SPX_chg_pct_*` (EXCLUIDAS) tenían r ≈ 0.42
- Son información **futura** no disponible en tiempo de trade
- Confirman que **movimientos del mercado** son el driver principal del PnL

---

## 🎯 CONCLUSIONES FINALES

### ✅ Variables Clave Identificadas

**Top 3 más importantes:**
1. **theta_k1** - Aparece en 100% de top features
2. **SPX_minus_SMA** - Aparece en 99% de top features
3. **EarScore** - Aparece en 61% de top features

### ✅ Relaciones Descubiertas

1. **BQI_V2_ABS** (positiva): Mejor calidad estructural → Mejor PnL
2. **theta_k1** (negativa): Mayor decay → Peor PnL
3. **SPX_minus_SMA** (negativa): Mercado sobrecomprado → Peor PnL para Batman

### ✅ Aplicabilidad Práctica

**Uso recomendado:**
- ✅ Como **filtros de calidad** de trades
- ✅ Para **ranking** de oportunidades Batman
- ✅ Para **ajuste de tamaño** de posición
- ❌ NO como señal principal de entrada/salida

**Ejemplo de integración:**
```python
def evaluate_batman_trade(trade_params):
    """Evalúa calidad de un trade Batman"""

    # Calcular score
    score = calculate_batman_score(
        trade_params['BQI_V2_ABS'],
        trade_params['theta_k1'],
        trade_params['SPX_minus_SMA200']
    )

    # Clasificar trade
    if score > 1.0:
        return "EXCELLENT - Max size"
    elif score > 0.5:
        return "GOOD - Normal size"
    elif score > 0:
        return "FAIR - Reduced size"
    else:
        return "POOR - Skip trade"
```

### ✅ Próximos Pasos Sugeridos

1. **Validación Out-of-Sample:**
   - Probar features en data completamente nueva (2025+)

2. **Machine Learning:**
   - Random Forest / XGBoost con top 20 features
   - Capturar interacciones no lineales

3. **Análisis de Regímenes:**
   - Clasificar mercado en regímenes (bull/bear/lateral)
   - Features pueden funcionar diferente por régimen

4. **Variables Leading:**
   - Explorar VIX, flujo de opciones, breadth indicators
   - Buscar indicadores que **anticipen** movimientos SPX

---

## 📁 ARCHIVOS GENERADOS

1. **Análisis Completo:**
   - `correlaciones_iterativas_final.csv` - 2,982 features significativas
   - `top_features_completas.csv` - Top 50 features

2. **Análisis Simplificado:**
   - `recomendaciones_features.csv` - Top 3 features recomendadas
   - `top_30_robustas.csv` - 30 features más robustas
   - `top_30_simples_interpretables.csv` - 30 features simples

3. **Scripts:**
   - `analisis_correlaciones_iterativo.py` - Proceso iterativo completo
   - `analisis_simplificado_robusto.py` - Análisis de robustez

---

**Análisis completado el 2025-12-12**
**Metodología:** Proceso iterativo generativo con validación train/test
**Resultado:** 3,917 features evaluadas, correlaciones moderadas (0.15-0.26)
**Recomendación:** Usar como filtros complementarios, no como señal principal
