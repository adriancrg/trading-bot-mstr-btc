# 📈 Quantitative Trading: Statistical Arbitrage & ML (MSTR/BTC vs Crypto)

Este repositorio documenta la investigación, desarrollo y despliegue de sistemas de trading algorítmico utilizando Machine Learning (LSTM y Random Forest). El proyecto explora la eficiencia de los mercados financieros contrastando dos enfoques radicalmente distintos: una ineficiencia estructural en Finanzas Tradicionales (TradFi) frente a un mercado de criptomonedas puro de alta frecuencia.

## 🧠 Tesis de Inversión y Conclusiones del Estudio

El objetivo principal fue explotar ineficiencias de mercado mediante **Reversión a la Media (Mean Reversion)** en activos altamente correlacionados.

### 🟢 El Éxito: MSTR / BTC (Statistical Arbitrage)
Se desarrolló un bot rentable explotando la relación entre MicroStrategy (MSTR) y Bitcoin (BTC).
* **Por qué funciona:** MSTR actúa como un derivado apalancado de BTC. Su cotización incluye un "Premium" impulsado por la psicología humana y el miedo a liquidaciones. Como MSTR cotiza en Nasdaq (con horarios de cierre) y BTC es 24/7, la "goma elástica" (cointegración) entre ambos activos se tensa y destensa en escalas de tiempo manejables (días/horas).
* **Resultado:** Un modelo LSTM con regularización estricta logró batir al mercado en periodos bajistas prolongados, preservando capital al operar en corto (Short) durante expansiones irracionales del spread, generando Alpha positivo en un entorno de mercado de -65%.

### 🔴 El Fracaso Educativo: SOL / BTC (Eficiencia de Mercado)
Se intentó replicar la estrategia en el ecosistema cripto 24/7 operando Solana (SOL) usando a BTC como indicador líder (Lead-Lag).
* **Por qué fracasó:** El análisis de cointegración (p-value) y correlación demostró que a largo plazo las Altcoins sufren *dilución* frente a BTC. Además, los modelos de predicción direccional (LSTM y Random Forest) chocaron contra la **Hipótesis del Mercado Eficiente**.
* **Conclusión Quant:** En cripto puro, cualquier ineficiencia en velas de 15m/1h es arbitrada en milisegundos por bots de Alta Frecuencia (HFT) y *colocation*. Un inversor retail no puede competir en latencia usando datos tabulares (OHLCV). El modelo Random Forest confirmó que el factor más predictivo era el movimiento de BTC en la misma hora, un evento ya "tasado" por el mercado.

---

## ⚙️ Arquitectura Técnica y Metodología

El proyecto está construido modularmente en Python, siguiendo las mejores prácticas de Data Science para evitar el sesgo de supervivencia y el *Data Leakage*.

### 1. Data Pipeline (Extracción y Feature Engineering)
* **APIs:** `yfinance` para datos TradFi y `ccxt` para datos de exchanges Cripto.
* **Sincronización Temporal:** Alineación rigurosa de índices UTC y resolución de *gaps* entre horarios de bolsa (EST) y mercados continuos.
* **Features Matemáticas:** Retornos Logarítmicos, Z-Score sobre medias móviles del Ratio, RSI, y Distancia a la Media Móvil (SMA).

### 2. Modelado de Machine Learning
* **Deep Learning (PyTorch):** Arquitectura **LSTM** de 1 capa con 32 nodos ocultos. Se aplicó fuerte regularización (`Dropout=0.5` y L2 `Weight Decay`) para evitar el sobreajuste al ruido financiero.
* **Ensemble Learning (Scikit-Learn):** Uso de **Random Forest Classifier** (`max_depth=5`) para extraer la "Importancia de Variables" (Feature Importance) y validar la aleatoriedad del mercado cripto.
* **Prevención de Data Leakage:** Separación temporal estricta Train/Test *antes* de ajustar el `StandardScaler`.

### 3. Backtesting Realista y Despliegue
* Simulación vectorial de eventos incluyendo **Comisiones (0.15%)** y **Tasas de Préstamo (Borrow Fees, 20% APR)** para posiciones en corto.
* **Live Execution:** Script integrado con la API de **Alpaca Markets** para Paper Trading automático, incluyendo cálculo dinámico de tamaño de posición (Position Sizing) en dólares reales.

---

## 📂 Estructura del Código

**MSTR**:
* `v4_robust.ipynb`: Script definitivo que descarga datos, limpia valores infinitos y computa indicadores técnicos. Contiene la definición de la red neuronal LSTM en PyTorch y el bucle de entrenamiento. Genera los archivos `.pth` y `.pkl`.
* `alpaca_bot.ipynb`: Script de inferencia y ejecución en vivo. Carga el modelo entrenado, lee el mercado actual en tiempo real y lanza órdenes a la API de Alpaca gestionando el riesgo automáticamente.

**SOL**:
* `botsol\a ver si este funciona\preparar_datos_*.py`: ETL que descarga datos, limpia valores infinitos y computa indicadores técnicos.
* `botsol\a ver si este funciona\entrenamiento_*.py`: Definición de la red neuronal LSTM en PyTorch, bucle de entrenamiento con *Early Stopping*, visualización del Loss y Backtesting. Genera los archivos `.pth` y `.pkl`.
