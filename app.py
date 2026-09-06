#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de Mercado BTC/EUR — versión Render + Telegram (24/7, sin depender del Mac)
======================================================================================

CÓMO FUNCIONA (arquitectura)
-----------------------------
Este es un pequeño servidor web. No tiene bucle propio: se queda esperando
"llamadas". Un servicio externo gratuito (cron-job.org) le hace una llamada
cada 5 minutos. Cada llamada dispara UNA comprobación del mercado y, si hay
señal, envía un aviso por TELEGRAM.

    cron-job.org (cada 5 min, fiable)  ->  este servidor en Render  ->  Telegram

Funciona 24/7 aunque tu ordenador esté apagado, porque corre en los
servidores de Render, no en tu Mac.

MISMA ESTRATEGIA VALIDADA (idéntica a las demás versiones):
  1. Acumulación: rango ≤0,60% en ventana de 2h (8 velas de 15m).
  2. Filtro de tendencia: precio > EMA50 y RSI(14) > 50.
  3. Filtro de funding rate: NO en el 10% más alto reciente (Binance/Bybit).
  4. Avisa de CADA señal nueva y distinta (identificada por su zona de
     acumulación), SIN límite de cuántas al día. No repite la misma señal
     mientras el precio sigue rompiendo la misma zona. Tú decides después
     cuántas de esas señales conviertes en operaciones reales.
  5. Orden: entrada 0,17% sobre el techo, SL en el suelo de la zona,
     TP a 1x el riesgo (ratio 1:1). Comprueba cada 5 min sobre la vela de
     15m en formación: avisa PROVISIONAL antes del cierre y CONFIRMADA al cerrar.

RESULTADOS: en bruto, SIN comisiones — pendiente de confirmar en tu bróker.

VARIABLES DE ENTORNO NECESARIAS (se configuran en Render, ver guía):
    TELEGRAM_TOKEN     el token que te dio BotFather
    TELEGRAM_CHAT_ID   tu chat ID
    CRON_SECRET        (opcional) una palabra secreta para que solo cron-job.org
                       pueda disparar la comprobación; si la pones aquí, hay que
                       incluirla en la URL que configures en cron-job.org.

NOTA sobre el estado: Render no garantiza guardar datos entre reinicios, así
que la memoria de "señales ya avisadas hoy" se mantiene en memoria y se
resetea cada día (o si Render reinicia el servicio). En el caso poco frecuente
de un reinicio a media jornada, podrías recibir de nuevo el aviso de una señal
que siga activa en ese momento. Es un mal menor asumible.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, request

# --------------------------------------------------------------------------
# CONFIGURACIÓN (parámetros validados)
# --------------------------------------------------------------------------
TICKER = "BTC-EUR"
FETCH_INTERVAL = "5m"
PERIOD = "60d"

WINDOW = 8
RANGE_THRESHOLD = 0.006
ENTRY_BUFFER = 0.001718
RR = 1.0
STALE_THRESHOLD = 0.0015

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_URL_BYBIT = "https://api.bybit.com/v5/market/funding/history"
FUNDING_SYMBOL = "BTCUSDT"
FUNDING_LOOKBACK = 200
FUNDING_EXTREME_PCTL = 0.90

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
CRON_SECRET = os.environ.get("CRON_SECRET")  # opcional

# Estado en memoria: recuerda las señales ya avisadas (por su zona) para no repetir,
# pero SIN límite de cuántas al día. Guardamos las claves de señal ya notificadas.
STATE = {"confirmadas_avisadas": set(), "provisionales_avisadas": set(), "dia": None}

app = Flask(__name__)


# --------------------------------------------------------------------------
# UTILIDADES
# --------------------------------------------------------------------------
def log(msg: str):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def send_telegram(text: str):
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        log("[AVISO] Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID. No se envía aviso.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        resp.raise_for_status()
        log("Aviso de Telegram enviado correctamente.")
    except Exception as e:
        log(f"[ERROR] No se pudo enviar el aviso de Telegram: {e}")


# --------------------------------------------------------------------------
# DATOS E INDICADORES (idénticos a la versión validada)
# --------------------------------------------------------------------------
def fetch_price_data() -> pd.DataFrame:
    df5 = yf.download(TICKER, interval=FETCH_INTERVAL, period=PERIOD, progress=False)
    if isinstance(df5.columns, pd.MultiIndex):
        df5.columns = df5.columns.get_level_values(0)
    df5 = df5.rename(columns={"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"})
    if df5.index.tz is None:
        df5.index = df5.index.tz_localize("UTC")
    df5 = df5[["open", "high", "low", "close", "volume"]].dropna()

    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    df15 = df5.resample("15min", label="left", closed="left").agg(agg).dropna()

    last_bucket_start = df15.index[-1]
    n_5m_in_last = int(((df5.index >= last_bucket_start) &
                        (df5.index < last_bucket_start + pd.Timedelta("15min"))).sum())
    df15.attrs["last_candle_closed"] = (n_5m_in_last >= 3)
    df15.attrs["n_5m_in_last"] = n_5m_in_last
    return df15


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s, n=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out[avg_loss == 0] = 100
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema50"] = _ema(df["close"], 50)
    df["rsi14"] = _rsi(df["close"], 14)
    return df


def fetch_funding_percentile():
    try:
        resp = requests.get(FUNDING_URL, params={"symbol": FUNDING_SYMBOL, "limit": FUNDING_LOOKBACK}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            rates = np.array([float(d["fundingRate"]) for d in data])
            current = rates[-1]
            pct = float((rates < current).mean())
            return dict(current=current, percentile=pct, is_extreme=pct >= FUNDING_EXTREME_PCTL, source="Binance")
    except Exception as e:
        log(f"Aviso: Binance funding rate no disponible ({e}). Probando Bybit...")
    try:
        resp = requests.get(FUNDING_URL_BYBIT, params={"category": "linear", "symbol": FUNDING_SYMBOL,
                                                        "limit": FUNDING_LOOKBACK}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("result", {}).get("list", [])
        if data:
            rates = np.array([float(d["fundingRate"]) for d in reversed(data)])
            current = rates[-1]
            pct = float((rates < current).mean())
            return dict(current=current, percentile=pct, is_extreme=pct >= FUNDING_EXTREME_PCTL, source="Bybit")
    except Exception as e:
        log(f"Aviso: Bybit funding rate tampoco disponible ({e}). Se omite ese filtro.")
    return None


# --------------------------------------------------------------------------
# DETECCIÓN DE LA SEÑAL (idéntica a la versión validada)
# --------------------------------------------------------------------------
def detect_signal(df: pd.DataFrame) -> dict:
    if len(df) < WINDOW + 55:
        return dict(status="datos_insuficientes", detail=f"Solo {len(df)} velas disponibles.")

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    ema50 = df["ema50"].values
    rsi14 = df["rsi14"].values
    n = len(df)
    i = n - 1

    window_high = high[i - WINDOW + 1: i + 1].max()
    window_low = low[i - WINDOW + 1: i + 1].min()
    current_range = (window_high - window_low) / window_low

    if current_range <= RANGE_THRESHOLD:
        return dict(status="en_acumulacion", zone_high=round(float(window_high), 2),
                    zone_low=round(float(window_low), 2), price=round(float(close[i]), 2),
                    range_pct=round(current_range * 100, 2))

    lookback_high = high[i - WINDOW - 4: i - 3]
    lookback_low = low[i - WINDOW - 4: i - 3]
    if len(lookback_high) < WINDOW:
        return dict(status="sin_patron")

    zone_high = lookback_high.max()
    zone_low = lookback_low.min()
    zone_range = (zone_high - zone_low) / zone_low

    if not (zone_range <= RANGE_THRESHOLD and close[i] > zone_high and close[i] > zone_low):
        return dict(status="sin_patron")

    price_above_ema50 = close[i] > ema50[i]
    rsi_above_50 = rsi14[i] > 50
    if not (price_above_ema50 and rsi_above_50):
        return dict(status="sin_patron", detail="Ruptura detectada pero no cumple EMA50+RSI.")

    funding = fetch_funding_percentile()
    if funding is not None and funding["is_extreme"]:
        return dict(status="filtrado_por_funding", zone_high=round(float(zone_high), 2),
                    zone_low=round(float(zone_low), 2), funding=funding)

    entry = round(zone_high * (1 + ENTRY_BUFFER), 2)
    sl = round(float(zone_low), 2)
    risk = entry - sl
    tp = round(entry + RR * risk, 2)

    current_price = float(close[i])
    if current_price > entry * (1 + STALE_THRESHOLD):
        distancia_pct = (current_price - entry) / entry * 100
        return dict(status="ruptura_avanzada", zone_high=round(float(zone_high), 2), zone_low=sl,
                    price=round(current_price, 2), entry_teorica=entry, tp_teorico=tp,
                    distancia_pct=round(distancia_pct, 2))

    return dict(status="señal", zone_high=round(float(zone_high), 2), zone_low=sl,
                price=round(current_price, 2), entry=entry, tp=tp, sl=sl,
                rsi=round(float(rsi14[i]), 1), funding=funding,
                provisional=not df.attrs.get("last_candle_closed", True))


def format_message(res: dict, encabezado: str) -> str:
    lines = [
        encabezado,
        f"Zona de acumulación: {res['zone_low']} - {res['zone_high']}",
        f"ENTRADA sugerida: {res['entry']}",
        f"TAKE PROFIT: {res['tp']}  ({(res['tp']/res['entry']-1)*100:+.2f}%)",
        f"STOP LOSS: {res['sl']}  ({(res['sl']/res['entry']-1)*100:+.2f}%)",
        f"RSI(14): {res['rsi']}",
    ]
    if res.get("funding"):
        lines.append(f"Funding rate: percentil {res['funding']['percentile']*100:.0f}% ({res['funding']['source']})")
    lines.append("")
    lines.append("Aviso, no orden automática. Revisa antes de operar.")
    lines.append("Resultados validados en bruto (sin comisiones).")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# LÓGICA DE UNA COMPROBACIÓN (se llama en cada request de cron-job.org)
# --------------------------------------------------------------------------
def do_check() -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # Al cambiar de día, reseteamos las señales avisadas (una misma zona en días
    # distintos se considera señal nueva; y evita que el conjunto crezca sin fin).
    if STATE.get("dia") != today:
        STATE["dia"] = today
        STATE["confirmadas_avisadas"] = set()
        STATE["provisionales_avisadas"] = set()

    df = fetch_price_data()
    df = add_indicators(df)
    res = detect_signal(df)

    if res["status"] == "señal":
        provisional = res.get("provisional", False)
        # Identificador único de la señal: su zona de acumulación (techo + suelo).
        # Mientras el precio rompa la misma zona, es la MISMA señal -> no se repite.
        # Una acumulación distinta (otro techo/suelo) = señal nueva -> sí se avisa.
        signal_key = f"{res['zone_low']}_{res['zone_high']}"

        if provisional:
            if signal_key in STATE["provisionales_avisadas"]:
                return "provisional de esta señal ya avisada, se omite"
            if signal_key in STATE["confirmadas_avisadas"]:
                return "esta señal ya fue confirmada, no se repite la provisional"
            send_telegram(format_message(res, "⚠️ SEÑAL PROVISIONAL (vela de 15m en formación, puede deshacerse)"))
            STATE["provisionales_avisadas"].add(signal_key)
            return "provisional enviada"
        else:
            if signal_key in STATE["confirmadas_avisadas"]:
                return "confirmada de esta señal ya avisada, se omite"
            send_telegram(format_message(res, "🔔 SEÑAL DE ENTRADA CONFIRMADA"))
            STATE["confirmadas_avisadas"].add(signal_key)
            return "confirmada enviada"

    elif res["status"] == "en_acumulacion":
        return f"en acumulación ({res['zone_low']}-{res['zone_high']}, rango {res['range_pct']}%)"
    elif res["status"] == "ruptura_avanzada":
        return f"ruptura avanzada, precio ya {res['distancia_pct']}% sobre entrada teórica, se descarta"
    elif res["status"] == "filtrado_por_funding":
        return "ruptura descartada por funding extremo"
    elif res["status"] == "datos_insuficientes":
        return f"datos insuficientes: {res.get('detail','')}"
    return "sin patrón relevante"


# --------------------------------------------------------------------------
# RUTAS WEB
# --------------------------------------------------------------------------
@app.route("/")
def home():
    # Página simple para comprobar que el servicio está vivo
    return "Monitor BTC/EUR activo. Usa /check para forzar una comprobación.", 200


@app.route("/check")
def check():
    # Si has configurado CRON_SECRET, exige que la llamada incluya ?secret=...
    if CRON_SECRET:
        if request.args.get("secret") != CRON_SECRET:
            return "no autorizado", 403
    try:
        resultado = do_check()
        log(f"Comprobación: {resultado}")
        return f"ok: {resultado}", 200
    except Exception as e:
        log(f"[ERROR] {e}")
        return f"error: {e}", 500


if __name__ == "__main__":
    # Para pruebas locales. En Render se arranca con gunicorn (ver Procfile).
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
