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
import json
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, request

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None

# --------------------------------------------------------------------------
# CONFIGURACIÓN (parámetros validados)
# --------------------------------------------------------------------------
TICKER = "BTC-EUR"
FETCH_INTERVAL = "5m"
PERIOD = "60d"

WINDOW = 8
RANGE_THRESHOLD = 0.006
ENTRY_BUFFER = 0.001718
RR = 1.0            # ratio principal (el que opera de siempre)
RR2 = 1.2           # segundo ratio, en seguimiento paralelo para comparar
# Salida parcial (validada como la mejor configuración, robusta en ambos periodos y
# con comisiones reales). Se operan DOS posiciones iguales e independientes:
#   - Posición A: TP = 1R (ratio 1:1),  SL = suelo de la zona
#   - Posición B: TP = 2.5R (ratio 2,5:1), SL = suelo de la zona
# Cuando A cierra en su TP, se mueve manualmente el SL de B al precio de entrada
# (break-even a 0%, que en backtest batió al +0,2%). 2,5R batió a 2R y a 3R.
RR_PARCIAL_1 = 1.0  # Posición A: objetivo 1R
RR_PARCIAL_2 = 2.5  # Posición B: objetivo 2,5R
RR3 = 2.5           # tercer ratio en seguimiento en Google Sheets = objetivo de la Posición B
STALE_THRESHOLD = 0.0015

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_URL_BYBIT = "https://api.bybit.com/v5/market/funding/history"
FUNDING_SYMBOL = "BTCUSDT"
FUNDING_LOOKBACK = 200
FUNDING_EXTREME_PCTL = 0.90

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
CRON_SECRET = os.environ.get("CRON_SECRET")  # opcional

# Google Sheets: ID de la hoja y credenciales del robot (cuenta de servicio).
# GOOGLE_CREDENTIALS_JSON contiene el CONTENIDO del archivo .json (pegado como
# variable de entorno en Render). SHEET_ID es el identificador de la hoja (de su URL).
SHEET_ID = os.environ.get("SHEET_ID")
GOOGLE_CREDENTIALS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
SHEET_TAB = "Avisos"
HORIZON_DIAS = 10   # días que se sigue una señal pendiente antes de marcarla "sin resolver"

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
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_SYMBOL = "BTCEUR"   # el mismo par de donde salieron los 6,5 años de datos


def _descargar_binance_5m(dias=60) -> pd.DataFrame:
    """Descarga velas de 5m de BTCEUR desde la API pública de Binance.
    Pagina en bloques de 1000 velas hasta cubrir ~`dias` días."""
    limite_por_llamada = 1000
    ahora_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    inicio_ms = ahora_ms - dias * 24 * 60 * 60 * 1000
    filas = []
    cursor = inicio_ms
    while cursor < ahora_ms:
        params = {"symbol": BINANCE_SYMBOL, "interval": "5m",
                  "startTime": cursor, "limit": limite_por_llamada}
        resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
        resp.raise_for_status()
        lote = resp.json()
        if not lote:
            break
        filas.extend(lote)
        ultimo = lote[-1][0]
        if ultimo <= cursor:
            break
        cursor = ultimo + 1
        if len(lote) < limite_por_llamada:
            break
    if not filas:
        raise ValueError("Binance devolvió vacío")
    df = pd.DataFrame(filas, columns=["open_time", "open", "high", "low", "close", "volume",
                                       "close_time", "qv", "trades", "tb", "tq", "ig"])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _descargar_yfinance_5m() -> pd.DataFrame:
    """Respaldo: velas de 5m desde yfinance (Yahoo)."""
    df5 = yf.download(TICKER, interval=FETCH_INTERVAL, period=PERIOD, progress=False)
    if isinstance(df5.columns, pd.MultiIndex):
        df5.columns = df5.columns.get_level_values(0)
    df5 = df5.rename(columns={"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"})
    if df5.index.tz is None:
        df5.index = df5.index.tz_localize("UTC")
    df5 = df5[["open", "high", "low", "close", "volume"]].dropna()
    if df5.empty:
        raise ValueError("yfinance devolvió vacío")
    return df5


def descargar_5m() -> pd.DataFrame:
    """Doble fuente: intenta Binance primero; si falla, usa yfinance. Si fallan
    las dos, lanza la excepción para que el llamador la gestione."""
    try:
        df = _descargar_binance_5m()
        log(f"Datos de precio: Binance OK ({len(df)} velas de 5m).")
        return df
    except Exception as e:
        log(f"[AVISO] Binance falló ({e}). Probando yfinance...")
    df = _descargar_yfinance_5m()
    log(f"Datos de precio: yfinance OK ({len(df)} velas de 5m).")
    return df


def fetch_price_data() -> pd.DataFrame:
    df5 = descargar_5m()

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
    tp = round(entry + RR * risk, 2)      # TP del ratio principal (1:1)
    tp2 = round(entry + RR2 * risk, 2)    # TP del segundo ratio (1,2:1) en seguimiento
    tp_parcial_1 = round(entry + RR_PARCIAL_1 * risk, 2)  # Posición A: TP a 1R
    tp_parcial_2 = round(entry + RR_PARCIAL_2 * risk, 2)  # Posición B: TP a 2,5R

    current_price = float(close[i])
    if current_price > entry * (1 + STALE_THRESHOLD):
        distancia_pct = (current_price - entry) / entry * 100
        return dict(status="ruptura_avanzada", zone_high=round(float(zone_high), 2), zone_low=sl,
                    price=round(current_price, 2), entry_teorica=entry, tp_teorico=tp,
                    distancia_pct=round(distancia_pct, 2))

    return dict(status="señal", zone_high=round(float(zone_high), 2), zone_low=sl,
                price=round(current_price, 2), entry=entry, tp=tp, tp2=tp2, sl=sl,
                tp_parcial_1=tp_parcial_1, tp_parcial_2=tp_parcial_2,
                rsi=round(float(rsi14[i]), 1), funding=funding,
                provisional=not df.attrs.get("last_candle_closed", True))


def format_message(res: dict, encabezado: str) -> str:
    entry = res['entry']
    sl = res['sl']
    tp_a = res.get('tp_parcial_1', res['tp'])   # Posición A: TP a 1R
    tp_b = res.get('tp_parcial_2')              # Posición B: TP a 2,5R
    pct = lambda x: (x / entry - 1) * 100
    lines = [
        encabezado,
        f"Zona de acumulación: {res['zone_low']} - {res['zone_high']}",
        f"ENTRADA sugerida: {entry}",
        "",
        "Abre DOS posiciones iguales (mitad de capital cada una):",
        "",
        "── Posición A ──",
        f"TP: {tp_a}  ({pct(tp_a):+.2f}%, ratio 1:1)",
        f"SL: {sl}  ({pct(sl):+.2f}%)",
        "",
        "── Posición B ──",
    ]
    if tp_b:
        lines.append(f"TP: {tp_b}  ({pct(tp_b):+.2f}%, ratio 2.5:1)")
    lines += [
        f"SL: {sl}  ({pct(sl):+.2f}%)",
        "",
        "⚠️ Si se cierra A en su TP → edita el SL de B",
        f"   y súbelo a {entry} (precio de entrada)",
        "",
        f"RSI(14): {res['rsi']}",
    ]
    if res.get("funding"):
        lines.append(f"Funding rate: percentil {res['funding']['percentile']*100:.0f}% ({res['funding']['source']})")
    lines.append("")
    lines.append("Aviso, no orden automática. Revisa antes de operar.")
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
            guardar_aviso(res, signal_key)  # guardar en el histórico de Google Sheets
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
# GOOGLE SHEETS: guardar avisos y seguir su resultado (TP/SL)
# --------------------------------------------------------------------------
def get_sheet():
    """Conecta con la hoja de Google. Devuelve el objeto worksheet o None si
    no está configurado / falla."""
    if gspread is None:
        log("[AVISO] Falta gspread; no se puede usar Google Sheets.")
        return None
    if not (SHEET_ID and GOOGLE_CREDENTIALS_JSON):
        log("[AVISO] Faltan SHEET_ID o GOOGLE_CREDENTIALS_JSON. No se guarda histórico.")
        return None
    try:
        info = json.loads(GOOGLE_CREDENTIALS_JSON)
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).worksheet(SHEET_TAB)
    except Exception as e:
        log(f"[ERROR] No se pudo conectar con Google Sheets: {e}")
        return None


def guardar_aviso(res: dict, signal_key: str):
    """Añade una fila a la hoja con el aviso confirmado (resultado 'Pendiente')."""
    ws = get_sheet()
    if ws is None:
        return
    try:
        fila = [
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "Confirmada",
            signal_key,
            f"{res['zone_low']}-{res['zone_high']}",
            res["entry"], res["tp"], res["sl"],
            "Pendiente", "",
            res.get("tp2", ""), "Pendiente",   # col J: TP 1,2:1 | col K: Resultado 1,2:1
        ]
        ws.append_row(fila, value_input_option="RAW")
        log(f"Aviso guardado en Google Sheets (ID {signal_key}).")
    except Exception as e:
        log(f"[ERROR] No se pudo guardar el aviso en Google Sheets: {e}")


def resolver_posicion_b(ventana, entry, sl, tp_1r, tp3):
    """Resuelve la Posición B (TP a 2,5R) con su stop DINÁMICO de break-even:
    el stop está en el suelo (sl) hasta que el precio toca 1R (= TP de la Posición A);
    a partir de ahí el stop sube a la entrada (break-even a 0%).
    Devuelve (resultado, fecha) con resultado en {'TP','BE','SL'} o (None, None)
    si todavía no se ha resuelto. Criterio conservador en velas que tocan dos
    niveles a la vez (se asume el desenlace peor)."""
    reached_1r = False
    for ts, row in ventana.iterrows():
        hi = float(row["high"]); lo = float(row["low"])
        if not reached_1r:
            # Stop todavía en el suelo de la zona
            if lo <= sl:
                return "SL", ts            # (si además tocara 1R la misma vela, conservador: SL)
            if hi >= tp3:
                return "TP", ts            # llegó a 2,5R de una vez (implica haber pasado 1R)
            if hi >= tp_1r:
                reached_1r = True
                if lo <= entry:            # la misma vela que alcanza 1R ya recae a la entrada
                    return "BE", ts
        else:
            # Stop ya en break-even (precio de entrada)
            if lo <= entry:
                return "BE", ts            # (si además tocara 2,5R la misma vela, conservador: BE)
            if hi >= tp3:
                return "TP", ts
    return None, None


def actualizar_pendientes():
    """Revisa las filas 'Pendiente' y comprueba, con el histórico de precios,
    si desde la fecha del aviso el precio tocó antes el TP o el SL."""
    ws = get_sheet()
    if ws is None:
        return
    try:
        registros = ws.get_all_values()  # incluye la cabecera en la fila 1
    except Exception as e:
        log(f"[ERROR] No se pudieron leer las filas: {e}")
        return

    if len(registros) < 2:
        return  # solo cabecera

    # Descargar histórico de precios 5m una sola vez (doble fuente Binance/yfinance)
    try:
        precios = descargar_5m()
    except Exception as e:
        log(f"[ERROR] No se pudo descargar histórico para seguir pendientes: {e}")
        return

    for i, fila in enumerate(registros[1:], start=2):  # start=2: fila real en la hoja
        try:
            # ¿Queda algo pendiente en esta fila?
            # col H = ratio 1:1 | col K = ratio 1,2:1 | col S = ratio 2,5:1 (Posición B)
            res1_actual = fila[7] if len(fila) > 7 else ""
            res2_actual = fila[10] if len(fila) > 10 else ""
            res3_actual = fila[18] if len(fila) > 18 else ""
            pend1 = (res1_actual == "Pendiente")
            pend2 = (res2_actual == "Pendiente")
            # S vacío = fila antigua anterior al 2,5:1 -> se rellena hacia atrás
            pend3 = (res3_actual in ("", "Pendiente"))
            if not (pend1 or pend2 or pend3):
                continue

            fecha_aviso = datetime.strptime(fila[0], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            entry = float(str(fila[4]).replace(",", "."))
            tp = float(str(fila[5]).replace(",", "."))
            sl = float(str(fila[6]).replace(",", "."))
            # TP del segundo ratio en col J (índice 9); si falta, lo recalculamos
            try:
                tp2 = float(str(fila[9]).replace(",", "."))
            except (IndexError, ValueError):
                tp2 = round(entry + RR2 * (entry - sl), 2)
            # TP del tercer ratio (2,5:1) en col R (índice 17); si falta, lo recalculamos
            try:
                tp3 = float(str(fila[17]).replace(",", "."))
            except (IndexError, ValueError):
                tp3 = round(entry + RR3 * (entry - sl), 2)

            ventana = precios[precios.index >= fecha_aviso]
            if ventana.empty:
                continue

            # Recorrer las velas una sola vez, resolviendo cada ratio en cuanto toca su TP o el SL
            res1 = None; fecha1 = None
            res2 = None; fecha2 = None
            for ts, row in ventana.iterrows():
                hi = float(row["high"]); lo = float(row["low"])
                # Ratio 1:1
                if res1 is None:
                    if hi >= tp and lo <= sl:
                        res1 = "SL"; fecha1 = ts   # misma vela: conservador -> SL
                    elif hi >= tp:
                        res1 = "TP"; fecha1 = ts
                    elif lo <= sl:
                        res1 = "SL"; fecha1 = ts
                # Ratio 1,2:1
                if res2 is None:
                    if hi >= tp2 and lo <= sl:
                        res2 = "SL"; fecha2 = ts
                    elif hi >= tp2:
                        res2 = "TP"; fecha2 = ts
                    elif lo <= sl:
                        res2 = "SL"; fecha2 = ts
                if res1 is not None and res2 is not None:
                    break

            # Ratio 2,5:1 (Posición B, con stop dinámico de break-even): pase aparte
            res3 = None; fecha3 = None
            if pend3:
                res3, fecha3 = resolver_posicion_b(ventana, entry, sl, tp, tp3)

            ahora = datetime.now(timezone.utc)
            caducado = (ahora - fecha_aviso) > timedelta(days=HORIZON_DIAS)

            # Actualizar ratio 1:1 (columnas H=8 resultado, I=9 fecha)
            if pend1:
                if res1 is not None:
                    ws.update_cell(i, 8, res1)
                    ws.update_cell(i, 9, fecha1.strftime("%Y-%m-%d"))
                    log(f"Fila {i} ratio 1:1 resuelto: {res1}")
                elif caducado:
                    ws.update_cell(i, 8, "Sin resolver")
                    ws.update_cell(i, 9, ahora.strftime("%Y-%m-%d"))

            # Actualizar ratio 1,2:1 (columna K=11 resultado)
            if pend2:
                if res2 is not None:
                    ws.update_cell(i, 11, res2)
                    log(f"Fila {i} ratio 1,2:1 resuelto: {res2}")
                elif caducado:
                    ws.update_cell(i, 11, "Sin resolver")

            # Actualizar ratio 2,5:1 / Posición B (col R=18 precio TP | col S=19 resultado)
            if pend3:
                # rellenar el precio del TP 2,5:1 en R si estaba vacío (filas antiguas)
                if str(res3_actual).strip() == "" and (len(fila) <= 17 or str(fila[17]).strip() == ""):
                    ws.update_cell(i, 18, tp3)
                if res3 is not None:
                    ws.update_cell(i, 19, res3)
                    log(f"Fila {i} ratio 2,5:1 (Posición B) resuelto: {res3}")
                elif caducado:
                    ws.update_cell(i, 19, "Sin resolver")
                elif res3_actual == "":
                    ws.update_cell(i, 19, "Pendiente")
        except Exception as e:
            log(f"[AVISO] No se pudo procesar la fila {i}: {e}")
            continue


# --------------------------------------------------------------------------
# RUTAS WEB
# --------------------------------------------------------------------------
@app.route("/diagpend")
def diagpend():
    """Diagnóstico del seguimiento TP/SL: muestra, fila por fila, qué lee de la
    hoja y por qué no se resuelve. No modifica nada."""
    out = []
    ws = get_sheet()
    if ws is None:
        return "No se pudo abrir la hoja.", 200
    try:
        registros = ws.get_all_values()
    except Exception as e:
        return f"Error leyendo filas: {e}", 200

    out.append(f"Total de filas (con cabecera): {len(registros)}")
    if len(registros) >= 2:
        out.append(f"Cabecera leída: {registros[0]}")
        out.append(f"Ejemplo primera fila de datos: {registros[1]}")

    # Descargar precios como hace la función real, y reportar el rango cubierto
    try:
        precios = descargar_5m()
        out.append("")
        out.append(f"Histórico de precios descargado: {len(precios)} velas de 5m")
        out.append(f"  desde {precios.index.min()} hasta {precios.index.max()}")
    except Exception as e:
        out.append(f"ERROR descargando precios: {e}")
        return "\n".join(out), 200

    out.append("")
    out.append("--- Análisis fila por fila de las PENDIENTES ---")
    pendientes = 0
    for i, fila in enumerate(registros[1:], start=2):
        if len(fila) < 8 or fila[7] != "Pendiente":
            continue
        pendientes += 1
        if pendientes > 20:
            out.append("... (más de 20, se corta el detalle)")
            break
        detalle = [f"Fila {i}:"]
        detalle.append(f"  columna Fecha (fila[0]) = '{fila[0]}'")
        # Probar el parseo de fecha con el formato actual
        try:
            fa = datetime.strptime(fila[0], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            detalle.append(f"  fecha parseada OK: {fa}")
            ventana = precios[precios.index >= fa]
            detalle.append(f"  velas de precio desde esa fecha: {len(ventana)}")
            if ventana.empty:
                detalle.append("  ⚠️ VENTANA VACÍA: el histórico descargado no llega tan atrás. "
                               "Esta es la causa de que no se resuelva.")
        except Exception as e:
            detalle.append(f"  ⚠️ FALLO al parsear la fecha: {e}")
            detalle.append("  Esta es la causa: el formato de la fecha en la hoja no coincide "
                           "con '%Y-%m-%d %H:%M'.")
        # Probar los números
        try:
            entry = float(fila[4]); tp = float(fila[5]); sl = float(fila[6])
            detalle.append(f"  entrada={entry} tp={tp} sl={sl} (números OK)")
        except Exception as e:
            detalle.append(f"  ⚠️ FALLO al leer los números (entrada/tp/sl): {e}")
        out.append("\n".join(detalle))

    if pendientes == 0:
        out.append("No hay filas marcadas como 'Pendiente'.")
    return "\n".join(out), 200


@app.route("/diag")
def diag():
    """Diagnóstico de la conexión con Google Sheets. Devuelve en texto claro
    qué está fallando, en vez de un volcado ilegible."""
    lineas = []
    # 1. ¿Están las variables?
    lineas.append(f"SHEET_ID presente: {'sí' if SHEET_ID else 'NO'}")
    if SHEET_ID:
        lineas.append(f"  SHEET_ID = '{SHEET_ID}'")
        lineas.append(f"  longitud: {len(SHEET_ID)} caracteres")
        if "/" in SHEET_ID or "http" in SHEET_ID or "edit" in SHEET_ID:
            lineas.append("  ⚠️ PROBLEMA: el SHEET_ID contiene '/', 'http' o 'edit'. "
                          "Debe ser SOLO la cadena de en medio de la URL, sin nada más.")
    lineas.append(f"GOOGLE_CREDENTIALS_JSON presente: {'sí' if GOOGLE_CREDENTIALS_JSON else 'NO'}")
    if GOOGLE_CREDENTIALS_JSON:
        lineas.append(f"  longitud: {len(GOOGLE_CREDENTIALS_JSON)} caracteres")
        # ¿Es JSON válido?
        try:
            info = json.loads(GOOGLE_CREDENTIALS_JSON)
            lineas.append("  JSON válido: sí")
            lineas.append(f"  client_email en el JSON: {info.get('client_email', 'NO ENCONTRADO')}")
            lineas.append(f"  project_id: {info.get('project_id', 'NO ENCONTRADO')}")
            lineas.append("  --> Comparte la hoja con ese client_email como Editor si no lo has hecho.")
        except Exception as e:
            lineas.append(f"  ⚠️ PROBLEMA: el JSON NO es válido: {e}")
            lineas.append("  Seguramente se pegó incompleto o con algún carácter cambiado en Render.")

    # 2. Intentar la conexión y capturar el error concreto
    lineas.append("")
    lineas.append("Intentando conectar y leer la hoja...")
    if gspread is None:
        lineas.append("  ⚠️ gspread no está instalado.")
        return "\n".join(lineas), 200
    if not (SHEET_ID and GOOGLE_CREDENTIALS_JSON):
        lineas.append("  No se puede intentar: faltan variables.")
        return "\n".join(lineas), 200
    try:
        info = json.loads(GOOGLE_CREDENTIALS_JSON)
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(SHEET_ID)
        lineas.append(f"  ✅ Hoja abierta correctamente: '{sh.title}'")
        ws = sh.worksheet(SHEET_TAB)
        lineas.append(f"  ✅ Pestaña '{SHEET_TAB}' encontrada. Filas actuales: {ws.row_count}")
        lineas.append("  TODO CORRECTO: la conexión funciona.")
    except Exception as e:
        msg = str(e)[:300]
        lineas.append(f"  ⚠️ Error al conectar: {msg}")
        if "PERMISSION_DENIED" in msg or "403" in msg or "does not have permission" in msg:
            lineas.append("  CAUSA PROBABLE: la hoja no está compartida con el client_email de arriba (como Editor).")
        elif "not found" in msg.lower() or "404" in msg:
            lineas.append("  CAUSA PROBABLE: el SHEET_ID no corresponde a ninguna hoja (revísalo).")
        elif "csp.withgoogle" in msg or "DOCTYPE" in msg:
            lineas.append("  CAUSA PROBABLE: SHEET_ID mal formado o credenciales inválidas (Google devuelve página de bloqueo).")
    return "\n".join(lineas), 200


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
        # Tras comprobar el mercado, actualizar el resultado (TP/SL) de los avisos
        # pendientes en el histórico. Se hace en cada llamada; es ligero.
        try:
            actualizar_pendientes()
        except Exception as e:
            log(f"[AVISO] Fallo al actualizar pendientes: {e}")
        log(f"Comprobación: {resultado}")
        return f"ok: {resultado}", 200
    except Exception as e:
        log(f"[ERROR] {e}")
        return f"error: {e}", 500


@app.route("/test")
def test():
    # Envía un mensaje de prueba a Telegram, para confirmar que la notificación
    # llega bien al móvil sin esperar a una señal real de mercado.
    send_telegram(
        "✅ Prueba del monitor BTC/EUR.\n\n"
        "Si ves este mensaje, Telegram está configurado correctamente y "
        "recibirás aquí los avisos de señal (provisional y confirmada).\n\n"
        "Puedes ignorar este mensaje de prueba."
    )
    return "Mensaje de prueba enviado a Telegram. Revisa tu móvil.", 200


if __name__ == "__main__":
    # Para pruebas locales. En Render se arranca con gunicorn (ver Procfile).
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
