import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Test de carga")

BASE_DIR = Path(__file__).parent.parent
PARQUET_PATH = BASE_DIR / "data" / "processed" / "contratos_gas.parquet"

st.write(f"Ruta: {PARQUET_PATH}")
st.write(f"Existe: {PARQUET_PATH.exists()}")

try:
    df = pd.read_parquet(PARQUET_PATH, columns=['fecha_dia', 'cantidad'])
    df = df[df['fecha_dia'] >= '2025-01-01']
    st.success(f"Cargado! {len(df):,} filas")
    st.write(df.head())
except Exception as e:
    st.error(f"Error: {e}")
    import traceback
    st.code(traceback.format_exc())