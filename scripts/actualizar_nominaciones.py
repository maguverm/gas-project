import pandas as pd
import os
from pathlib import Path

# Ruta ancla: portátil entre sistemas operativos.
# Este script debe vivir en GAS-PROJECT/scripts/ (un nivel bajo la raíz del proyecto).
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "nominaciones"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "nominaciones.parquet"

def fix_encoding(texto):
    """Repara mojibake típico (texto UTF-8 mal interpretado como Latin-1)."""
    if not isinstance(texto, str):
        return texto
    try:
        return texto.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texto

archivos = sorted(RAW_DIR.glob('*.xls'))
print(f"Archivos encontrados: {len(archivos)}")

dfs = []
for archivo in archivos:
    print(f"Procesando: {archivo.name}")
    tablas = pd.read_html(archivo)
    t = tablas[3]  # la tabla de datos es siempre la 4ta
    t.columns = t.iloc[0]
    t = t.iloc[1:].reset_index(drop=True)
    dfs.append(t)

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal filas consolidadas: {len(df):,}")

# Renombrar columnas a snake_case
df.columns = [
    'codigo_registro', 'fecha_reporte', 'fecha_gas', 'numero_operacion',
    'codigo_vendedor', 'nombre_vendedor', 'codigo_comprador', 'nombre_comprador',
    'rol_comprador', 'codigo_punto_snt', 'punto_snt', 'codigo_tipo_demanda',
    'tipo_demanda', 'codigo_sector_consumo', 'sector_consumo', 'codigo_destino',
    'destino', 'cantidad_mbtud', 'estado', 'vigente', 'fecha_recibo_nominacion',
    'hora_recibo_nominacion', 'estado_carga', 'usuario', 'fecha_hora_carga'
]

# Tipos de datos
df['fecha_reporte'] = pd.to_datetime(df['fecha_reporte'], format='%Y/%m/%d', errors='coerce')
df['fecha_gas'] = pd.to_datetime(df['fecha_gas'], format='%Y/%m/%d', errors='coerce')
df['cantidad_mbtud'] = pd.to_numeric(df['cantidad_mbtud'], errors='coerce')

# Reparar encoding ANTES de convertir a category
columnas_texto = ['nombre_vendedor', 'nombre_comprador', 'rol_comprador', 'punto_snt',
                   'tipo_demanda', 'sector_consumo', 'destino', 'estado', 'vigente', 'estado_carga']
for col in columnas_texto:
    df[col] = df[col].apply(fix_encoding)

for col in columnas_texto:
    df[col] = df[col].astype('category')

# Eliminar duplicados exactos por si hay solapamiento entre archivos
df = df.drop_duplicates(subset='codigo_registro')

df.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)
size = os.path.getsize(OUTPUT_PATH) / 1024**2
print(f"\n✅ Nominaciones: {size:.1f} MB | {len(df):,} filas")
print(f"   Rango fecha_gas: {df['fecha_gas'].min().date()} — {df['fecha_gas'].max().date()}")