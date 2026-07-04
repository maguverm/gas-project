import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
PARQUET_PATH = BASE_DIR / "data" / "processed" / "contratos_gas.parquet"
DEMANDA_PATH = BASE_DIR / "data" / "processed" / "demanda_gestor.parquet"
PRODUCCION_PATH = BASE_DIR / "data" / "processed" / "produccion_gestor.parquet"
POTENCIAL_PATH = BASE_DIR / "data" / "processed" / "potencial_produccion.parquet"
NOMINACIONES_PATH = BASE_DIR / "data" / "processed" / "nominaciones.parquet"

st.set_page_config(page_title="Monitor de Gas Natural - Superservicios", page_icon="⛽", layout="wide")


def fmt(valor, decimales=1):
    resultado = f"{valor:,.{decimales}f}"
    return resultado.replace(',','X').replace('.', ',').replace('X','.')


def fix_encoding(texto):
    """Repara mojibake típico (texto UTF-8 mal interpretado como Latin-1)."""
    if not isinstance(texto, str):
        return texto
    try:
        return texto.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return texto


def limpiar_columnas_texto(df, columnas):
    """Aplica fix_encoding a una lista de columnas de texto de un DataFrame."""
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].apply(fix_encoding)
    return df


@st.cache_data
def cargar_contratos():
    columnas = ['fecha_dia','cantidad','precio','sector_consumo','modalidad','mercado',
                'nombre_vendedor','nombre_comprador','fecha_inicial','fecha_final','no_operacion','tipo_demanda']
    df = pd.read_parquet(PARQUET_PATH, columns=columnas)
    df = df[df['fecha_dia'] >= '2021-01-01'].copy()
    df['cantidad'] = df['cantidad'].astype('int32')
    df['precio'] = df['precio'].astype('float32')
    df = limpiar_columnas_texto(df, ['sector_consumo','modalidad','mercado','nombre_vendedor','nombre_comprador','tipo_demanda'])
    for col in ['sector_consumo','modalidad','mercado','nombre_vendedor','nombre_comprador','tipo_demanda']:
        df[col] = df[col].astype('category')
    return df

@st.cache_data
def cargar_demanda():
    df = pd.read_parquet(DEMANDA_PATH)
    df = limpiar_columnas_texto(df, ['sector_consumo','nombre_operador','tipo_demanda'])
    return df

@st.cache_data
def cargar_produccion():
    df = pd.read_parquet(PRODUCCION_PATH)
    df = limpiar_columnas_texto(df, ['operador','fuente','tipo_produccion'])
    return df

@st.cache_data
def cargar_potencial():
    df = pd.read_parquet(POTENCIAL_PATH)
    df = limpiar_columnas_texto(df, ['campo','razon_social','periodo'])
    return df

@st.cache_data
def cargar_nominaciones():
    columnas = ['fecha_gas','numero_operacion','nombre_vendedor','nombre_comprador','sector_consumo',
                'tipo_demanda','destino','punto_snt','cantidad_mbtud','estado']
    df = pd.read_parquet(NOMINACIONES_PATH, columns=columnas)
    df = df[df['estado'] == 'Registrado'].copy()
    df = df.drop(columns='estado')
    df['numero_operacion'] = pd.to_numeric(df['numero_operacion'], errors='coerce')
    df = df.dropna(subset=['numero_operacion'])
    df['numero_operacion'] = df['numero_operacion'].astype('int64')
    df = df.rename(columns={'numero_operacion': 'no_operacion'})
    df = limpiar_columnas_texto(df, ['nombre_vendedor','nombre_comprador','sector_consumo','tipo_demanda','destino','punto_snt'])
    for col in ['nombre_vendedor','nombre_comprador','sector_consumo','tipo_demanda','destino','punto_snt']:
        df[col] = df[col].astype('category')
    return df

df = cargar_contratos()
df_dem = cargar_demanda()
df_pot = cargar_potencial()
df_nom = cargar_nominaciones()

def fmt(valor, decimales=1):
    resultado = f"{valor:,.{decimales}f}"
    return resultado.replace(',','X').replace('.', ',').replace('X','.')

def fix_encoding(texto):
    """Repara mojibake típico (texto UTF-8 mal interpretado como Latin-1)."""
    if pd.isna(texto):
        return texto
    try:
        return texto.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError, AttributeError):
        return texto

def agrupar(dff, granularidad):
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_dia'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo','dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_dia'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('Q')
        temp = dff[['periodo','fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('Y')
        temp = dff[['periodo','fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    precio_pond = (dff.groupby('periodo', observed=True)
        .apply(lambda x: (x['precio']*x['cantidad']).sum()/x['cantidad'].sum(), include_groups=False)
        .reset_index(name='precio_ponderado'))
    grp = dff.groupby(['periodo','sector_consumo'], observed=True)['cantidad'].sum().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud'] = grp['cantidad'] / (grp['dias_calendario']*1000)
    grp = grp.merge(precio_pond, on='periodo')
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp

def agrupar_demanda(dff, granularidad):
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_registro'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo','dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_registro'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('Q')
        temp = dff[['periodo','fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp['mes'] = temp['fecha_registro'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_registro'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('Y')
        temp = dff[['periodo','fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp['mes'] = temp['fecha_registro'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_registro'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    grp = dff.groupby(['periodo','sector_consumo'], observed=True)['cantidad_entregada'].sum().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud'] = grp['cantidad_entregada'] / (grp['dias_calendario']*1000)
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp

def agrupar_nominaciones(dff, granularidad, col_agrup):
    """
    Agrupa nominaciones por periodo y col_agrup (nombre_vendedor o nombre_comprador).
    cantidad_mbtud ya es MBTUD/día → dividir entre 1000 para GBTUD.
    En agrupaciones > Diario se promedia sobre los días del período.
    """
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_gas'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo','dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_gas'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('Q')
        temp = dff[['periodo','fecha_gas']].drop_duplicates('fecha_gas').copy()
        temp['mes'] = temp['fecha_gas'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_gas'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('Y')
        temp = dff[['periodo','fecha_gas']].drop_duplicates('fecha_gas').copy()
        temp['mes'] = temp['fecha_gas'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_gas'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    # Primero sumar por día y agente, luego promediar al período
    grp_dia = dff.groupby(['fecha_gas', 'periodo', col_agrup], observed=True)['cantidad_mbtud'].sum().reset_index()
    grp = grp_dia.groupby(['periodo', col_agrup], observed=True)['cantidad_mbtud'].mean().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud'] = grp['cantidad_mbtud'] / 1000
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp

def agrupar_nominaciones_sector(dff, granularidad):
    """Agrupación por sector_consumo para el tab Resumen."""
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_gas'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo','dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_gas'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('Q')
        temp = dff[['periodo','fecha_gas']].drop_duplicates('fecha_gas').copy()
        temp['mes'] = temp['fecha_gas'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_gas'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('Y')
        temp = dff[['periodo','fecha_gas']].drop_duplicates('fecha_gas').copy()
        temp['mes'] = temp['fecha_gas'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_gas'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    grp_dia = dff.groupby(['fecha_gas','periodo','sector_consumo'], observed=True)['cantidad_mbtud'].sum().reset_index()
    grp = grp_dia.groupby(['periodo','sector_consumo'], observed=True)['cantidad_mbtud'].mean().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud'] = grp['cantidad_mbtud'] / 1000
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp

def construir_filtros(df, key_prefix):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: vendedor = st.multiselect("Vendedor", sorted(df['nombre_vendedor'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_vendedor")
    with col2: comprador = st.multiselect("Comprador", sorted(df['nombre_comprador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_comprador")
    with col3: modalidad = st.multiselect("Modalidad", sorted(df['modalidad'].dropna().astype(str).unique().tolist()), placeholder="Todas", key=f"{key_prefix}_modalidad")
    with col4: sector = st.multiselect("Sector consumo", sorted(df['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_sector")
    with col5: mercado = st.multiselect("Mercado", sorted(df['mercado'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_mercado")
    with col6: tipo_demanda = st.multiselect("Tipo demanda", sorted(df['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key=f"{key_prefix}_tipo_demanda")
    return vendedor, comprador, modalidad, sector, mercado, tipo_demanda

def construir_filtros_nominaciones(df_n, key_prefix):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: vendedor = st.multiselect("Vendedor", sorted(df_n['nombre_vendedor'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_vendedor")
    with col2: comprador = st.multiselect("Comprador", sorted(df_n['nombre_comprador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_comprador")
    with col3: sector = st.multiselect("Sector consumo", sorted(df_n['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_sector")
    with col4: tipo_demanda = st.multiselect("Tipo demanda", sorted(df_n['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key=f"{key_prefix}_tipo_demanda")
    with col5: destino = st.multiselect("Destino", sorted(df_n['destino'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_destino")
    return vendedor, comprador, sector, tipo_demanda, destino

def aplicar_filtros(dff, vendedor, comprador, modalidad, sector, mercado, tipo_demanda):
    if vendedor: dff = dff[dff['nombre_vendedor'].isin(vendedor)]
    if comprador: dff = dff[dff['nombre_comprador'].isin(comprador)]
    if modalidad: dff = dff[dff['modalidad'].isin(modalidad)]
    if sector: dff = dff[dff['sector_consumo'].isin(sector)]
    if mercado: dff = dff[dff['mercado'].isin(mercado)]
    if tipo_demanda: dff = dff[dff['tipo_demanda'].isin(tipo_demanda)]
    return dff

def aplicar_filtros_nominaciones(dff, vendedor, comprador, sector, tipo_demanda, destino):
    if vendedor: dff = dff[dff['nombre_vendedor'].isin(vendedor)]
    if comprador: dff = dff[dff['nombre_comprador'].isin(comprador)]
    if sector: dff = dff[dff['sector_consumo'].isin(sector)]
    if tipo_demanda: dff = dff[dff['tipo_demanda'].isin(tipo_demanda)]
    if destino: dff = dff[dff['destino'].isin(destino)]
    return dff

def construir_grafico(grp, titulo):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for sec in sorted(grp['sector_consumo'].astype(str).unique()):
        data_sec = grp[grp['sector_consumo'].astype(str) == sec]
        fig.add_trace(go.Bar(name=sec, x=data_sec['periodo_str'], y=data_sec['gbtud']), secondary_y=False)
    precio_line = grp.drop_duplicates('periodo_str')[['periodo_str','precio_ponderado']]
    fig.add_trace(go.Scatter(name='Precio Ponderado', x=precio_line['periodo_str'], y=precio_line['precio_ponderado'],
        mode='lines+markers', line=dict(color='purple', width=2)), secondary_y=True)
    totales = grp.groupby('periodo_str')['gbtud'].sum().reset_index()
    for _, row in totales.iterrows():
        fig.add_annotation(x=row['periodo_str'], y=row['gbtud'], text=fmt(row['gbtud'],1),
            showarrow=False, textangle=-90, font=dict(size=14, color='black'), yshift=18)
    fig.update_layout(barmode='stack', title=titulo, xaxis_title='Período', height=450, legend=dict(orientation='v', x=1.08))
    fig.update_yaxes(title_text="GBTUD", secondary_y=False)
    fig.update_yaxes(title_text="Precio Ponderado (USD/MBTUD)", secondary_y=True)
    return fig

def construir_grafico_demanda(grp, titulo):
    fig = go.Figure()
    for sec in sorted(grp['sector_consumo'].astype(str).unique()):
        data_sec = grp[grp['sector_consumo'].astype(str) == sec]
        fig.add_trace(go.Bar(name=sec, x=data_sec['periodo_str'], y=data_sec['gbtud']))
    totales = grp.groupby('periodo_str')['gbtud'].sum().reset_index()
    for _, row in totales.iterrows():
        fig.add_annotation(x=row['periodo_str'], y=row['gbtud'], text=fmt(row['gbtud'],1),
            showarrow=False, textangle=-90, font=dict(size=14, color='black'), yshift=18)
    fig.update_layout(barmode='stack', title=titulo, xaxis_title='Período', yaxis_title='GBTUD', height=450, legend=dict(orientation='v', x=1.08))
    return fig

def construir_grafico_nom_sector(grp, titulo):
    """Barras apiladas por sector_consumo para nominaciones."""
    fig = go.Figure()
    for sec in sorted(grp['sector_consumo'].astype(str).unique()):
        data_sec = grp[grp['sector_consumo'].astype(str) == sec]
        fig.add_trace(go.Bar(name=sec, x=data_sec['periodo_str'], y=data_sec['gbtud']))
    totales = grp.groupby('periodo_str')['gbtud'].sum().reset_index()
    for _, row in totales.iterrows():
        fig.add_annotation(x=row['periodo_str'], y=row['gbtud'], text=fmt(row['gbtud'],1),
            showarrow=False, textangle=-90, font=dict(size=14, color='black'), yshift=18)
    fig.update_layout(barmode='stack', title=titulo, xaxis_title='Período', yaxis_title='GBTUD', height=450, legend=dict(orientation='v', x=1.08))
    return fig

def construir_grafico_nom_agente(grp, col_agrup, titulo):
    """Barras apiladas por vendedor o comprador para nominaciones."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Top agentes por volumen total para no saturar el gráfico
    top_agentes = (grp.groupby(col_agrup, observed=True)['gbtud']
                   .sum().sort_values(ascending=False).head(15).index.astype(str).tolist())
    grp_plot = grp[grp[col_agrup].astype(str).isin(top_agentes)].copy()
    otros = grp[~grp[col_agrup].astype(str).isin(top_agentes)].copy()
    for agente in top_agentes:
        data_ag = grp_plot[grp_plot[col_agrup].astype(str) == agente]
        fig.add_trace(go.Bar(name=agente, x=data_ag['periodo_str'], y=data_ag['gbtud']), secondary_y=False)
    if not otros.empty:
        otros_grp = otros.groupby('periodo_str', observed=True)['gbtud'].sum().reset_index()
        fig.add_trace(go.Bar(name='Otros', x=otros_grp['periodo_str'], y=otros_grp['gbtud'],
                             marker_color='lightgray'), secondary_y=False)
    # Línea de variación % total
    totales = grp.groupby('periodo_str')['gbtud'].sum().reset_index(name='gbtud_total')
    totales = totales.sort_values('periodo_str')
    totales['var_pct'] = totales['gbtud_total'].pct_change() * 100
    fig.add_trace(go.Scatter(name='Var% período anterior', x=totales['periodo_str'], y=totales['var_pct'],
        mode='lines+markers', line=dict(color='red', width=2),
        hovertemplate='%{x}<br>Var%: %{y:.1f}%<extra></extra>'), secondary_y=True)
    for _, row in totales.iterrows():
        fig.add_annotation(x=row['periodo_str'], y=row['gbtud_total'], text=fmt(row['gbtud_total'],1),
            showarrow=False, textangle=-90, font=dict(size=14, color='black'), yshift=18)
    fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, secondary_y=True)
    fig.update_layout(barmode='stack', title=titulo, xaxis_title='Período', height=480,
                      legend=dict(orientation='v', x=1.08))
    fig.update_yaxes(title_text="GBTUD", secondary_y=False)
    fig.update_yaxes(title_text="Variación % período anterior", secondary_y=True)
    return fig

def pie_chart(data, col, titulo, col_cantidad):
    grp = data.groupby(col, observed=True)[col_cantidad].sum().reset_index()
    grp.columns = [col, 'cantidad']
    grp = grp.sort_values('cantidad', ascending=False)
    total = grp['cantidad'].sum()
    grp['pct'] = grp['cantidad'] / total * 100
    grp['label'] = grp[col].astype(str) + '<br>' + grp['pct'].apply(lambda x: fmt(x,1)) + '%'
    fig = go.Figure(go.Pie(labels=grp[col].astype(str), values=grp['cantidad'], text=grp['label'],
        textinfo='text', textposition='inside', insidetextorientation='radial', hole=0.3,
        hovertemplate='%{label}<br>Valor: %{value:,.1f}<extra></extra>'))
    fig.update_layout(title=titulo, height=400, showlegend=True, legend=dict(orientation='v', x=1.0, y=0.5), margin=dict(t=50,b=20,l=20,r=120))
    return fig

def bar_chart_top(data, col, titulo, col_cantidad, top_n=10):
    grp = data.groupby(col, observed=True)[col_cantidad].sum().reset_index()
    grp.columns = [col, 'cantidad']
    grp = grp.sort_values('cantidad', ascending=False).head(top_n)
    total = grp['cantidad'].sum()
    grp['pct'] = grp['cantidad'] / total * 100
    grp['label'] = grp['pct'].apply(lambda x: fmt(x,1)) + '%'
    fig = go.Figure(go.Bar(x=grp['cantidad'], y=grp[col].astype(str), orientation='h',
        text=grp['label'], textposition='outside', marker_color='steelblue',
        hovertemplate='%{y}<br>Valor: %{x:,.1f}<extra></extra>'))
    fig.update_layout(title=titulo, height=350, yaxis=dict(autorange='reversed'), margin=dict(t=50,b=20,l=20,r=20), xaxis_title='GBTUD')
    return fig

def construir_estacionalidad(dff):
    dff = dff.copy()
    dff['mes'] = dff['fecha_registro'].dt.month
    dff['dias_mes'] = dff['fecha_registro'].dt.days_in_month
    grp = dff.groupby(['mes','dias_mes'])['cantidad_entregada'].sum().reset_index()
    anios = dff['fecha_registro'].dt.year.nunique()
    grp['gbtud'] = grp['cantidad_entregada'] / (grp['dias_mes']*1000*anios)
    meses_nombre = {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}
    grp['mes_str'] = grp['mes'].map(meses_nombre)
    fig = go.Figure(go.Bar(x=grp['mes_str'], y=grp['gbtud'], text=grp['gbtud'].apply(lambda x: fmt(x,1)),
        textposition='outside', marker_color='steelblue'))
    fig.update_layout(title='Estacionalidad — Promedio histórico por mes (GBTUD)', xaxis_title='', yaxis_title='GBTUD', height=400)
    return fig

def construir_tabla_resumen(dff):
    anios = sorted(dff['fecha_registro'].dt.year.unique())
    if len(anios) < 2:
        st.warning("No hay suficientes años para calcular variaciones.")
        return
    anio_t = anios[-1]
    anio_t1 = anios[-2]
    dff = dff.copy()
    dff['anio'] = dff['fecha_registro'].dt.year
    grp = dff.groupby(['nombre_operador','sector_consumo','anio'], observed=True)['cantidad_entregada'].sum().reset_index()
    pivot = grp.pivot_table(index=['nombre_operador','sector_consumo'], columns='anio', values='cantidad_entregada', fill_value=0).reset_index()
    pivot.columns.name = None
    for a in anios:
        if a not in pivot.columns: pivot[a] = 0
    op_totals = pivot.groupby('nombre_operador')[[a for a in anios]].sum().reset_index()
    op_totals['sector_consumo'] = '— TOTAL —'
    nacional = {a: pivot[a].sum() for a in anios}
    pivot['var_pct'] = (pivot[anio_t]-pivot[anio_t1])/pivot[anio_t1].replace(0,float('nan'))*100
    op_t1 = op_totals[['nombre_operador',anio_t1]].rename(columns={anio_t1:'op_t1'})
    pivot = pivot.merge(op_t1, on='nombre_operador')
    pivot['pp_op'] = (pivot[anio_t]-pivot[anio_t1])/pivot['op_t1'].replace(0,float('nan'))*100
    op_totals['var_pct'] = (op_totals[anio_t]-op_totals[anio_t1])/op_totals[anio_t1].replace(0,float('nan'))*100
    op_totals['pp_nac'] = (op_totals[anio_t]-op_totals[anio_t1])/nacional[anio_t1]*100
    filas = []
    for op in sorted(pivot['nombre_operador'].astype(str).unique()):
        op_row = op_totals[op_totals['nombre_operador']==op].iloc[0]
        fila_op = {'Operador / Sector': f'▶ {op}'}
        for a in anios: fila_op[str(a)] = fmt(op_row[a]/1000,1)
        fila_op[f'Var% {anio_t1}-{anio_t}'] = fmt(op_row['var_pct'],1)+'%' if not pd.isna(op_row['var_pct']) else 'N/D'
        fila_op['PP → Nacional'] = fmt(op_row['pp_nac'],2)+' pp' if not pd.isna(op_row['pp_nac']) else 'N/D'
        fila_op['PP → Op'] = ''
        filas.append(fila_op)
        sectores_op = pivot[pivot['nombre_operador']==op].sort_values(anio_t, ascending=False)
        for _, row in sectores_op.iterrows():
            fila_sec = {'Operador / Sector': f'   → {row["sector_consumo"]}'}
            for a in anios: fila_sec[str(a)] = fmt(row[a]/1000,1)
            fila_sec[f'Var% {anio_t1}-{anio_t}'] = fmt(row['var_pct'],1)+'%' if not pd.isna(row['var_pct']) else 'N/D'
            fila_sec['PP → Nacional'] = ''
            fila_sec['PP → Op'] = fmt(row['pp_op'],2)+' pp' if not pd.isna(row['pp_op']) else 'N/D'
            filas.append(fila_sec)
    st.dataframe(pd.DataFrame(filas), use_container_width=True, height=500)

def agrupar_balance(dff_cont, dff_dem, granularidad):
    dff_cont = dff_cont.copy()
    dff_dem = dff_dem.copy()
    if granularidad == "Diario":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('D')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('D')
        dias_cont = dff_cont.groupby('periodo', observed=True)['fecha_dia'].first().reset_index()
        dias_cont['dias_calendario'] = 1
        dias_cont = dias_cont[['periodo','dias_calendario']]
        dias_dem = dff_dem.groupby('periodo', observed=True)['fecha_registro'].first().reset_index()
        dias_dem['dias_calendario'] = 1
        dias_dem = dias_dem[['periodo','dias_calendario']]
    elif granularidad == "Mensual":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('M')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('M')
        dias_cont = dff_cont.groupby('periodo', observed=True)['fecha_dia'].first().dt.days_in_month.reset_index()
        dias_cont.columns = ['periodo','dias_calendario']
        dias_dem = dff_dem.groupby('periodo', observed=True)['fecha_registro'].first().dt.days_in_month.reset_index()
        dias_dem.columns = ['periodo','dias_calendario']
    elif granularidad == "Trimestral":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('Q')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('Q')
        temp = dff_cont[['periodo','fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias_cont = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_cont.columns = ['periodo','dias_calendario']
        temp2 = dff_dem[['periodo','fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp2['mes'] = temp2['fecha_registro'].dt.to_period('M')
        temp2['dias_mes'] = temp2['fecha_registro'].dt.days_in_month
        dias_dem = temp2.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_dem.columns = ['periodo','dias_calendario']
    elif granularidad == "Anual":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('Y')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('Y')
        temp = dff_cont[['periodo','fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias_cont = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_cont.columns = ['periodo','dias_calendario']
        temp2 = dff_dem[['periodo','fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp2['mes'] = temp2['fecha_registro'].dt.to_period('M')
        temp2['dias_mes'] = temp2['fecha_registro'].dt.days_in_month
        dias_dem = temp2.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_dem.columns = ['periodo','dias_calendario']
    grp_cont = dff_cont.groupby('periodo', observed=True)['cantidad'].sum().reset_index()
    grp_cont = grp_cont.merge(dias_cont, on='periodo')
    grp_cont['gbtud_cont'] = grp_cont['cantidad'] / (grp_cont['dias_calendario']*1000)
    grp_cont['periodo_str'] = grp_cont['periodo'].astype(str)
    grp_dem = dff_dem.groupby('periodo', observed=True)['cantidad_entregada'].sum().reset_index()
    grp_dem = grp_dem.merge(dias_dem, on='periodo')
    grp_dem['gbtud_dem'] = grp_dem['cantidad_entregada'] / (grp_dem['dias_calendario']*1000)
    grp_dem['periodo_str'] = grp_dem['periodo'].astype(str)
    balance = grp_cont[['periodo_str','gbtud_cont']].merge(grp_dem[['periodo_str','gbtud_dem']], on='periodo_str', how='outer').fillna(0)
    balance['diferencia'] = balance['gbtud_cont'] - balance['gbtud_dem']
    balance['pct_sobre_demanda'] = (balance['diferencia'] / balance['gbtud_dem'].replace(0,float('nan')) * 100)
    return balance

def bar_chart_top_gbtud(data, col, titulo, n_dias, top_n=10):
    """Ranking top N por GBTUD/día promedio en el intervalo (divide el acumulado entre n_dias)."""
    grp = data.groupby(['fecha_gas', col], observed=True)['cantidad_mbtud'].sum().reset_index()
    grp = grp.groupby(col, observed=True)['cantidad_mbtud'].sum().reset_index()
    grp['gbtud'] = grp['cantidad_mbtud'] / 1000 / n_dias
    grp = grp.sort_values('gbtud', ascending=False).head(top_n)
    grp['label'] = grp['gbtud'].apply(lambda x: fmt(x,1)) + ' GBTUD'
    fig = go.Figure(go.Bar(x=grp['gbtud'], y=grp[col].astype(str), orientation='h',
        text=grp['label'], textposition='outside', marker_color='steelblue',
        hovertemplate='%{y}<br>GBTUD: %{x:,.1f}<extra></extra>'))
    fig.update_layout(title=titulo, height=350, yaxis=dict(autorange='reversed'), margin=dict(t=50,b=20,l=20,r=20), xaxis_title='GBTUD')
    return fig

def cruzar_matched_periodo(dff_cont, dff_nom, granularidad):
    """Compara Contratado vs Nominado por período, usando SOLO las operaciones que cruzan (existen en ambos lados)."""
    ops_cont = set(dff_cont['no_operacion'].dropna().unique())
    ops_nom = set(dff_nom['no_operacion'].dropna().unique())
    ops_cruzan = ops_cont & ops_nom

    dff_cont_m = dff_cont[dff_cont['no_operacion'].isin(ops_cruzan)]
    dff_nom_m = dff_nom[dff_nom['no_operacion'].isin(ops_cruzan)]

    grp_cont = agrupar_contratos_total(dff_cont_m, granularidad) if not dff_cont_m.empty else pd.DataFrame(columns=['periodo_str','gbtud_cont'])
    grp_nom = agrupar_nominaciones_total(dff_nom_m, granularidad) if not dff_nom_m.empty else pd.DataFrame(columns=['periodo_str','gbtud_nom'])

    comp = grp_cont.merge(grp_nom, on='periodo_str', how='outer').fillna(0)
    comp = comp.sort_values('periodo_str')
    comp['diferencia'] = comp['gbtud_cont'] - comp['gbtud_nom']
    comp['pct_cumplimiento'] = comp['gbtud_nom'] / comp['gbtud_cont'].replace(0, float('nan')) * 100

    cobertura = {
        'n_ops_cont': len(ops_cont),
        'n_ops_nom': len(ops_nom),
        'n_cruzan': len(ops_cruzan),
        'n_huerfanas_nom': len(ops_nom - ops_cont),
        'n_huerfanas_cont': len(ops_cont - ops_nom),
    }
    return comp, cobertura

def ranking_agente_cruce(dff_cont, dff_nom, col_agrup, n_dias, top_n=15):
    """Ranking por agente (vendedor o comprador) de Contratado vs Nominado, usando solo operaciones que cruzan."""
    ops_cont = set(dff_cont['no_operacion'].dropna().unique())
    ops_nom = set(dff_nom['no_operacion'].dropna().unique())
    ops_cruzan = ops_cont & ops_nom

    dff_cont_m = dff_cont[dff_cont['no_operacion'].isin(ops_cruzan)]
    dff_nom_m = dff_nom[dff_nom['no_operacion'].isin(ops_cruzan)]

    grp_cont = dff_cont_m.groupby(col_agrup, observed=True)['cantidad'].sum().reset_index()
    grp_cont['gbtud_cont'] = grp_cont['cantidad'] / (n_dias*1000)

    grp_nom_dia = dff_nom_m.groupby(['fecha_gas', col_agrup], observed=True)['cantidad_mbtud'].sum().reset_index()
    grp_nom = grp_nom_dia.groupby(col_agrup, observed=True)['cantidad_mbtud'].sum().reset_index()
    grp_nom['gbtud_nom'] = grp_nom['cantidad_mbtud'] / 1000 / n_dias

    comp = grp_cont[[col_agrup,'gbtud_cont']].merge(grp_nom[[col_agrup,'gbtud_nom']], on=col_agrup, how='outer').fillna(0)
    comp['diferencia'] = comp['gbtud_cont'] - comp['gbtud_nom']
    comp['pct_cumplimiento'] = comp['gbtud_nom'] / comp['gbtud_cont'].replace(0, float('nan')) * 100
    comp = comp.sort_values('gbtud_cont', ascending=False).head(top_n)
    return comp

def construir_grafico_cruce_periodo(comp, titulo):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name='Contratado', x=comp['periodo_str'], y=comp['gbtud_cont'], marker_color='steelblue'), secondary_y=False)
    fig.add_trace(go.Bar(name='Nominado', x=comp['periodo_str'], y=comp['gbtud_nom'], marker_color='darkorange'), secondary_y=False)
    fig.add_trace(go.Scatter(name='% Cumplimiento', x=comp['periodo_str'], y=comp['pct_cumplimiento'],
        mode='lines+markers', line=dict(color='red', width=2)), secondary_y=True)
    fig.add_hline(y=100, line_dash='dash', line_color='gray', line_width=1, secondary_y=True)
    fig.update_layout(barmode='group', title=titulo, xaxis_title='Período', height=450, legend=dict(orientation='v', x=1.08))
    fig.update_yaxes(title_text="GBTUD", secondary_y=False)
    fig.update_yaxes(title_text="% Cumplimiento", secondary_y=True)
    return fig

def construir_grafico_ranking_cruce(comp, col_agrup, titulo):
    comp = comp.sort_values('gbtud_cont', ascending=True)  # ascendente para que el mayor quede arriba en horizontal
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Contratado', x=comp['gbtud_cont'], y=comp[col_agrup].astype(str), orientation='h', marker_color='steelblue'))
    fig.add_trace(go.Bar(name='Nominado', x=comp['gbtud_nom'], y=comp[col_agrup].astype(str), orientation='h', marker_color='darkorange'))
    fig.update_layout(barmode='group', title=titulo, xaxis_title='GBTUD/día', height=450, legend=dict(orientation='v', x=1.02))
    return fig

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⛽ Gas Natural")
    st.markdown("---")
    seccion = st.radio("Sección",
        ["⛽ Contratación","📊 Demanda","🔄 Balance de Mercado",
         "🔋 Producción","⚡ Declaración de Producción","📋 Nominaciones"],
        label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Contratos al: {df['fecha_dia'].max().strftime('%d/%m/%Y')}")
    st.caption(f"Demanda al: {df_dem['fecha_registro'].max().strftime('%d/%m/%Y')}")
    st.caption(f"Nominaciones al: {df_nom['fecha_gas'].max().strftime('%d/%m/%Y')}")
    
    if st.button("🔄 Actualizar datos"):
        st.cache_data.clear()
        st.rerun()

def agrupar_contratos_total(dff, granularidad):
    """Agrupa contratos por período, total (sin split por sector). Devuelve periodo_str y gbtud_cont."""
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_dia'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo','dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_dia'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('Q')
        temp = dff[['periodo','fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('Y')
        temp = dff[['periodo','fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    grp = dff.groupby('periodo', observed=True)['cantidad'].sum().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud_cont'] = grp['cantidad'] / (grp['dias_calendario']*1000)
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp[['periodo_str','gbtud_cont']]

def agrupar_nominaciones_total(dff, granularidad):
    """Agrupa nominaciones por período (dff ya filtrado a una empresa). Suma diaria, luego promedio del período."""
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_gas'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo','dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_gas'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('Q')
        temp = dff[['periodo','fecha_gas']].drop_duplicates('fecha_gas').copy()
        temp['mes'] = temp['fecha_gas'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_gas'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_gas'].dt.to_period('Y')
        temp = dff[['periodo','fecha_gas']].drop_duplicates('fecha_gas').copy()
        temp['mes'] = temp['fecha_gas'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_gas'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo','dias_calendario']
    grp_dia = dff.groupby(['fecha_gas','periodo'], observed=True)['cantidad_mbtud'].sum().reset_index()
    grp = grp_dia.groupby('periodo', observed=True)['cantidad_mbtud'].mean().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud_nom'] = grp['cantidad_mbtud'] / 1000
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp[['periodo_str','gbtud_nom']]

def construir_tabla_nominacion_detalle(dff_nom, col_agrup, col_detalle):
    """Tabla: filas = agente + (tipo_demanda o sector_consumo), columnas = fecha_gas diaria, valores = MBTUD nominado.
    Excluye filas (agente y detalle) cuyo total sea 0 en todo el período."""
    if dff_nom.empty:
        return pd.DataFrame()

    dff = dff_nom.copy()
    dff[col_detalle] = dff[col_detalle].astype(str)
    dff['fecha_col'] = dff['fecha_gas'].apply(fecha_corta_es)
    orden_fechas = (dff[['fecha_gas','fecha_col']].drop_duplicates()
                    .sort_values('fecha_gas')['fecha_col'].tolist())

    grp = dff.groupby([col_agrup, col_detalle, 'fecha_col'], observed=True)['cantidad_mbtud'].sum().reset_index()
    pivot = grp.pivot_table(index=[col_agrup, col_detalle], columns='fecha_col', values='cantidad_mbtud', fill_value=0)
    pivot = pivot.reindex(columns=orden_fechas, fill_value=0)

    # Excluir combinaciones (agente, detalle) cuyo total en el período sea 0
    pivot = pivot[pivot.sum(axis=1) > 0]

    if pivot.empty:
        return pd.DataFrame()

    # Excluir agentes cuyo total (todas sus filas de detalle) sea 0
    totales_agente = pivot.groupby(level=0).sum().sum(axis=1)
    agentes = sorted(totales_agente[totales_agente > 0].index.astype(str).unique())

    filas = []
    for agente in agentes:
        sub = pivot.loc[agente]
        if isinstance(sub, pd.Series):  # un solo detalle para este agente -> Series, no DataFrame
            sub = sub.to_frame().T
            sub.index = [pivot.loc[agente].name] if hasattr(pivot.loc[agente], 'name') else sub.index
        fila_agente = {'Agente / Detalle': f'▶ {agente}'}
        for col in orden_fechas:
            fila_agente[col] = fmt(sub[col].sum(), 0)
        filas.append(fila_agente)
        for detalle in sorted(sub.index.astype(str)):
            fila_det = {'Agente / Detalle': f'   → {detalle}'}
            for col in orden_fechas:
                fila_det[col] = fmt(sub.loc[detalle, col], 0)
            filas.append(fila_det)
    return pd.DataFrame(filas)

MESES_ES = {1:'ene',2:'feb',3:'mar',4:'abr',5:'may',6:'jun',7:'jul',8:'ago',9:'sep',10:'oct',11:'nov',12:'dic'}

def fecha_corta_es(fecha):
    return f"{fecha.day:02d} {MESES_ES[fecha.month]}"

# ── Contratación ──────────────────────────────────────────────────────────────
if seccion == "⛽ Contratación":
    st.title("⛽ Contratación de Gas Natural")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Contratación","📅 Contratación Vigente","📈 Análisis Adicionales","🏭 Concentración HHI"])

    with tab1:
        st.subheader("Contratación de Gas Natural")
        vendedor, comprador, modalidad, sector, mercado, tipo_demanda = construir_filtros(df, "tab1")
        col6, col7, col8 = st.columns(3)
        with col6: fecha_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(), min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t1_fi")
        with col7: fecha_fin = st.date_input("Fecha fin", value=df['fecha_dia'].max().date(), min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t1_ff")
        with col8: granularidad = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="t1_gran")
        dff = df.copy()
        dff = dff[(dff['fecha_dia'].dt.date >= fecha_inicio) & (dff['fecha_dia'].dt.date <= fecha_fin)]
        dff = aplicar_filtros(dff, vendedor, comprador, modalidad, sector, mercado, tipo_demanda)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total GBTUD", fmt(dff['cantidad'].sum()/(dff['fecha_dia'].dt.days_in_month.mean()*1000),1))
        k2.metric("Precio Ponderado (USD/MBTUD)", fmt((dff['precio']*dff['cantidad']).sum()/dff['cantidad'].sum(),2))
        k3.metric("N° Contratos", fmt(dff['no_operacion'].nunique(),0))
        k4.metric("N° Empresas", fmt(dff['nombre_vendedor'].nunique()+dff['nombre_comprador'].nunique(),0))
        grp = agrupar(dff, granularidad)
        st.plotly_chart(construir_grafico(grp, 'Contratado (GBTUD) por sector de consumo'), use_container_width=True)
        st.subheader("Detalle de contratos")
        tabla = dff[['nombre_vendedor','nombre_comprador','cantidad','precio','modalidad','sector_consumo','tipo_demanda','mercado','fecha_inicial','fecha_final','no_operacion']].copy()
        tabla['cantidad'] = tabla['cantidad'].apply(lambda x: fmt(x,0))
        tabla['precio'] = tabla['precio'].apply(lambda x: fmt(x,2))
        tabla.columns = ['Vendedor','Comprador','Cantidad (MBTU)','Precio (USD/MBTUD)','Modalidad','Sector Consumo','Tipo Demanda','Mercado','Fecha Inicial','Fecha Final','N° Operación']
        st.dataframe(tabla, use_container_width=True)

    with tab2:
        st.subheader("Contratación Vigente")
        vendedor2, comprador2, modalidad2, sector2, mercado2, tipo_demanda2 = construir_filtros(df, "tab2")
        col6b, col7b, col8b = st.columns(3)
        with col6b: vi_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2026-01-01').date(), min_value=df['fecha_inicial'].min().date(), max_value=df['fecha_final'].max().date(), key="t2_fi")
        with col7b: vi_fin = st.date_input("Fecha fin", value=pd.Timestamp('2030-12-31').date(), min_value=df['fecha_inicial'].min().date(), max_value=df['fecha_final'].max().date(), key="t2_ff")
        with col8b: granularidad2 = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="t2_gran")

        @st.cache_data
        def construir_vigente(vendedor, comprador, modalidad, sector, mercado, tipo_demanda):
            cols_contrato = ['no_operacion','nombre_vendedor','nombre_comprador','modalidad','mercado','sector_consumo','tipo_demanda','fecha_inicial','fecha_final','cantidad','precio']
            contratos = df[cols_contrato].groupby(['no_operacion','fecha_inicial','fecha_final','sector_consumo'], observed=True).first().reset_index().copy()
            contratos = aplicar_filtros(contratos, vendedor, comprador, modalidad, sector, mercado, tipo_demanda)
            if contratos.empty: return pd.DataFrame()
            contratos['fecha_inicial'] = pd.to_datetime(contratos['fecha_inicial'])
            contratos['fecha_final'] = pd.to_datetime(contratos['fecha_final'])
            contratos['fechas'] = contratos.apply(lambda r: pd.date_range(r['fecha_inicial'], r['fecha_final'], freq='D'), axis=1)
            df_vig = contratos.explode('fechas').rename(columns={'fechas':'fecha_dia'})
            df_vig = df_vig[['fecha_dia','cantidad','precio','sector_consumo']].copy()
            df_vig['sector_consumo'] = df_vig['sector_consumo'].astype('category')
            return df_vig

        with st.spinner("Calculando contratación vigente..."):
            df_vig = construir_vigente(tuple(vendedor2), tuple(comprador2), tuple(modalidad2), tuple(sector2), tuple(mercado2), tuple(tipo_demanda2))
        if df_vig.empty:
            st.warning("No hay contratos para los filtros seleccionados.")
        else:
            df_vig_filtrado = df_vig[(df_vig['fecha_dia'].dt.date >= vi_inicio) & (df_vig['fecha_dia'].dt.date <= vi_fin)]
            if df_vig_filtrado.empty:
                st.warning("No hay datos en el rango de fechas seleccionado.")
            else:
                k1b, k2b, _ = st.columns(3)
                k1b.metric("Total GBTUD", fmt(df_vig_filtrado['cantidad'].sum()/(df_vig_filtrado['fecha_dia'].dt.days_in_month.mean()*1000),1))
                k2b.metric("Precio Ponderado (USD/MBTUD)", fmt((df_vig_filtrado['precio']*df_vig_filtrado['cantidad']).sum()/df_vig_filtrado['cantidad'].sum(),2))
                grp2 = agrupar(df_vig_filtrado, granularidad2)
                st.plotly_chart(construir_grafico(grp2, 'Contratación Vigente (GBTUD) por sector de consumo'), use_container_width=True)
                st.subheader("Contratos en el período")
                cols_contrato = ['no_operacion','nombre_vendedor','nombre_comprador','modalidad','mercado','sector_consumo','tipo_demanda','fecha_inicial','fecha_final','cantidad','precio']
                contratos_tabla = df[cols_contrato].groupby(['no_operacion','fecha_inicial','fecha_final','sector_consumo'], observed=True).first().reset_index().copy()
                contratos_tabla = aplicar_filtros(contratos_tabla, vendedor2, comprador2, modalidad2, sector2, mercado2, tipo_demanda2)
                contratos_tabla = contratos_tabla[(contratos_tabla['fecha_inicial'].dt.date <= vi_fin) & (contratos_tabla['fecha_final'].dt.date >= vi_inicio)]
                contratos_tabla['cantidad'] = contratos_tabla['cantidad'].apply(lambda x: fmt(x,0))
                contratos_tabla['precio'] = contratos_tabla['precio'].apply(lambda x: fmt(x,2))
                contratos_tabla.columns = ['N° Operación','Vendedor','Comprador','Modalidad','Mercado','Sector','Tipo Demanda','Fecha Inicial','Fecha Final','Cantidad (MBTU)','Precio (USD/MBTUD)']
                st.dataframe(contratos_tabla, use_container_width=True)

    with tab3:
        st.subheader("Análisis Adicionales de Contratación")
        vendedor3, comprador3, modalidad3, sector3, mercado3, tipo_demanda3 = construir_filtros(df, "tab3")
        col_a, col_b, _ = st.columns(3)
        with col_a: fecha_inicio3 = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(), min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t3_fi")
        with col_b: fecha_fin3 = st.date_input("Fecha fin", value=df['fecha_dia'].max().date(), min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t3_ff")
        dff3 = df.copy()
        dff3 = dff3[(dff3['fecha_dia'].dt.date >= fecha_inicio3) & (dff3['fecha_dia'].dt.date <= fecha_fin3)]
        dff3 = aplicar_filtros(dff3, vendedor3, comprador3, modalidad3, sector3, mercado3, tipo_demanda3)
        dias_total3 = (fecha_fin3 - fecha_inicio3).days + 1
        dff3 = dff3.copy()
        dff3['gbtud'] = dff3['cantidad'] / (dias_total3 * 1000)
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(pie_chart(dff3, 'sector_consumo', 'GBTUD promedio por Sector de Consumo', 'gbtud'), use_container_width=True)
        with col2: st.plotly_chart(pie_chart(dff3, 'modalidad', 'GBTUD promedio por Modalidad', 'gbtud'), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3: st.plotly_chart(pie_chart(dff3, 'mercado', 'GBTUD promedio por Mercado', 'gbtud'), use_container_width=True)
        with col4: st.plotly_chart(bar_chart_top(dff3, 'nombre_vendedor', 'Top 10 Vendedores (GBTUD)', 'gbtud'), use_container_width=True)
        col5, _ = st.columns(2)
        with col5: st.plotly_chart(bar_chart_top(dff3, 'nombre_comprador', 'Top 10 Compradores (GBTUD)', 'gbtud'), use_container_width=True)

    with tab4:
        st.subheader("Concentración de Mercado — Índice HHI")
        st.caption("Valores < 1.500 indican competencia, entre 1.500 y 2.500 concentración moderada, y > 2.500 mercado concentrado.")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: hhi_mercado = st.multiselect("Mercado", sorted(df['mercado'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="hhi_mercado")
        with col2: hhi_sector = st.multiselect("Sector consumo", sorted(df['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="hhi_sector")
        with col3: hhi_modalidad = st.multiselect("Modalidad", sorted(df['modalidad'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="hhi_modalidad")
        with col4: hhi_granularidad = st.selectbox("Ver por", ["Mensual","Trimestral","Anual"], key="hhi_gran")
        with col5: hhi_por = st.selectbox("Calcular HHI por", ["Vendedor","Comprador"], key="hhi_por")
        col6, col7 = st.columns(2)
        with col6: hhi_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(), min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="hhi_fi")
        with col7: hhi_fin = st.date_input("Fecha fin", value=df['fecha_dia'].max().date(), min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="hhi_ff")
        dff4 = df.copy()
        dff4 = dff4[(dff4['fecha_dia'].dt.date >= hhi_inicio) & (dff4['fecha_dia'].dt.date <= hhi_fin)]
        if hhi_mercado: dff4 = dff4[dff4['mercado'].isin(hhi_mercado)]
        if hhi_sector: dff4 = dff4[dff4['sector_consumo'].isin(hhi_sector)]
        if hhi_modalidad: dff4 = dff4[dff4['modalidad'].isin(hhi_modalidad)]
        col_empresa = 'nombre_vendedor' if hhi_por == "Vendedor" else 'nombre_comprador'
        if hhi_granularidad == "Mensual": dff4['periodo'] = dff4['fecha_dia'].dt.to_period('M')
        elif hhi_granularidad == "Trimestral": dff4['periodo'] = dff4['fecha_dia'].dt.to_period('Q')
        elif hhi_granularidad == "Anual": dff4['periodo'] = dff4['fecha_dia'].dt.to_period('Y')
        grp_hhi = dff4.groupby(['periodo',col_empresa], observed=True)['cantidad'].sum().reset_index()
        total_periodo = grp_hhi.groupby('periodo', observed=True)['cantidad'].sum().reset_index(name='total')
        grp_hhi = grp_hhi.merge(total_periodo, on='periodo')
        grp_hhi['participacion'] = grp_hhi['cantidad'] / grp_hhi['total'] * 100
        grp_hhi['hhi_parcial'] = grp_hhi['participacion'] ** 2
        hhi_por_periodo = grp_hhi.groupby('periodo', observed=True)['hhi_parcial'].sum().reset_index(name='hhi')
        hhi_por_periodo['periodo_str'] = hhi_por_periodo['periodo'].astype(str)
        top3 = grp_hhi.sort_values(['periodo','participacion'], ascending=[True,False]).groupby('periodo', observed=True).head(3)
        top3_resumen = top3.groupby('periodo', observed=True)[col_empresa].apply(lambda x: ', '.join(x.astype(str))).reset_index()
        top3_resumen.columns = ['periodo','top3']
        hhi_por_periodo = hhi_por_periodo.merge(top3_resumen, on='periodo', how='left')
        hhi_por_periodo['top3'] = hhi_por_periodo['top3'].fillna('Sin datos')
        fig4 = go.Figure()
        fig4.add_hrect(y0=0, y1=1500, fillcolor='green', opacity=0.07, line_width=0, annotation_text='Competitivo', annotation_position='right')
        fig4.add_hrect(y0=1500, y1=2500, fillcolor='yellow', opacity=0.1, line_width=0, annotation_text='Moderado', annotation_position='right')
        fig4.add_hrect(y0=2500, y1=10000, fillcolor='red', opacity=0.07, line_width=0, annotation_text='Concentrado', annotation_position='right')
        fig4.add_trace(go.Scatter(x=hhi_por_periodo['periodo_str'], y=hhi_por_periodo['hhi'], mode='lines+markers', name='HHI',
            line=dict(color='darkblue', width=2), customdata=hhi_por_periodo['top3'],
            hovertemplate='<b>%{x}</b><br>HHI: %{y:,.0f}<br>Top 3: %{customdata}<extra></extra>'))
        fig4.update_layout(title=f'Índice HHI por {hhi_por} — {hhi_granularidad}', xaxis_title='Período', yaxis_title='HHI', height=450,
            yaxis=dict(range=[0, max(10000, hhi_por_periodo['hhi'].max()*1.1)]))
        st.plotly_chart(fig4, use_container_width=True)
        st.subheader("HHI por período")
        tabla_hhi = hhi_por_periodo[['periodo_str','hhi','top3']].copy()
        tabla_hhi['hhi'] = tabla_hhi['hhi'].apply(lambda x: fmt(x,0))
        tabla_hhi.columns = ['Período','HHI',f'Top 3 {hhi_por}es']
        st.dataframe(tabla_hhi, use_container_width=True)

# ── Demanda ───────────────────────────────────────────────────────────────────
elif seccion == "📊 Demanda":
    st.title("📊 Demanda de Gas Natural")
    tab_d1, tab_d2 = st.tabs(["📈 Demanda","🔍 Análisis de Demanda"])

    with tab_d1:
        st.subheader("Demanda de Gas Natural — Registro de Entregas")
        col1, col2, col3 = st.columns(3)
        with col1: d_operador = st.multiselect("Operador", sorted(df_dem['nombre_operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="d_operador")
        with col2: d_sector = st.multiselect("Sector consumo", sorted(df_dem['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="d_sector")
        with col3: d_tipo = st.multiselect("Tipo demanda", sorted(df_dem['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="d_tipo")
        col4, col5, col6 = st.columns(3)
        with col4: d_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(), min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="d_fi")
        with col5: d_fin = st.date_input("Fecha fin", value=df_dem['fecha_registro'].max().date(), min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="d_ff")
        with col6: d_gran = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="d_gran")
        dfd = df_dem.copy()
        dfd = dfd[(dfd['fecha_registro'].dt.date >= d_inicio) & (dfd['fecha_registro'].dt.date <= d_fin)]
        if d_operador: dfd = dfd[dfd['nombre_operador'].isin(d_operador)]
        if d_sector: dfd = dfd[dfd['sector_consumo'].isin(d_sector)]
        if d_tipo: dfd = dfd[dfd['tipo_demanda'].isin(d_tipo)]
        k1d, k2d, _ = st.columns(3)
        k1d.metric("Total GBTUD", fmt(dfd['cantidad_entregada'].sum()/(dfd['fecha_registro'].dt.days_in_month.mean()*1000),1))
        k2d.metric("N° Operadores", fmt(dfd['nombre_operador'].nunique(),0))
        grp_d = agrupar_demanda(dfd, d_gran)
        st.plotly_chart(construir_grafico_demanda(grp_d, 'Demanda (GBTUD) por sector de consumo'), use_container_width=True)
        st.subheader("Detalle de entregas")
        tabla_d = dfd[['fecha_registro','nombre_operador','sector_consumo','tipo_demanda','cantidad_entregada']].copy()
        tabla_d['cantidad_entregada'] = tabla_d['cantidad_entregada'].apply(lambda x: fmt(x,0))
        tabla_d.columns = ['Fecha','Operador','Sector','Tipo Demanda','Cantidad (MBTU)']
        st.dataframe(tabla_d, use_container_width=True)

    with tab_d2:
        st.subheader("Análisis Adicionales de Demanda")
        col1, col2, col3 = st.columns(3)
        with col1: da_operador = st.multiselect("Operador", sorted(df_dem['nombre_operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="da_operador")
        with col2: da_sector = st.multiselect("Sector consumo", sorted(df_dem['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="da_sector")
        with col3: da_tipo = st.multiselect("Tipo demanda", sorted(df_dem['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="da_tipo")
        col4, col5, _ = st.columns(3)
        with col4: da_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(), min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="da_fi")
        with col5: da_fin = st.date_input("Fecha fin", value=df_dem['fecha_registro'].max().date(), min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="da_ff")
        dfa = df_dem.copy()
        dfa = dfa[(dfa['fecha_registro'].dt.date >= da_inicio) & (dfa['fecha_registro'].dt.date <= da_fin)]
        if da_operador: dfa = dfa[dfa['nombre_operador'].isin(da_operador)]
        if da_sector: dfa = dfa[dfa['sector_consumo'].isin(da_sector)]
        if da_tipo: dfa = dfa[dfa['tipo_demanda'].isin(da_tipo)]
        dias_total_da = (da_fin - da_inicio).days + 1
        dfa_gbtud = dfa.copy()
        dfa_gbtud['gbtud'] = dfa_gbtud['cantidad_entregada'] / (dias_total_da * 1000)
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(pie_chart(dfa_gbtud, 'sector_consumo', 'GBTUD promedio por Sector de Consumo', 'gbtud'), use_container_width=True)
        with col2: st.plotly_chart(pie_chart(dfa_gbtud, 'tipo_demanda', 'GBTUD promedio por Tipo', 'gbtud'), use_container_width=True)
        col3, _ = st.columns(2)
        with col3: st.plotly_chart(bar_chart_top(dfa_gbtud, 'nombre_operador', 'Top 10 Operadores (GBTUD)', 'gbtud'), use_container_width=True)
        st.markdown("---")
        st.subheader("Estacionalidad")
        st.plotly_chart(construir_estacionalidad(dfa), use_container_width=True)
        st.markdown("---")
        st.subheader("Resumen por Operador y Sector — Variaciones y Contribuciones")
        construir_tabla_resumen(dfa)

# ── Balance de Mercado ────────────────────────────────────────────────────────
elif seccion == "🔄 Balance de Mercado":
    st.title("🔄 Balance de Mercado")
    st.subheader("Contratación vs Demanda")
    col1, col2, col3, col4 = st.columns(4)
    with col1: bm_empresa = st.multiselect("Empresa (Comprador/Operador)", sorted(set(df['nombre_comprador'].dropna().astype(str).unique()) & set(df_dem['nombre_operador'].dropna().astype(str).unique())), placeholder="Todas", key="bm_empresa")
    with col2: bm_sector = st.multiselect("Sector consumo", sorted(set(df['sector_consumo'].dropna().astype(str).unique()) & set(df_dem['sector_consumo'].dropna().astype(str).unique())), placeholder="Todos", key="bm_sector")
    with col3: bm_tipo = st.multiselect("Tipo demanda", sorted(df_dem['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="bm_tipo")
    with col4: bm_modalidad = st.multiselect("Modalidad (contratos)", sorted(df['modalidad'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="bm_modalidad")
    col5, col6, col7 = st.columns(3)
    with col5: bm_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(), min_value=max(df['fecha_dia'].min().date(), df_dem['fecha_registro'].min().date()), max_value=min(df['fecha_dia'].max().date(), df_dem['fecha_registro'].max().date()), key="bm_fi")
    with col6: bm_fin = st.date_input("Fecha fin", value=min(df['fecha_dia'].max().date(), df_dem['fecha_registro'].max().date()), min_value=max(df['fecha_dia'].min().date(), df_dem['fecha_registro'].min().date()), max_value=min(df['fecha_dia'].max().date(), df_dem['fecha_registro'].max().date()), key="bm_ff")
    with col7: bm_gran = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="bm_gran")
    dff_cont = df.copy()
    dff_cont = dff_cont[(dff_cont['fecha_dia'].dt.date >= bm_inicio) & (dff_cont['fecha_dia'].dt.date <= bm_fin)]
    if bm_empresa: dff_cont = dff_cont[dff_cont['nombre_comprador'].isin(bm_empresa)]
    if bm_sector: dff_cont = dff_cont[dff_cont['sector_consumo'].isin(bm_sector)]
    if bm_modalidad: dff_cont = dff_cont[dff_cont['modalidad'].isin(bm_modalidad)]
    if bm_tipo: dff_cont = dff_cont[dff_cont['tipo_demanda'].isin(bm_tipo)]
    dff_dem = df_dem.copy()
    dff_dem = dff_dem[(dff_dem['fecha_registro'].dt.date >= bm_inicio) & (dff_dem['fecha_registro'].dt.date <= bm_fin)]
    if bm_empresa: dff_dem = dff_dem[dff_dem['nombre_operador'].isin(bm_empresa)]
    if bm_sector: dff_dem = dff_dem[dff_dem['sector_consumo'].isin(bm_sector)]
    if bm_tipo: dff_dem = dff_dem[dff_dem['tipo_demanda'].isin(bm_tipo)]
    balance = agrupar_balance(dff_cont, dff_dem, bm_gran)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Promedio GBTUD Contratado", fmt(balance['gbtud_cont'].mean(),1))
    k2.metric("Promedio GBTUD Demanda", fmt(balance['gbtud_dem'].mean(),1))
    k3.metric("Diferencia promedio", fmt(balance['diferencia'].mean(),1))
    k4.metric("% Períodos sobrecontratados", fmt((balance['diferencia']>0).mean()*100,1)+'%')
    fig_bm = go.Figure()
    fig_bm.add_trace(go.Bar(name='Contratado', x=balance['periodo_str'], y=balance['gbtud_cont'], marker_color='steelblue', text=balance['gbtud_cont'].apply(lambda x: fmt(x,1)), textposition='outside', textangle=-90, textfont=dict(size=14)))
    fig_bm.add_trace(go.Bar(name='Demanda', x=balance['periodo_str'], y=balance['gbtud_dem'], marker_color='goldenrod', text=balance['gbtud_dem'].apply(lambda x: fmt(x,1)), textposition='outside', textangle=-90, textfont=dict(size=14)))
    fig_bm.update_layout(barmode='group', title='Contratación vs Demanda (GBTUD)', xaxis_title='Período', yaxis_title='GBTUD', height=450, legend=dict(orientation='v', x=1.02))
    st.plotly_chart(fig_bm, use_container_width=True)
    balance['label_dif'] = balance.apply(lambda r: fmt(r['diferencia'],1)+' ('+fmt(r['pct_sobre_demanda'],1)+'%)', axis=1)
    fig_dif = go.Figure()
    fig_dif.add_trace(go.Bar(name='Diferencia (Cont - Dem)', x=balance['periodo_str'], y=balance['diferencia'], marker_color='darkorange', text=balance['label_dif'], textposition='outside', textangle=-90, textfont=dict(size=14)))
    fig_dif.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
    fig_dif.update_layout(title='Diferencia Contratación − Demanda (GBTUD) y % sobre Demanda', xaxis_title='Período', yaxis_title='GBTUD', height=400, legend=dict(orientation='v', x=1.02))
    st.plotly_chart(fig_dif, use_container_width=True)
    st.subheader("Detalle por período")
    tabla_bm = balance.copy()
    tabla_bm['gbtud_cont'] = tabla_bm['gbtud_cont'].apply(lambda x: fmt(x,1))
    tabla_bm['gbtud_dem'] = tabla_bm['gbtud_dem'].apply(lambda x: fmt(x,1))
    tabla_bm['diferencia'] = tabla_bm['diferencia'].apply(lambda x: fmt(x,1))
    tabla_bm['pct_sobre_demanda'] = tabla_bm['pct_sobre_demanda'].apply(lambda x: fmt(x,1)+'%' if not pd.isna(x) else 'N/D')
    tabla_bm = tabla_bm[['periodo_str','gbtud_cont','gbtud_dem','diferencia','pct_sobre_demanda']]
    tabla_bm.columns = ['Período','Contratado (GBTUD)','Demanda (GBTUD)','Diferencia (GBTUD)','% sobre Demanda']
    st.dataframe(tabla_bm, use_container_width=True)
    st.markdown("---")
    st.subheader("Sobrecontratación por Empresa")
    grp_cont_emp = dff_cont.groupby(['nombre_comprador','tipo_demanda'], observed=True)['cantidad'].sum().reset_index()
    dias_prom = (pd.to_datetime(bm_fin) - pd.to_datetime(bm_inicio)).days + 1
    grp_cont_emp['gbtud'] = grp_cont_emp['cantidad'] / (dias_prom*1000)
    grp_cont_emp = grp_cont_emp.rename(columns={'nombre_comprador':'empresa','gbtud':'gbtud_cont'})
    grp_dem_emp = dff_dem.groupby(['nombre_operador','tipo_demanda'], observed=True)['cantidad_entregada'].sum().reset_index()
    grp_dem_emp['gbtud'] = grp_dem_emp['cantidad_entregada'] / (dias_prom*1000)
    grp_dem_emp = grp_dem_emp.rename(columns={'nombre_operador':'empresa','gbtud':'gbtud_dem'})
    sobrecont = grp_cont_emp[['empresa','tipo_demanda','gbtud_cont']].merge(grp_dem_emp[['empresa','tipo_demanda','gbtud_dem']], on=['empresa','tipo_demanda'], how='outer').fillna(0)
    sobrecont['diferencia'] = sobrecont['gbtud_cont'] - sobrecont['gbtud_dem']
    sobrecont['pct'] = (sobrecont['diferencia'] / sobrecont['gbtud_dem'].replace(0,float('nan')) * 100)
    emp_totals = sobrecont.groupby('empresa')[['gbtud_cont','gbtud_dem','diferencia']].sum().reset_index()
    emp_totals['pct'] = (emp_totals['diferencia'] / emp_totals['gbtud_dem'].replace(0,float('nan')) * 100)
    emp_totals = emp_totals[emp_totals['gbtud_dem']>0].sort_values('diferencia', ascending=False)
    filas = []
    for _, emp_row in emp_totals.iterrows():
        emp = emp_row['empresa']
        filas.append({'Empresa / Tipo Demanda': f'▶ {emp}', 'Contratado (GBTUD)': fmt(emp_row['gbtud_cont'],1), 'Demanda (GBTUD)': fmt(emp_row['gbtud_dem'],1), 'Diferencia (GBTUD)': fmt(emp_row['diferencia'],1), '% sobre Demanda': fmt(emp_row['pct'],1)+'%' if not pd.isna(emp_row['pct']) else 'N/D'})
        tipos = sobrecont[(sobrecont['empresa']==emp) & (sobrecont['gbtud_dem']>0)].sort_values('diferencia', ascending=False)
        for _, tipo_row in tipos.iterrows():
            filas.append({'Empresa / Tipo Demanda': f'   → {tipo_row["tipo_demanda"]}', 'Contratado (GBTUD)': fmt(tipo_row['gbtud_cont'],1), 'Demanda (GBTUD)': fmt(tipo_row['gbtud_dem'],1), 'Diferencia (GBTUD)': fmt(tipo_row['diferencia'],1), '% sobre Demanda': fmt(tipo_row['pct'],1)+'%' if not pd.isna(tipo_row['pct']) else 'N/D'})
    st.dataframe(pd.DataFrame(filas), use_container_width=True, height=500)

# ── Producción ────────────────────────────────────────────────────────────────
elif seccion == "🔋 Producción":
    st.title("🔋 Producción de Gas Natural")
    tab_p1, tab_p2 = st.tabs(["📊 Producción","🔍 Análisis de Producción"])

    @st.cache_data
    def orden_operadores_produccion():
        df_p = cargar_produccion()
        return df_p.groupby('operador', observed=True)['energia_mbtu'].sum().sort_values(ascending=False).index.astype(str).tolist()

    orden_ops = orden_operadores_produccion()
    df_prod = cargar_produccion()

    with tab_p1:
        st.subheader("Producción de Gas Natural")
        col1, col2, col3 = st.columns(3)
        with col1: p_operador = st.multiselect("Operador", sorted(df_prod['operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="p_operador")
        with col2: p_fuente = st.multiselect("Fuente", sorted(df_prod['fuente'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="p_fuente")
        with col3: p_tipo = st.multiselect("Tipo de producción", sorted(df_prod['tipo_produccion'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="p_tipo")
        col4, col5, col6 = st.columns(3)
        with col4: p_inicio = st.date_input("Fecha inicio", value=df_prod['fecha'].min().date(), min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="p_fi")
        with col5: p_fin = st.date_input("Fecha fin", value=df_prod['fecha'].max().date(), min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="p_ff")
        with col6: p_gran = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="p_gran")
        dfp = df_prod.copy()
        dfp = dfp[(dfp['fecha'].dt.date >= p_inicio) & (dfp['fecha'].dt.date <= p_fin)]
        if p_operador: dfp = dfp[dfp['operador'].isin(p_operador)]
        if p_fuente: dfp = dfp[dfp['fuente'].isin(p_fuente)]
        if p_tipo: dfp = dfp[dfp['tipo_produccion'].isin(p_tipo)]
        k1p, k2p, k3p = st.columns(3)
        k1p.metric("Total GBTUD", fmt(dfp['energia_mbtu'].sum()/(dfp['fecha'].dt.days_in_month.mean()*1000),1))
        k2p.metric("N° Operadores", fmt(dfp['operador'].nunique(),0))
        k3p.metric("N° Fuentes", fmt(dfp['fuente'].nunique(),0))
        dfp_grp = dfp.copy()
        if p_gran == "Diario":
            dfp_grp['periodo'] = dfp_grp['fecha'].dt.to_period('D')
            dias_per = dfp_grp.groupby('periodo', observed=True)['fecha'].first().reset_index()
            dias_per['dias_calendario'] = 1
            dias_per = dias_per[['periodo','dias_calendario']]
        elif p_gran == "Mensual":
            dfp_grp['periodo'] = dfp_grp['fecha'].dt.to_period('M')
            dias_per = dfp_grp.groupby('periodo', observed=True)['fecha'].first().dt.days_in_month.reset_index()
            dias_per.columns = ['periodo','dias_calendario']
        elif p_gran in ["Trimestral","Anual"]:
            dfp_grp['periodo'] = dfp_grp['fecha'].dt.to_period('Q' if p_gran=="Trimestral" else 'Y')
            temp = dfp_grp[['periodo','fecha']].drop_duplicates('fecha').copy()
            temp['mes'] = temp['fecha'].dt.to_period('M')
            temp['dias_mes'] = temp['fecha'].dt.days_in_month
            dias_per = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
            dias_per.columns = ['periodo','dias_calendario']
        grp_p = dfp_grp.groupby(['periodo','operador'], observed=True)['energia_mbtu'].sum().reset_index()
        grp_p = grp_p.merge(dias_per, on='periodo')
        grp_p['gbtud'] = grp_p['energia_mbtu'] / (grp_p['dias_calendario']*1000)
        grp_p['periodo_str'] = grp_p['periodo'].astype(str)
        total_per = grp_p.groupby('periodo')['gbtud'].sum().reset_index(name='gbtud_total')
        total_per = total_per.sort_values('periodo')
        total_per['var_pct'] = total_per['gbtud_total'].pct_change() * 100
        total_per['periodo_str'] = total_per['periodo'].astype(str)
        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        ops_en_grp = [o for o in orden_ops if o in grp_p['operador'].astype(str).unique()]
        for op in ops_en_grp:
            data_op = grp_p[grp_p['operador'].astype(str)==op]
            fig_p.add_trace(go.Bar(name=op, x=data_op['periodo_str'], y=data_op['gbtud']), secondary_y=False)
        fig_p.add_trace(go.Scatter(name='Var% período anterior', x=total_per['periodo_str'], y=total_per['var_pct'],
            mode='lines+markers', line=dict(color='red', width=2), hovertemplate='%{x}<br>Var%: %{y:.1f}%<extra></extra>'), secondary_y=True)
        for _, row in total_per.iterrows():
            fig_p.add_annotation(x=row['periodo_str'], y=row['gbtud_total'], text=fmt(row['gbtud_total'],1),
                showarrow=False, textangle=-90, font=dict(size=14, color='black'), yshift=18)
        fig_p.update_layout(barmode='stack', title='Producción (GBTUD) por Operador', xaxis_title='Período', height=450, legend=dict(orientation='v', x=1.08))
        fig_p.update_yaxes(title_text="GBTUD", secondary_y=False)
        fig_p.update_yaxes(title_text="Variación % período anterior", secondary_y=True)
        fig_p.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, secondary_y=True)
        st.plotly_chart(fig_p, use_container_width=True)
        st.subheader("Detalle de producción")
        tabla_p = dfp[['fecha','operador','fuente','tipo_produccion','energia_mbtu']].copy()
        tabla_p['energia_mbtu'] = tabla_p['energia_mbtu'].apply(lambda x: fmt(x,0))
        tabla_p.columns = ['Fecha','Operador','Fuente','Tipo Producción','Energía (MBTU)']
        st.dataframe(tabla_p, use_container_width=True)

    with tab_p2:
        st.subheader("Análisis de Producción")
        col1, col2, col3 = st.columns(3)
        with col1: pa_operador = st.multiselect("Operador", sorted(df_prod['operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="pa_operador")
        with col2: pa_fuente = st.multiselect("Fuente", sorted(df_prod['fuente'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="pa_fuente")
        with col3: pa_tipo = st.multiselect("Tipo de producción", sorted(df_prod['tipo_produccion'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="pa_tipo")
        col4, col5, _ = st.columns(3)
        with col4: pa_inicio = st.date_input("Fecha inicio", value=df_prod['fecha'].min().date(), min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="pa_fi")
        with col5: pa_fin = st.date_input("Fecha fin", value=df_prod['fecha'].max().date(), min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="pa_ff")
        dfpa = df_prod.copy()
        dfpa = dfpa[(dfpa['fecha'].dt.date >= pa_inicio) & (dfpa['fecha'].dt.date <= pa_fin)]
        if pa_operador: dfpa = dfpa[dfpa['operador'].isin(pa_operador)]
        if pa_fuente: dfpa = dfpa[dfpa['fuente'].isin(pa_fuente)]
        if pa_tipo: dfpa = dfpa[dfpa['tipo_produccion'].isin(pa_tipo)]
        CANACOL = ['CNE OIL & GAS SAS','CNEOG COLOMBIA']
        dfpa_canacol = dfpa.copy()
        dfpa_canacol['operador'] = dfpa_canacol['operador'].astype(str).apply(lambda x: 'Canacol Energy' if x in CANACOL else x)
        col1, col2 = st.columns(2)
        with col1: st.plotly_chart(pie_chart(dfpa_canacol, 'operador', 'Producción por Operador (Canacol agrupado)', 'energia_mbtu'), use_container_width=True)
        with col2: st.plotly_chart(pie_chart(dfpa, 'fuente', 'Producción por Fuente/Campo', 'energia_mbtu'), use_container_width=True)
        st.plotly_chart(bar_chart_top(dfpa, 'fuente', 'Top 10 Fuentes por Producción', 'energia_mbtu'), use_container_width=True)

# ── Declaración de Producción ─────────────────────────────────────────────────
elif seccion == "⚡ Declaración de Producción":
    st.title("⚡ Declaración de Producción")
    tab_dp1, tab_dp2 = st.tabs(["📈 Potencial de Producción","⚖️ PP vs Contratación vs PTDV"])

    with tab_dp1:
        st.subheader("Potencial de Producción (GBTUD) según Declaratoria")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1: dp_declaratoria = st.multiselect("Declaratoria", sorted(df_pot['periodo'].dropna().astype(str).unique().tolist()), default=['2026-2035'], key="dp_declaratoria")
        with col2: dp_campo = st.multiselect("Campo", sorted(df_pot['campo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="dp_campo")
        with col3: dp_operador = st.multiselect("Operador", sorted(df_pot['razon_social'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="dp_operador")
        with col4: dp_variable = st.selectbox("Variable", ["PP","PTDV","PC (todas)"], key="dp_variable")
        with col5:
            anios_disponibles = sorted(df_pot['mes'].dt.year.unique().tolist())
            dp_anios = st.multiselect("Año", anios_disponibles, default=[a for a in anios_disponibles if a >= 2026], key="dp_anios")
        with col6: dp_gran = st.selectbox("Ver por", ["Mensual","Trimestral","Anual"], key="dp_gran")

        dfp = df_pot.copy()
        if dp_declaratoria: dfp = dfp[dfp['periodo'].astype(str).isin(dp_declaratoria)]
        if dp_campo: dfp = dfp[dfp['campo'].astype(str).isin(dp_campo)]
        if dp_operador: dfp = dfp[dfp['razon_social'].astype(str).isin(dp_operador)]
        if dp_anios: dfp = dfp[dfp['mes'].dt.year.isin(dp_anios)]

        if dp_variable == "PP": col_var = 'pp'
        elif dp_variable == "PTDV": col_var = 'ptdv'
        else: col_var = None

        if col_var:
            grp_mes = dfp.groupby(['mes','periodo'], observed=True)[col_var].sum().reset_index()
            grp_mes['gbtud'] = grp_mes[col_var] / 1000
        else:
            dfp['pc_total'] = dfp['pc_consumo_interno'] + dfp['pc_exportaciones'] + dfp['pc_refineria_barranca'] + dfp['pc_refineria_cartagena']
            grp_mes = dfp.groupby(['mes','periodo'], observed=True)['pc_total'].sum().reset_index()
            grp_mes['gbtud'] = grp_mes['pc_total'] / 1000
        grp_mes['mes_str'] = grp_mes['mes'].dt.strftime('%Y-%m')
        grp_mes['anio'] = grp_mes['mes'].dt.year
        grp_mes['trimestre'] = grp_mes['mes'].dt.to_period('Q').astype(str)

        if dp_gran == "Anual":
            grp_plot = grp_mes.groupby(['anio','periodo'], observed=True)['gbtud'].mean().reset_index()
            grp_plot['eje'] = grp_plot['anio'].astype(str)
        elif dp_gran == "Trimestral":
            grp_plot = grp_mes.groupby(['trimestre','periodo'], observed=True)['gbtud'].mean().reset_index()
            grp_plot['eje'] = grp_plot['trimestre']
        else:
            grp_plot = grp_mes.copy()
            grp_plot['eje'] = grp_plot['mes_str']

        fig_plot = go.Figure()
        for per in sorted(grp_plot['periodo'].astype(str).unique()):
            data_per = grp_plot[grp_plot['periodo'].astype(str)==per]
            fig_plot.add_trace(go.Scatter(name=per, x=data_per['eje'], y=data_per['gbtud'], mode='lines+markers', fill='tozeroy'))
        fig_plot.update_layout(title=f'Potencial de Producción (GBTUD) — {dp_variable} — {dp_gran}', xaxis_title=dp_gran, yaxis_title='GBTUD', height=450, legend=dict(orientation='v', x=1.02))
        st.plotly_chart(fig_plot, use_container_width=True)

        if len(dp_declaratoria) >= 2:
            st.markdown("---")
            st.subheader("Variaciones entre Declaratorias")
            tabla_var = grp_mes.pivot_table(index='mes_str', columns='periodo', values='gbtud').reset_index()
            tabla_var.columns.name = None
            decl_cols = sorted([c for c in tabla_var.columns if c != 'mes_str'])
            for i in range(len(decl_cols)-1):
                d1 = decl_cols[i]; d2 = decl_cols[i+1]
                tabla_var[f'Var% {d1}→{d2}'] = ((tabla_var[d2]-tabla_var[d1])/tabla_var[d1].replace(0,float('nan'))*100)
            tabla_fmt = tabla_var.copy()
            tabla_fmt = tabla_fmt.rename(columns={'mes_str':'Mes'})
            for col in decl_cols: tabla_fmt[col] = tabla_fmt[col].apply(lambda x: fmt(x,1))
            for col in tabla_fmt.columns:
                if 'Var%' in col: tabla_fmt[col] = tabla_fmt[col].apply(lambda x: fmt(x,1)+'%' if not pd.isna(x) else 'N/D')
            st.dataframe(tabla_fmt, use_container_width=True, height=400)

            st.markdown("---")
            st.subheader(f"Ranking por Operador y Campo — {dp_variable}")
            if col_var:
                grp_rank = dfp.groupby(['razon_social','campo','periodo'], observed=True)[col_var].sum().reset_index()
                grp_rank['gbtud'] = grp_rank[col_var] / 1000
            else:
                dfp['pc_total'] = dfp['pc_consumo_interno'] + dfp['pc_exportaciones'] + dfp['pc_refineria_barranca'] + dfp['pc_refineria_cartagena']
                grp_rank = dfp.groupby(['razon_social','campo','periodo'], observed=True)['pc_total'].sum().reset_index()
                grp_rank['gbtud'] = grp_rank['pc_total'] / 1000
            pivot_rank = grp_rank.pivot_table(index=['razon_social','campo'], columns='periodo', values='gbtud', fill_value=0).reset_index()
            pivot_rank.columns.name = None
            decl_cols_rank = sorted([c for c in pivot_rank.columns if c not in ['razon_social','campo']])
            for i in range(len(decl_cols_rank)-1):
                d1 = decl_cols_rank[i]; d2 = decl_cols_rank[i+1]
                pivot_rank[f'Var% {d1}→{d2}'] = ((pivot_rank[d2]-pivot_rank[d1])/pivot_rank[d1].replace(0,float('nan'))*100)
            op_totals_rank = pivot_rank.groupby('razon_social')[decl_cols_rank].sum().reset_index()
            for i in range(len(decl_cols_rank)-1):
                d1 = decl_cols_rank[i]; d2 = decl_cols_rank[i+1]
                op_totals_rank[f'Var% {d1}→{d2}'] = ((op_totals_rank[d2]-op_totals_rank[d1])/op_totals_rank[d1].replace(0,float('nan'))*100)
            ultima_decl = decl_cols_rank[-1]
            op_totals_rank = op_totals_rank.sort_values(ultima_decl, ascending=False)
            var_cols_rank = [c for c in pivot_rank.columns if 'Var%' in str(c)]
            filas_rank = []
            for _, op_row in op_totals_rank.iterrows():
                op = op_row['razon_social']
                fila = {'Operador / Campo': f'▶ {op}'}
                for col in decl_cols_rank: fila[col] = fmt(op_row[col],1)
                for col in var_cols_rank: fila[col] = fmt(op_row[col],1)+'%' if not pd.isna(op_row[col]) else 'N/D'
                filas_rank.append(fila)
                campos_op = pivot_rank[pivot_rank['razon_social']==op].sort_values(ultima_decl, ascending=False)
                for _, campo_row in campos_op.iterrows():
                    fila_c = {'Operador / Campo': f'   → {campo_row["campo"]}'}
                    for col in decl_cols_rank: fila_c[col] = fmt(campo_row[col],1)
                    for col in var_cols_rank: fila_c[col] = fmt(campo_row[col],1)+'%' if not pd.isna(campo_row[col]) else 'N/D'
                    filas_rank.append(fila_c)
            st.dataframe(pd.DataFrame(filas_rank), use_container_width=True, height=500)
        else:
            st.info("Selecciona al menos 2 declaratorias para ver las variaciones y el ranking.")

    with tab_dp2:
        st.subheader("PP, Producción Contratada y PTDV (GBTUD)")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: dp2_declaratoria = st.multiselect("Declaratoria", sorted(df_pot['periodo'].dropna().astype(str).unique().tolist()), default=['2026-2035'], key="dp2_declaratoria")
        with col2: dp2_campo = st.multiselect("Campo", sorted(df_pot['campo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="dp2_campo")
        with col3: dp2_operador = st.multiselect("Operador", sorted(df_pot['razon_social'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="dp2_operador")
        with col4:
            anios_disponibles2 = sorted(df_pot['mes'].dt.year.unique().tolist())
            dp2_anios = st.multiselect("Año", anios_disponibles2, default=[a for a in anios_disponibles2 if a >= 2026], key="dp2_anios")
        with col5: dp2_gran = st.selectbox("Ver por", ["Mensual","Trimestral","Anual"], key="dp2_gran")

        dfp2 = df_pot.copy()
        if dp2_declaratoria: dfp2 = dfp2[dfp2['periodo'].astype(str).isin(dp2_declaratoria)]
        if dp2_campo: dfp2 = dfp2[dfp2['campo'].astype(str).isin(dp2_campo)]
        if dp2_operador: dfp2 = dfp2[dfp2['razon_social'].astype(str).isin(dp2_operador)]
        if dp2_anios: dfp2 = dfp2[dfp2['mes'].dt.year.isin(dp2_anios)]

        grp2 = dfp2.groupby('mes')[['pp','pc_consumo_interno','ptdv']].sum().reset_index()
        grp2['gbtud_pp'] = grp2['pp'] / 1000
        grp2['gbtud_pc'] = grp2['pc_consumo_interno'] / 1000
        grp2['gbtud_ptdv'] = grp2['ptdv'] / 1000
        grp2['mes_str'] = grp2['mes'].dt.strftime('%Y-%m')
        grp2['trimestre'] = grp2['mes'].dt.to_period('Q').astype(str)
        grp2['anio'] = grp2['mes'].dt.year

        if dp2_gran == "Anual":
            grp2 = grp2.groupby('anio')[['gbtud_pp','gbtud_pc','gbtud_ptdv']].mean().reset_index()
            eje_x = grp2['anio'].astype(str); x_title = 'Año'
        elif dp2_gran == "Trimestral":
            grp2 = grp2.groupby('trimestre')[['gbtud_pp','gbtud_pc','gbtud_ptdv']].mean().reset_index()
            eje_x = grp2['trimestre']; x_title = 'Trimestre'
        else:
            eje_x = grp2['mes_str']; x_title = 'Mes'

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(name='Potencial de Producción (PP)', x=eje_x, y=grp2['gbtud_pp'], mode='lines', fill='tozeroy', line=dict(color='steelblue'), fillcolor='rgba(70,130,180,0.3)'))
        fig2.add_trace(go.Scatter(name='PC - Consumo Interno', x=eje_x, y=grp2['gbtud_pc'], mode='lines', fill='tozeroy', line=dict(color='green'), fillcolor='rgba(0,128,0,0.3)'))
        fig2.add_trace(go.Scatter(name='PTDV', x=eje_x, y=grp2['gbtud_ptdv'], mode='lines', fill='tozeroy', line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'))
        fig2.update_layout(title='Potencial de Producción, Producción Contratada y PTDV (GBTUD)', xaxis_title=x_title, yaxis_title='GBTUD', height=500, legend=dict(orientation='h', y=1.08))

# ── Nominaciones ──────────────────────────────────────────────────────────────
elif seccion == "📋 Nominaciones":
    st.title("📋 Nominaciones — Programación Definitiva de Suministro")
    tab_n1, tab_n2, tab_n3 = st.tabs(["📊 Resumen", "⚖️ Vs Contratación", "🔎 Cruce por Operación"])

    # ── Tab 1: Resumen ────────────────────────────────────────────────────────
    with tab_n1:
        st.subheader("Resumen de Nominaciones")
        n1_vendedor, n1_comprador, n1_sector, n1_tipo, n1_destino = construir_filtros_nominaciones(df_nom, "n1")
        col_fi, col_ff, col_gran = st.columns(3)
        with col_fi:
            n1_inicio = st.date_input("Fecha inicio",
                value=(pd.Timestamp.today() - pd.Timedelta(days=30)).date(),
                min_value=df_nom['fecha_gas'].min().date(),
                max_value=df_nom['fecha_gas'].max().date(),
                key="n1_fi")
        with col_ff:
            n1_fin = st.date_input("Fecha fin",
                value=df_nom['fecha_gas'].max().date(),
                min_value=df_nom['fecha_gas'].min().date(),
                max_value=df_nom['fecha_gas'].max().date(),
                key="n1_ff")
        with col_gran:
            n1_gran = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="n1_gran")

        dfn1 = df_nom.copy()
        dfn1 = dfn1[(dfn1['fecha_gas'].dt.date >= n1_inicio) & (dfn1['fecha_gas'].dt.date <= n1_fin)]
        dfn1 = aplicar_filtros_nominaciones(dfn1, n1_vendedor, n1_comprador, n1_sector, n1_tipo, n1_destino)

        if dfn1.empty:
            st.warning("No hay datos para los filtros seleccionados.")
        else:
            n_dias_n1 = (n1_fin - n1_inicio).days + 1

            gbtud_prom = dfn1.groupby('fecha_gas')['cantidad_mbtud'].sum().mean() / 1000
            n_vendedores = dfn1['nombre_vendedor'].nunique()
            n_compradores = dfn1['nombre_comprador'].nunique()
            n_dias = dfn1['fecha_gas'].nunique()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("GBTUD promedio/día", fmt(gbtud_prom, 1))
            k2.metric("N° Vendedores", fmt(n_vendedores, 0))
            k3.metric("N° Compradores", fmt(n_compradores, 0))
            k4.metric("Días con nominación", fmt(n_dias, 0))

            grp_n1 = agrupar_nominaciones_sector(dfn1, n1_gran)
            st.plotly_chart(
                construir_grafico_nom_sector(grp_n1, 'Nominación (GBTUD/día) por Sector de Consumo'),
                use_container_width=True
            )

            dfn1_gbtud = dfn1.copy()
            dfn1_gbtud['gbtud'] = dfn1_gbtud['cantidad_mbtud'] / 1000

            st.markdown("---")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.plotly_chart(
                    pie_chart(dfn1_gbtud, 'sector_consumo', 'Distribución por Sector de Consumo', 'gbtud'),
                    use_container_width=True
                )
            with col_p2:
                st.plotly_chart(
                    pie_chart(dfn1_gbtud, 'destino', 'Distribución por Destino', 'gbtud'),
                    use_container_width=True
                )

            st.markdown("---")
            st.subheader("Ranking de Vendedores")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.plotly_chart(
                    bar_chart_top_gbtud(dfn1, 'nombre_vendedor', 'Top 10 Vendedores por GBTUD/día', n_dias_n1),
                    use_container_width=True
                )
            with col_v2:
                st.plotly_chart(
                    pie_chart(dfn1_gbtud, 'nombre_vendedor', 'Participación por Vendedor', 'gbtud'),
                    use_container_width=True
                )

            st.markdown("---")
            st.subheader("Ranking de Compradores")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.plotly_chart(
                    bar_chart_top_gbtud(dfn1, 'nombre_comprador', 'Top 10 Compradores por GBTUD/día', n_dias_n1),
                    use_container_width=True
                )
            with col_c2:
                st.plotly_chart(
                    pie_chart(dfn1_gbtud, 'nombre_comprador', 'Participación por Comprador', 'gbtud'),
                    use_container_width=True
                )
            st.markdown("---")
            st.subheader("Detalle de Nominación (MBTUD)")
            col_agr1, col_agr2 = st.columns(2)
            with col_agr1:
                n1_agrupar_por = st.radio("Agrupar por", ["Comprador","Vendedor"], key="n1_agrupar_por", horizontal=True)
            with col_agr2:
                n1_detalle_por = st.radio("Desglosar por", ["Tipo demanda","Sector consumo"], key="n1_detalle_por", horizontal=True)

            col_tabla = 'nombre_comprador' if n1_agrupar_por == "Comprador" else 'nombre_vendedor'
            col_detalle_tabla = 'tipo_demanda' if n1_detalle_por == "Tipo demanda" else 'sector_consumo'

            if n_dias_n1 > 31:
                st.info("El rango seleccionado supera 31 días — la tabla tendrá muchas columnas. Considera acortar el rango de fechas para una vista más legible.")

            tabla_nom_detalle = construir_tabla_nominacion_detalle(dfn1, col_tabla, col_detalle_tabla)
            if tabla_nom_detalle.empty:
                st.warning("No hay datos para construir la tabla con los filtros seleccionados.")
            else:
                def resaltar_totales(row):
                    es_total = row['Agente / Detalle'].startswith('▶')
                    estilo = 'background-color: #e8eef7; color: #1a1a2e' if es_total else ''
                    return [estilo for _ in row]

                tabla_estilizada = tabla_nom_detalle.style.apply(resaltar_totales, axis=1)
                st.dataframe(tabla_estilizada, use_container_width=True, height=500)
                        # ── Tab 2: Vs Contratación ────────────────────────────────────────────────
    with tab_n2:
        st.subheader("Contratación vs Nominación por Empresa")
        col1, col2 = st.columns(2)
        with col1:
            cvn_rol = st.radio("Rol de la empresa", ["Vendedor","Comprador"], key="cvn_rol", horizontal=True)
        col_empresa = 'nombre_vendedor' if cvn_rol == "Vendedor" else 'nombre_comprador'
        with col2:
            empresas_disponibles = sorted(df_nom[col_empresa].dropna().astype(str).unique().tolist())
            cvn_empresas = st.multiselect("Empresa(s)", empresas_disponibles, placeholder="Selecciona una o varias", key="cvn_empresas")

        col3, col4, col5 = st.columns(3)
        with col3:
            sectores_disp = sorted(set(df['sector_consumo'].dropna().astype(str).unique()) | set(df_nom['sector_consumo'].dropna().astype(str).unique()))
            cvn_sector = st.multiselect("Sector consumo", sectores_disp, placeholder="Todos", key="cvn_sector")
        with col4:
            tipos_disp = sorted(set(df['tipo_demanda'].dropna().astype(str).unique()) | set(df_nom['tipo_demanda'].dropna().astype(str).unique()))
            cvn_tipo = st.multiselect("Tipo demanda", tipos_disp, placeholder="Todas", key="cvn_tipo")
        with col5:
            cvn_modalidad = st.multiselect("Modalidad (contratos)", sorted(df['modalidad'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="cvn_modalidad")

        col6, col7, col8 = st.columns(3)
        fecha_min_cvn = max(df['fecha_dia'].min().date(), df_nom['fecha_gas'].min().date())
        fecha_max_cvn = min(df['fecha_dia'].max().date(), df_nom['fecha_gas'].max().date())
        with col6:
            cvn_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(),
                min_value=fecha_min_cvn, max_value=fecha_max_cvn, key="cvn_fi")
        with col7:
            cvn_fin = st.date_input("Fecha fin", value=fecha_max_cvn,
                min_value=fecha_min_cvn, max_value=fecha_max_cvn, key="cvn_ff")
        with col8:
            cvn_gran = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="cvn_gran")

        if not cvn_empresas:
            st.info("Selecciona al menos una empresa para ver la comparación.")
        else:
            dff_cont_e = df[(df[col_empresa].isin(cvn_empresas)) &
                            (df['fecha_dia'].dt.date >= cvn_inicio) & (df['fecha_dia'].dt.date <= cvn_fin)]
            dff_nom_e = df_nom[(df_nom[col_empresa].isin(cvn_empresas)) &
                               (df_nom['fecha_gas'].dt.date >= cvn_inicio) & (df_nom['fecha_gas'].dt.date <= cvn_fin)]

            if cvn_sector:
                dff_cont_e = dff_cont_e[dff_cont_e['sector_consumo'].isin(cvn_sector)]
                dff_nom_e = dff_nom_e[dff_nom_e['sector_consumo'].isin(cvn_sector)]
            if cvn_tipo:
                dff_cont_e = dff_cont_e[dff_cont_e['tipo_demanda'].isin(cvn_tipo)]
                dff_nom_e = dff_nom_e[dff_nom_e['tipo_demanda'].isin(cvn_tipo)]
            if cvn_modalidad:
                dff_cont_e = dff_cont_e[dff_cont_e['modalidad'].isin(cvn_modalidad)]
                # La modalidad es un atributo del contrato; no existe en nominaciones, por lo que no se filtra df_nom por ella.

            if dff_cont_e.empty and dff_nom_e.empty:
                st.warning("No hay datos de contratación ni nominación para los filtros seleccionados.")
            else:
                grp_cont_e = agrupar_contratos_total(dff_cont_e, cvn_gran) if not dff_cont_e.empty else pd.DataFrame(columns=['periodo_str','gbtud_cont'])
                grp_nom_e = agrupar_nominaciones_total(dff_nom_e, cvn_gran) if not dff_nom_e.empty else pd.DataFrame(columns=['periodo_str','gbtud_nom'])
                comp = grp_cont_e.merge(grp_nom_e, on='periodo_str', how='outer').fillna(0)
                comp = comp.sort_values('periodo_str')
                comp['diferencia'] = comp['gbtud_cont'] - comp['gbtud_nom']
                comp['pct_cumplimiento'] = comp['gbtud_nom'] / comp['gbtud_cont'].replace(0, float('nan')) * 100

                titulo_emp = ', '.join(cvn_empresas) if len(cvn_empresas) <= 3 else f'{len(cvn_empresas)} empresas seleccionadas'

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Promedio Contratado (GBTUD)", fmt(comp['gbtud_cont'].mean(),1))
                k2.metric("Promedio Nominado (GBTUD)", fmt(comp['gbtud_nom'].mean(),1))
                k3.metric("Diferencia promedio (GBTUD)", fmt(comp['diferencia'].mean(),1))
                pct_prom = comp['pct_cumplimiento'].mean()
                k4.metric("% Cumplimiento promedio", fmt(pct_prom,1)+'%' if not pd.isna(pct_prom) else 'N/D')

                fig_cvn = go.Figure()
                fig_cvn.add_trace(go.Bar(name='Contratado', x=comp['periodo_str'], y=comp['gbtud_cont'], marker_color='steelblue',
                    text=comp['gbtud_cont'].apply(lambda x: fmt(x,1)), textposition='outside', textangle=-90, textfont=dict(size=14)))
                fig_cvn.add_trace(go.Bar(name='Nominado', x=comp['periodo_str'], y=comp['gbtud_nom'], marker_color='darkorange',
                    text=comp['gbtud_nom'].apply(lambda x: fmt(x,1)), textposition='outside', textangle=-90, textfont=dict(size=14)))
                fig_cvn.update_layout(barmode='group', title=f'Contratación vs Nominación — {titulo_emp} ({cvn_rol})',
                    xaxis_title='Período', yaxis_title='GBTUD', height=450, legend=dict(orientation='v', x=1.02))
                st.plotly_chart(fig_cvn, use_container_width=True)

                fig_pct = go.Figure()
                fig_pct.add_trace(go.Scatter(name='% Cumplimiento (Nom/Cont)', x=comp['periodo_str'], y=comp['pct_cumplimiento'],
                    mode='lines+markers', line=dict(color='red', width=2)))
                fig_pct.add_hline(y=100, line_dash='dash', line_color='gray', line_width=1)
                fig_pct.update_layout(title='% Cumplimiento de Nominación sobre lo Contratado', xaxis_title='Período', yaxis_title='%', height=350)
                st.plotly_chart(fig_pct, use_container_width=True)

                st.subheader("Detalle por período")
                tabla_cvn = comp.copy()
                tabla_cvn['gbtud_cont'] = tabla_cvn['gbtud_cont'].apply(lambda x: fmt(x,1))
                tabla_cvn['gbtud_nom'] = tabla_cvn['gbtud_nom'].apply(lambda x: fmt(x,1))
                tabla_cvn['diferencia'] = tabla_cvn['diferencia'].apply(lambda x: fmt(x,1))
                tabla_cvn['pct_cumplimiento'] = tabla_cvn['pct_cumplimiento'].apply(lambda x: fmt(x,1)+'%' if not pd.isna(x) else 'N/D')
                tabla_cvn = tabla_cvn[['periodo_str','gbtud_cont','gbtud_nom','diferencia','pct_cumplimiento']]
                tabla_cvn.columns = ['Período','Contratado (GBTUD)','Nominado (GBTUD)','Diferencia (GBTUD)','% Cumplimiento']
                st.dataframe(tabla_cvn, use_container_width=True)
    # ── Tab 3: Cruce por Operación ───────────────────────────────────────────
    with tab_n3:
        st.subheader("Cruce Contratación vs Nominación por N° de Operación SEGAS")

        st.info(
            "**Nota normativa (Resolución CREG 102 015 de 2025):** el Artículo 26, Parágrafo 4 establece que, "
            "con excepción de los contratos **Con Interrupciones**, las obligaciones de las demás modalidades "
            "(Firme, Firmeza Condicionada, Opción de Compra, Contingencia) se consideran **permanentes y por el "
            "100% del gas contratado** durante toda su vigencia — es decir, no requieren un acto de nominación "
            "diaria para activarse. El Artículo 38, por su parte, condiciona la firmeza diaria de los contratos "
            "**Con Interrupciones** a que ocurra la nominación. Por eso, un contrato Firme sin nominación "
            "cruzada **no es necesariamente un incumplimiento** — puede ser el comportamiento esperado por diseño "
            "regulatorio. Se recomienda interpretar el cruce con esto en mente, especialmente al excluir o no la "
            "modalidad Firme del análisis."
        )

        col1, col2 = st.columns(2)
        with col1:
            cop_rol = st.radio("Filtrar empresa por rol", ["Todas","Vendedor","Comprador"], key="cop_rol", horizontal=True)
        col_empresa_op = 'nombre_vendedor' if cop_rol == "Vendedor" else ('nombre_comprador' if cop_rol == "Comprador" else None)
        with col2:
            if col_empresa_op:
                empresas_op_disp = sorted(set(df[col_empresa_op].dropna().astype(str).unique()) | set(df_nom[col_empresa_op].dropna().astype(str).unique()))
                cop_empresas = st.multiselect("Empresa(s)", empresas_op_disp, placeholder="Todas", key="cop_empresas")
            else:
                cop_empresas = []

        col3, col4, col5 = st.columns(3)
        with col3:
            cop_sector = st.multiselect("Sector consumo", sorted(set(df['sector_consumo'].dropna().astype(str).unique()) | set(df_nom['sector_consumo'].dropna().astype(str).unique())), placeholder="Todos", key="cop_sector")
        with col4:
            cop_tipo = st.multiselect("Tipo demanda", sorted(set(df['tipo_demanda'].dropna().astype(str).unique()) | set(df_nom['tipo_demanda'].dropna().astype(str).unique())), placeholder="Todas", key="cop_tipo")
        with col5:
            cop_mercado = st.multiselect("Mercado (contratos)", sorted(df['mercado'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="cop_mercado")

        col6, col7 = st.columns(2)
        with col6:
            cop_modalidad = st.multiselect("Modalidad (contratos)", sorted(df['modalidad'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="cop_modalidad")
        with col7:
            cop_excluir_firme = st.checkbox("Excluir modalidad 'Firme' (permanente por regulación, ver nota arriba)", value=False, key="cop_excluir_firme")

        col8, col9, col10 = st.columns(3)
        fecha_min_cop = max(df['fecha_dia'].min().date(), df_nom['fecha_gas'].min().date())
        fecha_max_cop = min(df['fecha_dia'].max().date(), df_nom['fecha_gas'].max().date())
        with col8:
            cop_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2025-01-01').date(),
                min_value=fecha_min_cop, max_value=fecha_max_cop, key="cop_fi")
        with col9:
            cop_fin = st.date_input("Fecha fin", value=fecha_max_cop,
                min_value=fecha_min_cop, max_value=fecha_max_cop, key="cop_ff")
        with col10:
            cop_gran = st.selectbox("Ver por", ["Mensual","Diario","Trimestral","Anual"], key="cop_gran")

        dff_cont_op = df[(df['fecha_dia'].dt.date >= cop_inicio) & (df['fecha_dia'].dt.date <= cop_fin)]
        dff_nom_op = df_nom[(df_nom['fecha_gas'].dt.date >= cop_inicio) & (df_nom['fecha_gas'].dt.date <= cop_fin)]

        if col_empresa_op and cop_empresas:
            dff_cont_op = dff_cont_op[dff_cont_op[col_empresa_op].isin(cop_empresas)]
            dff_nom_op = dff_nom_op[dff_nom_op[col_empresa_op].isin(cop_empresas)]
        if cop_sector:
            dff_cont_op = dff_cont_op[dff_cont_op['sector_consumo'].isin(cop_sector)]
            dff_nom_op = dff_nom_op[dff_nom_op['sector_consumo'].isin(cop_sector)]
        if cop_tipo:
            dff_cont_op = dff_cont_op[dff_cont_op['tipo_demanda'].isin(cop_tipo)]
            dff_nom_op = dff_nom_op[dff_nom_op['tipo_demanda'].isin(cop_tipo)]
        if cop_mercado:
            dff_cont_op = dff_cont_op[dff_cont_op['mercado'].isin(cop_mercado)]
            # Mercado no existe en nominaciones; el filtro solo restringe el universo de operaciones contratadas a cruzar.
        if cop_modalidad:
            dff_cont_op = dff_cont_op[dff_cont_op['modalidad'].isin(cop_modalidad)]
        if cop_excluir_firme:
            dff_cont_op = dff_cont_op[dff_cont_op['modalidad'] != 'Firme']

        n_dias_cop = (cop_fin - cop_inicio).days + 1

        if dff_cont_op.empty and dff_nom_op.empty:
            st.warning("No hay datos para los filtros seleccionados.")
        else:
            comp, cobertura = cruzar_matched_periodo(dff_cont_op, dff_nom_op, cop_gran)

            st.markdown("#### Diagnóstico de cobertura")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("N° Operaciones contratadas", fmt(cobertura['n_ops_cont'],0))
            k2.metric("N° Operaciones nominadas", fmt(cobertura['n_ops_nom'],0))
            k3.metric("N° Operaciones que cruzan", fmt(cobertura['n_cruzan'],0))
            pct_cob = cobertura['n_cruzan']/cobertura['n_ops_nom']*100 if cobertura['n_ops_nom']>0 else 0
            k4.metric("% Cobertura (sobre nominadas)", fmt(pct_cob,1)+'%')

            k5, k6 = st.columns(2)
            k5.metric("Operaciones solo nominadas (sin contrato)", fmt(cobertura['n_huerfanas_nom'],0))
            k6.metric("Operaciones solo contratadas (sin nominar)", fmt(cobertura['n_huerfanas_cont'],0))

            if comp.empty or (comp['gbtud_cont'].sum()==0 and comp['gbtud_nom'].sum()==0):
                st.warning("No hay operaciones que crucen (Contratado ∩ Nominado) para graficar con estos filtros.")
            else:
                st.markdown("---")
                st.markdown("#### Contratado vs Nominado — solo operaciones que cruzan")
                titulo_emp = ', '.join(cop_empresas) if (cop_empresas and len(cop_empresas) <= 3) else (f'{len(cop_empresas)} empresas' if cop_empresas else 'Todas las empresas')
                st.plotly_chart(
                    construir_grafico_cruce_periodo(comp, f'Contratado vs Nominado (GBTUD) — {titulo_emp}'),
                    use_container_width=True
                )

                st.markdown("---")
                st.markdown("#### Ranking por Agente")
                col_rank1, col_rank2 = st.columns(2)
                with col_rank1:
                    if not dff_cont_op.empty:
                        rank_vend = ranking_agente_cruce(dff_cont_op, dff_nom_op, 'nombre_vendedor', n_dias_cop)
                        if not rank_vend.empty:
                            st.plotly_chart(construir_grafico_ranking_cruce(rank_vend, 'nombre_vendedor', 'Top Vendedores — Contratado vs Nominado (GBTUD/día)'), use_container_width=True)
                        else:
                            st.info("No hay operaciones cruzadas para rankear vendedores.")
                with col_rank2:
                    if not dff_cont_op.empty:
                        rank_comp = ranking_agente_cruce(dff_cont_op, dff_nom_op, 'nombre_comprador', n_dias_cop)
                        if not rank_comp.empty:
                            st.plotly_chart(construir_grafico_ranking_cruce(rank_comp, 'nombre_comprador', 'Top Compradores — Contratado vs Nominado (GBTUD/día)'), use_container_width=True)
                        else:
                            st.info("No hay operaciones cruzadas para rankear compradores.")

            with st.expander("Ver detalle de operaciones con mayor diferencia (top 30)"):
                comp_ops, _ = cruzar_matched_periodo(dff_cont_op, dff_nom_op, "Anual")  # solo para reutilizar cálculo agregado si se necesita; el detalle real es por operación
                # Detalle real por operación, limitado a top 30 por diferencia absoluta
                grp_cont_det = dff_cont_op.groupby('no_operacion', observed=True).agg(
                    nombre_vendedor=('nombre_vendedor','first'), nombre_comprador=('nombre_comprador','first'),
                    modalidad=('modalidad','first'), cantidad_total=('cantidad','sum')).reset_index()
                grp_cont_det['gbtud_cont'] = grp_cont_det['cantidad_total'] / (n_dias_cop*1000)
                grp_nom_det = dff_nom_op.groupby('no_operacion', observed=True)['cantidad_mbtud'].sum().reset_index()
                grp_nom_det['gbtud_nom'] = grp_nom_det['cantidad_mbtud'] / 1000 / n_dias_cop
                det = grp_cont_det.merge(grp_nom_det[['no_operacion','gbtud_nom']], on='no_operacion', how='inner')
                det['diferencia'] = det['gbtud_cont'] - det['gbtud_nom']
                det = det.sort_values('diferencia', key=abs, ascending=False).head(30)
                det['gbtud_cont'] = det['gbtud_cont'].apply(lambda x: fmt(x,2))
                det['gbtud_nom'] = det['gbtud_nom'].apply(lambda x: fmt(x,2))
                det['diferencia'] = det['diferencia'].apply(lambda x: fmt(x,2))
                det = det[['no_operacion','nombre_vendedor','nombre_comprador','modalidad','gbtud_cont','gbtud_nom','diferencia']]
                det.columns = ['N° Operación','Vendedor','Comprador','Modalidad','Contratado (GBTUD)','Nominado (GBTUD)','Diferencia (GBTUD)']
                st.dataframe(det, use_container_width=True, height=400)