import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import traceback
import sys

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
PARQUET_PATH = BASE_DIR / "data" / "processed" / "contratos_gas.parquet"
DEMANDA_PATH = BASE_DIR / "data" / "processed" / "demanda_gestor.parquet"
PRODUCCION_PATH = BASE_DIR / "data" / "processed" / "produccion_gestor.parquet"
POTENCIAL_PATH = BASE_DIR / "data" / "processed" / "potencial_produccion.parquet"

# ── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Monitor de Gas Natural - Superservicios",
    page_icon="⛽",
    layout="wide"
)

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def cargar_contratos():
    columnas = [
        'fecha_dia', 'cantidad', 'precio', 'sector_consumo',
        'modalidad', 'mercado', 'nombre_vendedor', 'nombre_comprador',
        'fecha_inicial', 'fecha_final', 'no_operacion', 'tipo_demanda'
    ]
    df = pd.read_parquet(PARQUET_PATH, columns=columnas)
    df = df[df['fecha_dia'] >= '2021-01-01'].copy()
    # Reducir memoria al máximo
    df['cantidad'] = df['cantidad'].astype('int32')
    df['precio'] = df['precio'].astype('float32')
    for col in ['sector_consumo', 'modalidad', 'mercado',
                'nombre_vendedor', 'nombre_comprador', 'tipo_demanda']:
        df[col] = df[col].astype('category')
    return df

@st.cache_data
def cargar_demanda():
    df = pd.read_parquet(DEMANDA_PATH)
    return df

@st.cache_data
def cargar_produccion():
    df = pd.read_parquet(PRODUCCION_PATH)
    return df

@st.cache_data
def cargar_potencial():
    df = pd.read_parquet(POTENCIAL_PATH)
    return df

df = cargar_contratos()
df_dem = cargar_demanda()
df_pot = cargar_potencial()

# ── Formato colombiano ────────────────────────────────────────────────────────
def fmt(valor, decimales=1):
    resultado = f"{valor:,.{decimales}f}"
    return resultado.replace(',', 'X').replace('.', ',').replace('X', '.')

# ── Agrupación contratación ───────────────────────────────────────────────────
def agrupar(dff, granularidad):
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_dia'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo', 'dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_dia'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo', 'dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('Q')
        temp = dff[['periodo', 'fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo', 'dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_dia'].dt.to_period('Y')
        temp = dff[['periodo', 'fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo', 'dias_calendario']

    precio_pond = (
        dff.groupby('periodo', observed=True)
        .apply(lambda x: (x['precio'] * x['cantidad']).sum() / x['cantidad'].sum(), include_groups=False)
        .reset_index(name='precio_ponderado')
    )
    grp = dff.groupby(['periodo', 'sector_consumo'], observed=True)['cantidad'].sum().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud'] = grp['cantidad'] / (grp['dias_calendario'] * 1000)
    grp = grp.merge(precio_pond, on='periodo')
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp

# ── Agrupación demanda ────────────────────────────────────────────────────────
def agrupar_demanda(dff, granularidad):
    dff = dff.copy()
    if granularidad == "Diario":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('D')
        dias = dff.groupby('periodo', observed=True)['fecha_registro'].first().reset_index()
        dias['dias_calendario'] = 1
        dias = dias[['periodo', 'dias_calendario']]
    elif granularidad == "Mensual":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('M')
        dias = dff.groupby('periodo', observed=True)['fecha_registro'].first().dt.days_in_month.reset_index()
        dias.columns = ['periodo', 'dias_calendario']
    elif granularidad == "Trimestral":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('Q')
        temp = dff[['periodo', 'fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp['mes'] = temp['fecha_registro'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_registro'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo', 'dias_calendario']
    elif granularidad == "Anual":
        dff['periodo'] = dff['fecha_registro'].dt.to_period('Y')
        temp = dff[['periodo', 'fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp['mes'] = temp['fecha_registro'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_registro'].dt.days_in_month
        dias = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias.columns = ['periodo', 'dias_calendario']

    grp = dff.groupby(['periodo', 'sector_consumo'], observed=True)['cantidad_entregada'].sum().reset_index()
    grp = grp.merge(dias, on='periodo')
    grp['gbtud'] = grp['cantidad_entregada'] / (grp['dias_calendario'] * 1000)
    grp['periodo_str'] = grp['periodo'].astype(str)
    return grp

# ── Filtros contratación ──────────────────────────────────────────────────────
def construir_filtros(df, key_prefix):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        vendedor = st.multiselect("Vendedor", sorted(df['nombre_vendedor'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_vendedor")
    with col2:
        comprador = st.multiselect("Comprador", sorted(df['nombre_comprador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_comprador")
    with col3:
        modalidad = st.multiselect("Modalidad", sorted(df['modalidad'].dropna().astype(str).unique().tolist()), placeholder="Todas", key=f"{key_prefix}_modalidad")
    with col4:
        sector = st.multiselect("Sector consumo", sorted(df['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_sector")
    with col5:
        mercado = st.multiselect("Mercado", sorted(df['mercado'].dropna().astype(str).unique().tolist()), placeholder="Todos", key=f"{key_prefix}_mercado")
    with col6:
        tipo_demanda = st.multiselect("Tipo demanda", sorted(df['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key=f"{key_prefix}_tipo_demanda")
    return vendedor, comprador, modalidad, sector, mercado, tipo_demanda

def aplicar_filtros(dff, vendedor, comprador, modalidad, sector, mercado, tipo_demanda):
    if vendedor:
        dff = dff[dff['nombre_vendedor'].isin(vendedor)]
    if comprador:
        dff = dff[dff['nombre_comprador'].isin(comprador)]
    if modalidad:
        dff = dff[dff['modalidad'].isin(modalidad)]
    if sector:
        dff = dff[dff['sector_consumo'].isin(sector)]
    if mercado:
        dff = dff[dff['mercado'].isin(mercado)]
    if tipo_demanda:
        dff = dff[dff['tipo_demanda'].isin(tipo_demanda)]
    return dff

# ── Gráfico contratación ──────────────────────────────────────────────────────
def construir_grafico(grp, titulo):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for sec in sorted(grp['sector_consumo'].astype(str).unique()):
        data_sec = grp[grp['sector_consumo'].astype(str) == sec]
        fig.add_trace(go.Bar(name=sec, x=data_sec['periodo_str'], y=data_sec['gbtud']), secondary_y=False)
    precio_line = grp.drop_duplicates('periodo_str')[['periodo_str', 'precio_ponderado']]
    fig.add_trace(go.Scatter(name='Precio Ponderado', x=precio_line['periodo_str'],
                             y=precio_line['precio_ponderado'], mode='lines+markers',
                             line=dict(color='purple', width=2)), secondary_y=True)
    totales = grp.groupby('periodo_str')['gbtud'].sum().reset_index()
    for _, row in totales.iterrows():
        fig.add_annotation(x=row['periodo_str'], y=row['gbtud'], text=fmt(row['gbtud'], 1),
                           showarrow=False, textangle=-90, font=dict(size=7, color='black'), yshift=18)
    fig.update_layout(barmode='stack', title=titulo, xaxis_title='Período', height=450, legend=dict(orientation='v', x=1.08))
    fig.update_yaxes(title_text="GBTUD", secondary_y=False)
    fig.update_yaxes(title_text="Precio Ponderado (USD/MBTUD)", secondary_y=True)
    return fig

# ── Gráfico demanda ───────────────────────────────────────────────────────────
def construir_grafico_demanda(grp, titulo):
    fig = go.Figure()
    for sec in sorted(grp['sector_consumo'].astype(str).unique()):
        data_sec = grp[grp['sector_consumo'].astype(str) == sec]
        fig.add_trace(go.Bar(name=sec, x=data_sec['periodo_str'], y=data_sec['gbtud']))
    totales = grp.groupby('periodo_str')['gbtud'].sum().reset_index()
    for _, row in totales.iterrows():
        fig.add_annotation(x=row['periodo_str'], y=row['gbtud'], text=fmt(row['gbtud'], 1),
                           showarrow=False, textangle=-90, font=dict(size=7, color='black'), yshift=18)
    fig.update_layout(barmode='stack', title=titulo, xaxis_title='Período',
                      yaxis_title='GBTUD', height=450, legend=dict(orientation='v', x=1.08))
    return fig

# ── Gráficos análisis ─────────────────────────────────────────────────────────
def pie_chart(data, col, titulo, col_cantidad):
    grp = data.groupby(col, observed=True)[col_cantidad].sum().reset_index()
    grp.columns = [col, 'cantidad']
    grp = grp.sort_values('cantidad', ascending=False)
    total = grp['cantidad'].sum()
    grp['pct'] = grp['cantidad'] / total * 100
    grp['label'] = grp[col].astype(str) + '<br>' + grp['pct'].apply(lambda x: fmt(x, 1)) + '%'
    fig = go.Figure(go.Pie(labels=grp[col].astype(str), values=grp['cantidad'],
                           text=grp['label'], textinfo='text', textposition='inside',
                           insidetextorientation='radial', hole=0.3,
                           hovertemplate='%{label}<br>MBTU: %{value:,.0f}<extra></extra>'))
    fig.update_layout(title=titulo, height=400, showlegend=True,
                      legend=dict(orientation='v', x=1.0, y=0.5),
                      margin=dict(t=50, b=20, l=20, r=120))
    return fig

def bar_chart_top(data, col, titulo, col_cantidad, top_n=10):
    grp = data.groupby(col, observed=True)[col_cantidad].sum().reset_index()
    grp.columns = [col, 'cantidad']
    grp = grp.sort_values('cantidad', ascending=False).head(top_n)
    total = grp['cantidad'].sum()
    grp['pct'] = grp['cantidad'] / total * 100
    grp['label'] = grp['pct'].apply(lambda x: fmt(x, 1)) + '%'
    fig = go.Figure(go.Bar(x=grp['cantidad'], y=grp[col].astype(str), orientation='h',
                           text=grp['label'], textposition='outside', marker_color='steelblue',
                           hovertemplate='%{y}<br>MBTU: %{x:,.0f}<extra></extra>'))
    fig.update_layout(title=titulo, height=350, yaxis=dict(autorange='reversed'),
                      margin=dict(t=50, b=20, l=20, r=20), xaxis_title='MBTU')
    return fig

# ── Estacionalidad ────────────────────────────────────────────────────────────
def construir_estacionalidad(dff):
    dff = dff.copy()
    dff['mes'] = dff['fecha_registro'].dt.month
    dff['dias_mes'] = dff['fecha_registro'].dt.days_in_month
    grp = dff.groupby(['mes', 'dias_mes'])['cantidad_entregada'].sum().reset_index()
    anios = dff['fecha_registro'].dt.year.nunique()
    grp['gbtud'] = grp['cantidad_entregada'] / (grp['dias_mes'] * 1000 * anios)
    meses_nombre = {1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
                    7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'}
    grp['mes_str'] = grp['mes'].map(meses_nombre)
    fig = go.Figure(go.Bar(x=grp['mes_str'], y=grp['gbtud'],
                           text=grp['gbtud'].apply(lambda x: fmt(x, 1)),
                           textposition='outside', marker_color='steelblue'))
    fig.update_layout(title='Estacionalidad — Promedio histórico por mes (GBTUD)',
                      xaxis_title='', yaxis_title='GBTUD', height=400)
    return fig

# ── Tabla resumen con variaciones y PP ───────────────────────────────────────
def construir_tabla_resumen(dff):
    anios = sorted(dff['fecha_registro'].dt.year.unique())
    if len(anios) < 2:
        st.warning("No hay suficientes años para calcular variaciones.")
        return
    anio_t = anios[-1]
    anio_t1 = anios[-2]

    dff = dff.copy()
    dff['anio'] = dff['fecha_registro'].dt.year
    grp = dff.groupby(['nombre_operador', 'sector_consumo', 'anio'], observed=True)['cantidad_entregada'].sum().reset_index()
    pivot = grp.pivot_table(index=['nombre_operador', 'sector_consumo'],
                            columns='anio', values='cantidad_entregada', fill_value=0).reset_index()
    pivot.columns.name = None
    for a in anios:
        if a not in pivot.columns:
            pivot[a] = 0

    op_totals = pivot.groupby('nombre_operador')[[a for a in anios]].sum().reset_index()
    op_totals['sector_consumo'] = '— TOTAL —'
    nacional = {a: pivot[a].sum() for a in anios}

    pivot['var_pct'] = (pivot[anio_t] - pivot[anio_t1]) / pivot[anio_t1].replace(0, float('nan')) * 100
    op_t1 = op_totals[['nombre_operador', anio_t1]].rename(columns={anio_t1: 'op_t1'})
    pivot = pivot.merge(op_t1, on='nombre_operador')
    pivot['pp_op'] = (pivot[anio_t] - pivot[anio_t1]) / pivot['op_t1'].replace(0, float('nan')) * 100

    op_totals['var_pct'] = (op_totals[anio_t] - op_totals[anio_t1]) / op_totals[anio_t1].replace(0, float('nan')) * 100
    op_totals['pp_nac'] = (op_totals[anio_t] - op_totals[anio_t1]) / nacional[anio_t1] * 100

    filas = []
    for op in sorted(pivot['nombre_operador'].astype(str).unique()):
        op_row = op_totals[op_totals['nombre_operador'] == op].iloc[0]
        fila_op = {'Operador / Sector': f'▶ {op}'}
        for a in anios:
            fila_op[str(a)] = fmt(op_row[a] / 1000, 1)
        fila_op[f'Var% {anio_t1}-{anio_t}'] = fmt(op_row['var_pct'], 1) + '%' if not pd.isna(op_row['var_pct']) else 'N/D'
        fila_op['PP → Nacional'] = fmt(op_row['pp_nac'], 2) + ' pp' if not pd.isna(op_row['pp_nac']) else 'N/D'
        fila_op['PP → Op'] = ''
        filas.append(fila_op)

        sectores_op = pivot[pivot['nombre_operador'] == op].sort_values(anio_t, ascending=False)
        for _, row in sectores_op.iterrows():
            fila_sec = {'Operador / Sector': f'   → {row["sector_consumo"]}'}
            for a in anios:
                fila_sec[str(a)] = fmt(row[a] / 1000, 1)
            fila_sec[f'Var% {anio_t1}-{anio_t}'] = fmt(row['var_pct'], 1) + '%' if not pd.isna(row['var_pct']) else 'N/D'
            fila_sec['PP → Nacional'] = ''
            fila_sec['PP → Op'] = fmt(row['pp_op'], 2) + ' pp' if not pd.isna(row['pp_op']) else 'N/D'
            filas.append(fila_sec)

    tabla = pd.DataFrame(filas)
    st.dataframe(tabla, use_container_width=True, height=500)

# ── Balance de Mercado: agrupación combinada ──────────────────────────────────
def agrupar_balance(dff_cont, dff_dem, granularidad):
    dff_cont = dff_cont.copy()
    dff_dem = dff_dem.copy()

    if granularidad == "Diario":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('D')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('D')
        dias_cont = dff_cont.groupby('periodo', observed=True)['fecha_dia'].first().reset_index()
        dias_cont['dias_calendario'] = 1
        dias_cont = dias_cont[['periodo', 'dias_calendario']]
        dias_dem = dff_dem.groupby('periodo', observed=True)['fecha_registro'].first().reset_index()
        dias_dem['dias_calendario'] = 1
        dias_dem = dias_dem[['periodo', 'dias_calendario']]
    elif granularidad == "Mensual":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('M')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('M')
        dias_cont = dff_cont.groupby('periodo', observed=True)['fecha_dia'].first().dt.days_in_month.reset_index()
        dias_cont.columns = ['periodo', 'dias_calendario']
        dias_dem = dff_dem.groupby('periodo', observed=True)['fecha_registro'].first().dt.days_in_month.reset_index()
        dias_dem.columns = ['periodo', 'dias_calendario']
    elif granularidad == "Trimestral":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('Q')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('Q')
        temp = dff_cont[['periodo', 'fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias_cont = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_cont.columns = ['periodo', 'dias_calendario']
        temp2 = dff_dem[['periodo', 'fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp2['mes'] = temp2['fecha_registro'].dt.to_period('M')
        temp2['dias_mes'] = temp2['fecha_registro'].dt.days_in_month
        dias_dem = temp2.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_dem.columns = ['periodo', 'dias_calendario']
    elif granularidad == "Anual":
        dff_cont['periodo'] = dff_cont['fecha_dia'].dt.to_period('Y')
        dff_dem['periodo'] = dff_dem['fecha_registro'].dt.to_period('Y')
        temp = dff_cont[['periodo', 'fecha_dia']].drop_duplicates('fecha_dia').copy()
        temp['mes'] = temp['fecha_dia'].dt.to_period('M')
        temp['dias_mes'] = temp['fecha_dia'].dt.days_in_month
        dias_cont = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_cont.columns = ['periodo', 'dias_calendario']
        temp2 = dff_dem[['periodo', 'fecha_registro']].drop_duplicates('fecha_registro').copy()
        temp2['mes'] = temp2['fecha_registro'].dt.to_period('M')
        temp2['dias_mes'] = temp2['fecha_registro'].dt.days_in_month
        dias_dem = temp2.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
        dias_dem.columns = ['periodo', 'dias_calendario']

    grp_cont = dff_cont.groupby('periodo', observed=True)['cantidad'].sum().reset_index()
    grp_cont = grp_cont.merge(dias_cont, on='periodo')
    grp_cont['gbtud_cont'] = grp_cont['cantidad'] / (grp_cont['dias_calendario'] * 1000)
    grp_cont['periodo_str'] = grp_cont['periodo'].astype(str)

    grp_dem = dff_dem.groupby('periodo', observed=True)['cantidad_entregada'].sum().reset_index()
    grp_dem = grp_dem.merge(dias_dem, on='periodo')
    grp_dem['gbtud_dem'] = grp_dem['cantidad_entregada'] / (grp_dem['dias_calendario'] * 1000)
    grp_dem['periodo_str'] = grp_dem['periodo'].astype(str)

    balance = grp_cont[['periodo_str', 'gbtud_cont']].merge(
        grp_dem[['periodo_str', 'gbtud_dem']], on='periodo_str', how='outer'
    ).fillna(0)
    balance['diferencia'] = balance['gbtud_cont'] - balance['gbtud_dem']
    balance['pct_sobre_demanda'] = (balance['diferencia'] / balance['gbtud_dem'].replace(0, float('nan')) * 100)
    return balance

# ════════════════════════════════════════════════════════════════════════════
# ── Sidebar ───────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⛽ Gas Natural")
    st.markdown("---")
    seccion = st.radio("Sección",
        ["⛽ Contratación", "📊 Demanda", "🔄 Balance de Mercado", "🔋 Producción", "⚡ Declaración de Producción"],
        label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Contratos al: {df['fecha_dia'].max().strftime('%d/%m/%Y')}")
    st.caption(f"Demanda al: {df_dem['fecha_registro'].max().strftime('%d/%m/%Y')}")

# ════════════════════════════════════════════════════════════════════════════
# ── SECCIÓN CONTRATACIÓN ──────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
if seccion == "⛽ Contratación":
    st.title("⛽ Contratación de Gas Natural")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Contratación", "📅 Contratación Vigente", "📈 Análisis Adicionales", "🏭 Concentración HHI"])

    with tab1:
        st.subheader("Contratación de Gas Natural")
        vendedor, comprador, modalidad, sector, mercado, tipo_demanda = construir_filtros(df, "tab1")
        col6, col7, col8 = st.columns(3)
        with col6:
            fecha_inicio = st.date_input("Fecha inicio", value=df['fecha_dia'].min().date(),
                                         min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t1_fi")
        with col7:
            fecha_fin = st.date_input("Fecha fin", value=df['fecha_dia'].max().date(),
                                      min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t1_ff")
        with col8:
            granularidad = st.selectbox("Ver por", ["Mensual", "Diario", "Trimestral", "Anual"], key="t1_gran")

        dff = df.copy()
        dff = dff[(dff['fecha_dia'].dt.date >= fecha_inicio) & (dff['fecha_dia'].dt.date <= fecha_fin)]
        dff = aplicar_filtros(dff, vendedor, comprador, modalidad, sector, mercado, tipo_demanda)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total GBTUD", fmt(dff['cantidad'].sum() / (dff['fecha_dia'].dt.days_in_month.mean() * 1000), 1))
        k2.metric("Precio Ponderado (USD/MBTUD)", fmt((dff['precio'] * dff['cantidad']).sum() / dff['cantidad'].sum(), 2))
        k3.metric("N° Contratos", fmt(dff['no_operacion'].nunique(), 0))
        k4.metric("N° Empresas", fmt(dff['nombre_vendedor'].nunique() + dff['nombre_comprador'].nunique(), 0))

        grp = agrupar(dff, granularidad)
        st.plotly_chart(construir_grafico(grp, 'Contratado (GBTUD) por sector de consumo'), use_container_width=True)

        st.subheader("Detalle de contratos")
        tabla = dff[['nombre_vendedor', 'nombre_comprador', 'cantidad', 'precio',
                     'modalidad', 'fecha_inicial', 'fecha_final', 'no_operacion']].copy()
        tabla['cantidad'] = tabla['cantidad'].apply(lambda x: fmt(x, 0))
        tabla['precio'] = tabla['precio'].apply(lambda x: fmt(x, 2))
        tabla.columns = ['Vendedor', 'Comprador', 'Cantidad (MBTU)', 'Precio (USD/MBTUD)', 'Modalidad', 'Fecha Inicial', 'Fecha Final', 'N° Operación']
        st.dataframe(tabla, use_container_width=True)

    with tab2:
        st.subheader("Contratación Vigente")
        vendedor2, comprador2, modalidad2, sector2, mercado2, tipo_demanda2 = construir_filtros(df, "tab2")
        col6b, col7b, col8b = st.columns(3)
        with col6b:
            vi_inicio = st.date_input("Fecha inicio", value=pd.Timestamp('2021-01-01').date(),
                                      min_value=df['fecha_inicial'].min().date(), max_value=df['fecha_final'].max().date(), key="t2_fi")
        with col7b:
            vi_fin = st.date_input("Fecha fin", value=df['fecha_final'].max().date(),
                                   min_value=df['fecha_inicial'].min().date(), max_value=df['fecha_final'].max().date(), key="t2_ff")
        with col8b:
            granularidad2 = st.selectbox("Ver por", ["Mensual", "Diario", "Trimestral", "Anual"], key="t2_gran")

        @st.cache_data
        def construir_vigente(vendedor, comprador, modalidad, sector, mercado, tipo_demanda):
            cols_contrato = ['no_operacion', 'nombre_vendedor', 'nombre_comprador',
                             'modalidad', 'mercado', 'sector_consumo', 'tipo_demanda',
                             'fecha_inicial', 'fecha_final', 'cantidad', 'precio']
            contratos = df[cols_contrato].groupby(
                ['no_operacion', 'fecha_inicial', 'fecha_final', 'sector_consumo'], observed=True
            ).first().reset_index().copy()
            contratos = aplicar_filtros(contratos, vendedor, comprador, modalidad, sector, mercado, tipo_demanda)
            if contratos.empty:
                return pd.DataFrame()
            contratos['fecha_inicial'] = pd.to_datetime(contratos['fecha_inicial'])
            contratos['fecha_final'] = pd.to_datetime(contratos['fecha_final'])
            contratos['fechas'] = contratos.apply(lambda r: pd.date_range(r['fecha_inicial'], r['fecha_final'], freq='D'), axis=1)
            df_vig = contratos.explode('fechas').rename(columns={'fechas': 'fecha_dia'})
            df_vig = df_vig[['fecha_dia', 'cantidad', 'precio', 'sector_consumo']].copy()
            df_vig['sector_consumo'] = df_vig['sector_consumo'].astype('category')
            return df_vig

        with st.spinner("Calculando contratación vigente..."):
            df_vig = construir_vigente(tuple(vendedor2), tuple(comprador2), tuple(modalidad2),
                                       tuple(sector2), tuple(mercado2), tuple(tipo_demanda2))

        if df_vig.empty:
            st.warning("No hay contratos para los filtros seleccionados.")
        else:
            df_vig_filtrado = df_vig[(df_vig['fecha_dia'].dt.date >= vi_inicio) & (df_vig['fecha_dia'].dt.date <= vi_fin)]
            if df_vig_filtrado.empty:
                st.warning("No hay datos en el rango de fechas seleccionado.")
            else:
                k1b, k2b, _ = st.columns(3)
                k1b.metric("Total GBTUD", fmt(df_vig_filtrado['cantidad'].sum() / (df_vig_filtrado['fecha_dia'].dt.days_in_month.mean() * 1000), 1))
                k2b.metric("Precio Ponderado (USD/MBTUD)", fmt((df_vig_filtrado['precio'] * df_vig_filtrado['cantidad']).sum() / df_vig_filtrado['cantidad'].sum(), 2))
                grp2 = agrupar(df_vig_filtrado, granularidad2)
                st.plotly_chart(construir_grafico(grp2, 'Contratación Vigente (GBTUD) por sector de consumo'), use_container_width=True)

                st.subheader("Contratos en el período")
                cols_contrato = ['no_operacion', 'nombre_vendedor', 'nombre_comprador',
                                 'modalidad', 'mercado', 'sector_consumo', 'tipo_demanda',
                                 'fecha_inicial', 'fecha_final', 'cantidad', 'precio']
                contratos_tabla = df[cols_contrato].groupby(
                    ['no_operacion', 'fecha_inicial', 'fecha_final', 'sector_consumo'], observed=True
                ).first().reset_index().copy()
                contratos_tabla = aplicar_filtros(contratos_tabla, vendedor2, comprador2, modalidad2, sector2, mercado2, tipo_demanda2)
                contratos_tabla = contratos_tabla[(contratos_tabla['fecha_inicial'].dt.date <= vi_fin) & (contratos_tabla['fecha_final'].dt.date >= vi_inicio)]
                contratos_tabla['cantidad'] = contratos_tabla['cantidad'].apply(lambda x: fmt(x, 0))
                contratos_tabla['precio'] = contratos_tabla['precio'].apply(lambda x: fmt(x, 2))
                contratos_tabla.columns = ['N° Operación', 'Vendedor', 'Comprador', 'Modalidad', 'Mercado', 'Sector', 'Tipo Demanda', 'Fecha Inicial', 'Fecha Final', 'Cantidad (MBTU)', 'Precio (USD/MBTUD)']
                st.dataframe(contratos_tabla, use_container_width=True)

    with tab3:
        st.subheader("Análisis Adicionales de Contratación")
        vendedor3, comprador3, modalidad3, sector3, mercado3, tipo_demanda3 = construir_filtros(df, "tab3")
        col_a, col_b, _ = st.columns(3)
        with col_a:
            fecha_inicio3 = st.date_input("Fecha inicio", value=df['fecha_dia'].min().date(),
                                          min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t3_fi")
        with col_b:
            fecha_fin3 = st.date_input("Fecha fin", value=df['fecha_dia'].max().date(),
                                       min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="t3_ff")
        dff3 = df.copy()
        dff3 = dff3[(dff3['fecha_dia'].dt.date >= fecha_inicio3) & (dff3['fecha_dia'].dt.date <= fecha_fin3)]
        dff3 = aplicar_filtros(dff3, vendedor3, comprador3, modalidad3, sector3, mercado3, tipo_demanda3)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie_chart(dff3, 'sector_consumo', 'GBTUD por Sector de Consumo', 'cantidad'), use_container_width=True)
        with col2:
            st.plotly_chart(pie_chart(dff3, 'modalidad', 'GBTUD por Modalidad', 'cantidad'), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(pie_chart(dff3, 'mercado', 'GBTUD por Mercado', 'cantidad'), use_container_width=True)
        with col4:
            st.plotly_chart(bar_chart_top(dff3, 'nombre_vendedor', 'Top 10 Vendedores', 'cantidad'), use_container_width=True)
        col5, _ = st.columns(2)
        with col5:
            st.plotly_chart(bar_chart_top(dff3, 'nombre_comprador', 'Top 10 Compradores', 'cantidad'), use_container_width=True)

    with tab4:
        st.subheader("Concentración de Mercado — Índice HHI")
        st.caption("Valores < 1.500 indican competencia, entre 1.500 y 2.500 concentración moderada, y > 2.500 mercado concentrado.")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            hhi_mercado = st.multiselect("Mercado", sorted(df['mercado'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="hhi_mercado")
        with col2:
            hhi_sector = st.multiselect("Sector consumo", sorted(df['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="hhi_sector")
        with col3:
            hhi_modalidad = st.multiselect("Modalidad", sorted(df['modalidad'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="hhi_modalidad")
        with col4:
            hhi_granularidad = st.selectbox("Ver por", ["Mensual", "Trimestral", "Anual"], key="hhi_gran")
        with col5:
            hhi_por = st.selectbox("Calcular HHI por", ["Vendedor", "Comprador"], key="hhi_por")
        col6, col7 = st.columns(2)
        with col6:
            hhi_inicio = st.date_input("Fecha inicio", value=df['fecha_dia'].min().date(),
                                       min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="hhi_fi")
        with col7:
            hhi_fin = st.date_input("Fecha fin", value=df['fecha_dia'].max().date(),
                                    min_value=df['fecha_dia'].min().date(), max_value=df['fecha_dia'].max().date(), key="hhi_ff")

        dff4 = df.copy()
        dff4 = dff4[(dff4['fecha_dia'].dt.date >= hhi_inicio) & (dff4['fecha_dia'].dt.date <= hhi_fin)]
        if hhi_mercado:
            dff4 = dff4[dff4['mercado'].isin(hhi_mercado)]
        if hhi_sector:
            dff4 = dff4[dff4['sector_consumo'].isin(hhi_sector)]
        if hhi_modalidad:
            dff4 = dff4[dff4['modalidad'].isin(hhi_modalidad)]

        col_empresa = 'nombre_vendedor' if hhi_por == "Vendedor" else 'nombre_comprador'
        if hhi_granularidad == "Mensual":
            dff4['periodo'] = dff4['fecha_dia'].dt.to_period('M')
        elif hhi_granularidad == "Trimestral":
            dff4['periodo'] = dff4['fecha_dia'].dt.to_period('Q')
        elif hhi_granularidad == "Anual":
            dff4['periodo'] = dff4['fecha_dia'].dt.to_period('Y')

        grp_hhi = dff4.groupby(['periodo', col_empresa], observed=True)['cantidad'].sum().reset_index()
        total_periodo = grp_hhi.groupby('periodo', observed=True)['cantidad'].sum().reset_index(name='total')
        grp_hhi = grp_hhi.merge(total_periodo, on='periodo')
        grp_hhi['participacion'] = grp_hhi['cantidad'] / grp_hhi['total'] * 100
        grp_hhi['hhi_parcial'] = grp_hhi['participacion'] ** 2
        hhi_por_periodo = grp_hhi.groupby('periodo', observed=True)['hhi_parcial'].sum().reset_index(name='hhi')
        hhi_por_periodo['periodo_str'] = hhi_por_periodo['periodo'].astype(str)
        top3 = grp_hhi.sort_values(['periodo', 'participacion'], ascending=[True, False]).groupby('periodo', observed=True).head(3)
        top3_resumen = top3.groupby('periodo', observed=True)[col_empresa].apply(lambda x: ', '.join(x.astype(str))).reset_index()
        top3_resumen.columns = ['periodo', 'top3']
        hhi_por_periodo = hhi_por_periodo.merge(top3_resumen, on='periodo', how='left')
        hhi_por_periodo['top3'] = hhi_por_periodo['top3'].fillna('Sin datos')

        fig4 = go.Figure()
        fig4.add_hrect(y0=0, y1=1500, fillcolor='green', opacity=0.07, line_width=0, annotation_text='Competitivo', annotation_position='right')
        fig4.add_hrect(y0=1500, y1=2500, fillcolor='yellow', opacity=0.1, line_width=0, annotation_text='Moderado', annotation_position='right')
        fig4.add_hrect(y0=2500, y1=10000, fillcolor='red', opacity=0.07, line_width=0, annotation_text='Concentrado', annotation_position='right')
        fig4.add_trace(go.Scatter(x=hhi_por_periodo['periodo_str'], y=hhi_por_periodo['hhi'],
                                  mode='lines+markers', name='HHI', line=dict(color='darkblue', width=2),
                                  customdata=hhi_por_periodo['top3'],
                                  hovertemplate='<b>%{x}</b><br>HHI: %{y:,.0f}<br>Top 3: %{customdata}<extra></extra>'))
        fig4.update_layout(title=f'Índice HHI por {hhi_por} — {hhi_granularidad}', xaxis_title='Período',
                           yaxis_title='HHI', height=450,
                           yaxis=dict(range=[0, max(10000, hhi_por_periodo['hhi'].max() * 1.1)]))
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("HHI por período")
        tabla_hhi = hhi_por_periodo[['periodo_str', 'hhi', 'top3']].copy()
        tabla_hhi['hhi'] = tabla_hhi['hhi'].apply(lambda x: fmt(x, 0))
        tabla_hhi.columns = ['Período', 'HHI', f'Top 3 {hhi_por}es']
        st.dataframe(tabla_hhi, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# ── SECCIÓN DEMANDA ───────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
elif seccion == "📊 Demanda":
    st.title("📊 Demanda de Gas Natural")
    tab_d1, tab_d2 = st.tabs(["📈 Demanda", "🔍 Análisis de Demanda"])

    with tab_d1:
        st.subheader("Demanda de Gas Natural — Registro de Entregas")
        col1, col2, col3 = st.columns(3)
        with col1:
            d_operador = st.multiselect("Operador", sorted(df_dem['nombre_operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="d_operador")
        with col2:
            d_sector = st.multiselect("Sector consumo", sorted(df_dem['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="d_sector")
        with col3:
            d_tipo = st.multiselect("Tipo demanda", sorted(df_dem['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="d_tipo")
        col4, col5, col6 = st.columns(3)
        with col4:
            d_inicio = st.date_input("Fecha inicio", value=df_dem['fecha_registro'].min().date(),
                                     min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="d_fi")
        with col5:
            d_fin = st.date_input("Fecha fin", value=df_dem['fecha_registro'].max().date(),
                                  min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="d_ff")
        with col6:
            d_gran = st.selectbox("Ver por", ["Mensual", "Diario", "Trimestral", "Anual"], key="d_gran")

        dfd = df_dem.copy()
        dfd = dfd[(dfd['fecha_registro'].dt.date >= d_inicio) & (dfd['fecha_registro'].dt.date <= d_fin)]
        if d_operador:
            dfd = dfd[dfd['nombre_operador'].isin(d_operador)]
        if d_sector:
            dfd = dfd[dfd['sector_consumo'].isin(d_sector)]
        if d_tipo:
            dfd = dfd[dfd['tipo_demanda'].isin(d_tipo)]

        k1d, k2d, _ = st.columns(3)
        k1d.metric("Total GBTUD", fmt(dfd['cantidad_entregada'].sum() / (dfd['fecha_registro'].dt.days_in_month.mean() * 1000), 1))
        k2d.metric("N° Operadores", fmt(dfd['nombre_operador'].nunique(), 0))

        grp_d = agrupar_demanda(dfd, d_gran)
        st.plotly_chart(construir_grafico_demanda(grp_d, 'Demanda (GBTUD) por sector de consumo'), use_container_width=True)

        st.subheader("Detalle de entregas")
        tabla_d = dfd[['fecha_registro', 'nombre_operador', 'sector_consumo', 'tipo_demanda', 'cantidad_entregada']].copy()
        tabla_d['cantidad_entregada'] = tabla_d['cantidad_entregada'].apply(lambda x: fmt(x, 0))
        tabla_d.columns = ['Fecha', 'Operador', 'Sector', 'Tipo Demanda', 'Cantidad (MBTU)']
        st.dataframe(tabla_d, use_container_width=True)

    with tab_d2:
        st.subheader("Análisis Adicionales de Demanda")
        col1, col2, col3 = st.columns(3)
        with col1:
            da_operador = st.multiselect("Operador", sorted(df_dem['nombre_operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="da_operador")
        with col2:
            da_sector = st.multiselect("Sector consumo", sorted(df_dem['sector_consumo'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="da_sector")
        with col3:
            da_tipo = st.multiselect("Tipo demanda", sorted(df_dem['tipo_demanda'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="da_tipo")
        col4, col5, _ = st.columns(3)
        with col4:
            da_inicio = st.date_input("Fecha inicio", value=df_dem['fecha_registro'].min().date(),
                                      min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="da_fi")
        with col5:
            da_fin = st.date_input("Fecha fin", value=df_dem['fecha_registro'].max().date(),
                                   min_value=df_dem['fecha_registro'].min().date(), max_value=df_dem['fecha_registro'].max().date(), key="da_ff")

        dfa = df_dem.copy()
        dfa = dfa[(dfa['fecha_registro'].dt.date >= da_inicio) & (dfa['fecha_registro'].dt.date <= da_fin)]
        if da_operador:
            dfa = dfa[dfa['nombre_operador'].isin(da_operador)]
        if da_sector:
            dfa = dfa[dfa['sector_consumo'].isin(da_sector)]
        if da_tipo:
            dfa = dfa[dfa['tipo_demanda'].isin(da_tipo)]

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie_chart(dfa, 'sector_consumo', 'Demanda por Sector de Consumo', 'cantidad_entregada'), use_container_width=True)
        with col2:
            st.plotly_chart(pie_chart(dfa, 'tipo_demanda', 'Demanda por Tipo', 'cantidad_entregada'), use_container_width=True)
        col3, _ = st.columns(2)
        with col3:
            st.plotly_chart(bar_chart_top(dfa, 'nombre_operador', 'Top 10 Operadores', 'cantidad_entregada'), use_container_width=True)

        st.markdown("---")
        st.subheader("Estacionalidad")
        st.plotly_chart(construir_estacionalidad(dfa), use_container_width=True)

        st.markdown("---")
        st.subheader("Resumen por Operador y Sector — Variaciones y Contribuciones")
        construir_tabla_resumen(dfa)

# ════════════════════════════════════════════════════════════════════════════
# ── SECCIÓN BALANCE DE MERCADO ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
elif seccion == "🔄 Balance de Mercado":
    st.title("🔄 Balance de Mercado")
    st.subheader("Contratación vs Demanda")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bm_empresa = st.multiselect("Empresa (Comprador/Operador)",
            sorted(set(df['nombre_comprador'].dropna().astype(str).unique()) &
                   set(df_dem['nombre_operador'].dropna().astype(str).unique())),
            placeholder="Todas", key="bm_empresa")
    with col2:
        bm_sector = st.multiselect("Sector consumo",
            sorted(set(df['sector_consumo'].dropna().astype(str).unique()) &
                   set(df_dem['sector_consumo'].dropna().astype(str).unique())),
            placeholder="Todos", key="bm_sector")
    with col3:
        bm_tipo = st.multiselect("Tipo demanda",
            sorted(df_dem['tipo_demanda'].dropna().astype(str).unique().tolist()),
            placeholder="Todas", key="bm_tipo")
    with col4:
        bm_modalidad = st.multiselect("Modalidad (contratos)",
            sorted(df['modalidad'].dropna().astype(str).unique().tolist()),
            placeholder="Todas", key="bm_modalidad")

    col5, col6, col7 = st.columns(3)
    with col5:
        bm_inicio = st.date_input("Fecha inicio",
            value=max(df['fecha_dia'].min().date(), df_dem['fecha_registro'].min().date()),
            min_value=max(df['fecha_dia'].min().date(), df_dem['fecha_registro'].min().date()),
            max_value=min(df['fecha_dia'].max().date(), df_dem['fecha_registro'].max().date()), key="bm_fi")
    with col6:
        bm_fin = st.date_input("Fecha fin",
            value=min(df['fecha_dia'].max().date(), df_dem['fecha_registro'].max().date()),
            min_value=max(df['fecha_dia'].min().date(), df_dem['fecha_registro'].min().date()),
            max_value=min(df['fecha_dia'].max().date(), df_dem['fecha_registro'].max().date()), key="bm_ff")
    with col7:
        bm_gran = st.selectbox("Ver por", ["Mensual", "Diario", "Trimestral", "Anual"], key="bm_gran")

    dff_cont = df.copy()
    dff_cont = dff_cont[(dff_cont['fecha_dia'].dt.date >= bm_inicio) & (dff_cont['fecha_dia'].dt.date <= bm_fin)]
    if bm_empresa:
        dff_cont = dff_cont[dff_cont['nombre_comprador'].isin(bm_empresa)]
    if bm_sector:
        dff_cont = dff_cont[dff_cont['sector_consumo'].isin(bm_sector)]
    if bm_modalidad:
        dff_cont = dff_cont[dff_cont['modalidad'].isin(bm_modalidad)]
    if bm_tipo:
        dff_cont = dff_cont[dff_cont['tipo_demanda'].isin(bm_tipo)]

    dff_dem = df_dem.copy()
    dff_dem = dff_dem[(dff_dem['fecha_registro'].dt.date >= bm_inicio) & (dff_dem['fecha_registro'].dt.date <= bm_fin)]
    if bm_empresa:
        dff_dem = dff_dem[dff_dem['nombre_operador'].isin(bm_empresa)]
    if bm_sector:
        dff_dem = dff_dem[dff_dem['sector_consumo'].isin(bm_sector)]
    if bm_tipo:
        dff_dem = dff_dem[dff_dem['tipo_demanda'].isin(bm_tipo)]

    balance = agrupar_balance(dff_cont, dff_dem, bm_gran)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Promedio GBTUD Contratado", fmt(balance['gbtud_cont'].mean(), 1))
    k2.metric("Promedio GBTUD Demanda", fmt(balance['gbtud_dem'].mean(), 1))
    k3.metric("Diferencia promedio", fmt(balance['diferencia'].mean(), 1))
    k4.metric("% Períodos sobrecontratados", fmt((balance['diferencia'] > 0).mean() * 100, 1) + '%')

    fig_bm = go.Figure()
    fig_bm.add_trace(go.Bar(name='Contratado', x=balance['periodo_str'], y=balance['gbtud_cont'],
        marker_color='steelblue', text=balance['gbtud_cont'].apply(lambda x: fmt(x, 1)),
        textposition='outside', textangle=-90, textfont=dict(size=7)))
    fig_bm.add_trace(go.Bar(name='Demanda', x=balance['periodo_str'], y=balance['gbtud_dem'],
        marker_color='goldenrod', text=balance['gbtud_dem'].apply(lambda x: fmt(x, 1)),
        textposition='outside', textangle=-90, textfont=dict(size=7)))
    fig_bm.update_layout(barmode='group', title='Contratación vs Demanda (GBTUD)',
        xaxis_title='Período', yaxis_title='GBTUD', height=450, legend=dict(orientation='v', x=1.02))
    st.plotly_chart(fig_bm, use_container_width=True)

    balance['label_dif'] = balance.apply(
        lambda r: fmt(r['diferencia'], 1) + ' (' + fmt(r['pct_sobre_demanda'], 1) + '%)', axis=1)
    fig_dif = go.Figure()
    fig_dif.add_trace(go.Bar(name='Diferencia (Cont - Dem)', x=balance['periodo_str'], y=balance['diferencia'],
        marker_color='darkorange', text=balance['label_dif'],
        textposition='outside', textangle=-90, textfont=dict(size=7)))
    fig_dif.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
    fig_dif.update_layout(title='Diferencia Contratación − Demanda (GBTUD) y % sobre Demanda',
        xaxis_title='Período', yaxis_title='GBTUD', height=400, legend=dict(orientation='v', x=1.02))
    st.plotly_chart(fig_dif, use_container_width=True)

    st.subheader("Detalle por período")
    tabla_bm = balance.copy()
    tabla_bm['gbtud_cont'] = tabla_bm['gbtud_cont'].apply(lambda x: fmt(x, 1))
    tabla_bm['gbtud_dem'] = tabla_bm['gbtud_dem'].apply(lambda x: fmt(x, 1))
    tabla_bm['diferencia'] = tabla_bm['diferencia'].apply(lambda x: fmt(x, 1))
    tabla_bm['pct_sobre_demanda'] = tabla_bm['pct_sobre_demanda'].apply(lambda x: fmt(x, 1) + '%' if not pd.isna(x) else 'N/D')
    tabla_bm = tabla_bm[['periodo_str', 'gbtud_cont', 'gbtud_dem', 'diferencia', 'pct_sobre_demanda']]
    tabla_bm.columns = ['Período', 'Contratado (GBTUD)', 'Demanda (GBTUD)', 'Diferencia (GBTUD)', '% sobre Demanda']
    st.dataframe(tabla_bm, use_container_width=True)

    st.markdown("---")
    st.subheader("Sobrecontratación por Empresa")

    grp_cont_emp = dff_cont.groupby(['nombre_comprador', 'tipo_demanda'], observed=True)['cantidad'].sum().reset_index()
    dias_prom = (pd.to_datetime(bm_fin) - pd.to_datetime(bm_inicio)).days + 1
    grp_cont_emp['gbtud'] = grp_cont_emp['cantidad'] / (dias_prom * 1000)
    grp_cont_emp = grp_cont_emp.rename(columns={'nombre_comprador': 'empresa', 'gbtud': 'gbtud_cont'})

    grp_dem_emp = dff_dem.groupby(['nombre_operador', 'tipo_demanda'], observed=True)['cantidad_entregada'].sum().reset_index()
    grp_dem_emp['gbtud'] = grp_dem_emp['cantidad_entregada'] / (dias_prom * 1000)
    grp_dem_emp = grp_dem_emp.rename(columns={'nombre_operador': 'empresa', 'gbtud': 'gbtud_dem'})

    sobrecont = grp_cont_emp[['empresa', 'tipo_demanda', 'gbtud_cont']].merge(
        grp_dem_emp[['empresa', 'tipo_demanda', 'gbtud_dem']], on=['empresa', 'tipo_demanda'], how='outer').fillna(0)
    sobrecont['diferencia'] = sobrecont['gbtud_cont'] - sobrecont['gbtud_dem']
    sobrecont['pct'] = (sobrecont['diferencia'] / sobrecont['gbtud_dem'].replace(0, float('nan')) * 100)

    emp_totals = sobrecont.groupby('empresa')[['gbtud_cont', 'gbtud_dem', 'diferencia']].sum().reset_index()
    emp_totals['pct'] = (emp_totals['diferencia'] / emp_totals['gbtud_dem'].replace(0, float('nan')) * 100)
    emp_totals = emp_totals[emp_totals['gbtud_dem'] > 0]
    emp_totals = emp_totals.sort_values('diferencia', ascending=False)

    filas = []
    for _, emp_row in emp_totals.iterrows():
        emp = emp_row['empresa']
        filas.append({'Empresa / Tipo Demanda': f'▶ {emp}',
                      'Contratado (GBTUD)': fmt(emp_row['gbtud_cont'], 1),
                      'Demanda (GBTUD)': fmt(emp_row['gbtud_dem'], 1),
                      'Diferencia (GBTUD)': fmt(emp_row['diferencia'], 1),
                      '% sobre Demanda': fmt(emp_row['pct'], 1) + '%' if not pd.isna(emp_row['pct']) else 'N/D'})
        tipos = sobrecont[(sobrecont['empresa'] == emp) & (sobrecont['gbtud_dem'] > 0)].sort_values('diferencia', ascending=False)
        for _, tipo_row in tipos.iterrows():
            filas.append({'Empresa / Tipo Demanda': f'   → {tipo_row["tipo_demanda"]}',
                          'Contratado (GBTUD)': fmt(tipo_row['gbtud_cont'], 1),
                          'Demanda (GBTUD)': fmt(tipo_row['gbtud_dem'], 1),
                          'Diferencia (GBTUD)': fmt(tipo_row['diferencia'], 1),
                          '% sobre Demanda': fmt(tipo_row['pct'], 1) + '%' if not pd.isna(tipo_row['pct']) else 'N/D'})
    st.dataframe(pd.DataFrame(filas), use_container_width=True, height=500)

# ════════════════════════════════════════════════════════════════════════════
# ── SECCIÓN PRODUCCIÓN ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
elif seccion == "🔋 Producción":
    st.title("🔋 Producción de Gas Natural")
    tab_p1, tab_p2 = st.tabs(["📊 Producción", "🔍 Análisis de Producción"])

    @st.cache_data
    def orden_operadores_produccion():
        df_p = cargar_produccion()
        return df_p.groupby('operador', observed=True)['energia_mbtu'].sum() \
                   .sort_values(ascending=False).index.astype(str).tolist()

    orden_ops = orden_operadores_produccion()
    df_prod = cargar_produccion()

    with tab_p1:
        st.subheader("Producción de Gas Natural")
        col1, col2, col3 = st.columns(3)
        with col1:
            p_operador = st.multiselect("Operador", sorted(df_prod['operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="p_operador")
        with col2:
            p_fuente = st.multiselect("Fuente", sorted(df_prod['fuente'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="p_fuente")
        with col3:
            p_tipo = st.multiselect("Tipo de producción", sorted(df_prod['tipo_produccion'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="p_tipo")
        col4, col5, col6 = st.columns(3)
        with col4:
            p_inicio = st.date_input("Fecha inicio", value=df_prod['fecha'].min().date(),
                                     min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="p_fi")
        with col5:
            p_fin = st.date_input("Fecha fin", value=df_prod['fecha'].max().date(),
                                  min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="p_ff")
        with col6:
            p_gran = st.selectbox("Ver por", ["Mensual", "Diario", "Trimestral", "Anual"], key="p_gran")

        dfp = df_prod.copy()
        dfp = dfp[(dfp['fecha'].dt.date >= p_inicio) & (dfp['fecha'].dt.date <= p_fin)]
        if p_operador:
            dfp = dfp[dfp['operador'].isin(p_operador)]
        if p_fuente:
            dfp = dfp[dfp['fuente'].isin(p_fuente)]
        if p_tipo:
            dfp = dfp[dfp['tipo_produccion'].isin(p_tipo)]

        k1p, k2p, k3p = st.columns(3)
        k1p.metric("Total GBTUD", fmt(dfp['energia_mbtu'].sum() / (dfp['fecha'].dt.days_in_month.mean() * 1000), 1))
        k2p.metric("N° Operadores", fmt(dfp['operador'].nunique(), 0))
        k3p.metric("N° Fuentes", fmt(dfp['fuente'].nunique(), 0))

        dfp_grp = dfp.copy()
        if p_gran == "Diario":
            dfp_grp['periodo'] = dfp_grp['fecha'].dt.to_period('D')
            dias_per = dfp_grp.groupby('periodo', observed=True)['fecha'].first().reset_index()
            dias_per['dias_calendario'] = 1
            dias_per = dias_per[['periodo', 'dias_calendario']]
        elif p_gran == "Mensual":
            dfp_grp['periodo'] = dfp_grp['fecha'].dt.to_period('M')
            dias_per = dfp_grp.groupby('periodo', observed=True)['fecha'].first().dt.days_in_month.reset_index()
            dias_per.columns = ['periodo', 'dias_calendario']
        elif p_gran in ["Trimestral", "Anual"]:
            dfp_grp['periodo'] = dfp_grp['fecha'].dt.to_period('Q' if p_gran == "Trimestral" else 'Y')
            temp = dfp_grp[['periodo', 'fecha']].drop_duplicates('fecha').copy()
            temp['mes'] = temp['fecha'].dt.to_period('M')
            temp['dias_mes'] = temp['fecha'].dt.days_in_month
            dias_per = temp.drop_duplicates('mes').groupby('periodo', observed=True)['dias_mes'].sum().reset_index()
            dias_per.columns = ['periodo', 'dias_calendario']

        grp_p = dfp_grp.groupby(['periodo', 'operador'], observed=True)['energia_mbtu'].sum().reset_index()
        grp_p = grp_p.merge(dias_per, on='periodo')
        grp_p['gbtud'] = grp_p['energia_mbtu'] / (grp_p['dias_calendario'] * 1000)
        grp_p['periodo_str'] = grp_p['periodo'].astype(str)

        total_per = grp_p.groupby('periodo')['gbtud'].sum().reset_index(name='gbtud_total')
        total_per = total_per.sort_values('periodo')
        total_per['var_pct'] = total_per['gbtud_total'].pct_change() * 100
        total_per['periodo_str'] = total_per['periodo'].astype(str)

        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        ops_en_grp = [o for o in orden_ops if o in grp_p['operador'].astype(str).unique()]
        for op in ops_en_grp:
            data_op = grp_p[grp_p['operador'].astype(str) == op]
            fig_p.add_trace(go.Bar(name=op, x=data_op['periodo_str'], y=data_op['gbtud']), secondary_y=False)
        fig_p.add_trace(go.Scatter(name='Var% período anterior', x=total_per['periodo_str'],
                                   y=total_per['var_pct'], mode='lines+markers',
                                   line=dict(color='red', width=2),
                                   hovertemplate='%{x}<br>Var%: %{y:.1f}%<extra></extra>'), secondary_y=True)
        for _, row in total_per.iterrows():
            fig_p.add_annotation(x=row['periodo_str'], y=row['gbtud_total'], text=fmt(row['gbtud_total'], 1),
                                 showarrow=False, textangle=-90, font=dict(size=7, color='black'), yshift=18)
        fig_p.update_layout(barmode='stack', title='Producción (GBTUD) por Operador',
                            xaxis_title='Período', height=450, legend=dict(orientation='v', x=1.08))
        fig_p.update_yaxes(title_text="GBTUD", secondary_y=False)
        fig_p.update_yaxes(title_text="Variación % período anterior", secondary_y=True)
        fig_p.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, secondary_y=True)
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Detalle de producción")
        tabla_p = dfp[['fecha', 'operador', 'fuente', 'tipo_produccion', 'energia_mbtu']].copy()
        tabla_p['energia_mbtu'] = tabla_p['energia_mbtu'].apply(lambda x: fmt(x, 0))
        tabla_p.columns = ['Fecha', 'Operador', 'Fuente', 'Tipo Producción', 'Energía (MBTU)']
        st.dataframe(tabla_p, use_container_width=True)

    with tab_p2:
        st.subheader("Análisis de Producción")
        col1, col2, col3 = st.columns(3)
        with col1:
            pa_operador = st.multiselect("Operador", sorted(df_prod['operador'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="pa_operador")
        with col2:
            pa_fuente = st.multiselect("Fuente", sorted(df_prod['fuente'].dropna().astype(str).unique().tolist()), placeholder="Todas", key="pa_fuente")
        with col3:
            pa_tipo = st.multiselect("Tipo de producción", sorted(df_prod['tipo_produccion'].dropna().astype(str).unique().tolist()), placeholder="Todos", key="pa_tipo")
        col4, col5, _ = st.columns(3)
        with col4:
            pa_inicio = st.date_input("Fecha inicio", value=df_prod['fecha'].min().date(),
                                      min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="pa_fi")
        with col5:
            pa_fin = st.date_input("Fecha fin", value=df_prod['fecha'].max().date(),
                                   min_value=df_prod['fecha'].min().date(), max_value=df_prod['fecha'].max().date(), key="pa_ff")

        dfpa = df_prod.copy()
        dfpa = dfpa[(dfpa['fecha'].dt.date >= pa_inicio) & (dfpa['fecha'].dt.date <= pa_fin)]
        if pa_operador:
            dfpa = dfpa[dfpa['operador'].isin(pa_operador)]
        if pa_fuente:
            dfpa = dfpa[dfpa['fuente'].isin(pa_fuente)]
        if pa_tipo:
            dfpa = dfpa[dfpa['tipo_produccion'].isin(pa_tipo)]

        CANACOL = ['CNE OIL & GAS SAS', 'CNEOG COLOMBIA']
        dfpa_canacol = dfpa.copy()
        dfpa_canacol['operador'] = dfpa_canacol['operador'].astype(str).apply(
            lambda x: 'Canacol Energy' if x in CANACOL else x)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(pie_chart(dfpa_canacol, 'operador', 'Producción por Operador (Canacol agrupado)', 'energia_mbtu'), use_container_width=True)
        with col2:
            st.plotly_chart(pie_chart(dfpa, 'fuente', 'Producción por Fuente/Campo', 'energia_mbtu'), use_container_width=True)
        st.plotly_chart(bar_chart_top(dfpa, 'fuente', 'Top 10 Fuentes por Producción', 'energia_mbtu'), use_container_width=True)

elif seccion == "⚡ Declaración de Producción":
    st.title("⚡ Declaración de Producción")
    tab_dp1, tab_dp2 = st.tabs(["📈 Potencial de Producción", "⚖️ PP vs Contratación vs PTDV"])

    with tab_dp1:
        st.subheader("Potencial de Producción (GBTUD) según Declaratoria")

        # ── Filtros ───────────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            dp_declaratoria = st.multiselect("Declaratoria",
                sorted(df_pot['periodo'].dropna().astype(str).unique().tolist()),
                default=['2026-2035'],
                key="dp_declaratoria")
        with col2:
            dp_campo = st.multiselect("Campo",
                sorted(df_pot['campo'].dropna().astype(str).unique().tolist()),
                placeholder="Todos", key="dp_campo")
        with col3:
            dp_operador = st.multiselect("Operador",
                sorted(df_pot['razon_social'].dropna().astype(str).unique().tolist()),
                placeholder="Todos", key="dp_operador")
        with col4:
            dp_variable = st.selectbox("Variable", ["PP", "PTDV", "PC (todas)"], key="dp_variable")

        with col5:
            anios_disponibles = sorted(df_pot['mes'].dt.year.unique().tolist())
            dp_anios = st.multiselect("Año", anios_disponibles,
                default=[a for a in anios_disponibles if a >= 2026],
                key="dp_anios")

        # ── Filtrar ───────────────────────────────────────────────────────
        dfp = df_pot.copy()
        if dp_declaratoria:
            dfp = dfp[dfp['periodo'].astype(str).isin(dp_declaratoria)]
        if dp_campo:
            dfp = dfp[dfp['campo'].astype(str).isin(dp_campo)]
        if dp_operador:
            dfp = dfp[dfp['razon_social'].astype(str).isin(dp_operador)]
        if dp_anios:
            dfp = dfp[dfp['mes'].dt.year.isin(dp_anios)]

        # Seleccionar variable
        if dp_variable == "PP":
            col_var = 'pp'
        elif dp_variable == "PTDV":
            col_var = 'ptdv'
        else:
            col_var = None  # suma de PCs

        # ── Calcular GBTUD mensual ────────────────────────────────────────
        if col_var:
            grp_mes = dfp.groupby(['mes', 'periodo'], observed=True)[col_var].sum().reset_index()
            grp_mes['gbtud'] = grp_mes[col_var] / 1000
        else:
            dfp['pc_total'] = (dfp['pc_consumo_interno'] + dfp['pc_exportaciones'] +
                               dfp['pc_refineria_barranca'] + dfp['pc_refineria_cartagena'])
            grp_mes = dfp.groupby(['mes', 'periodo'], observed=True)['pc_total'].sum().reset_index()
            grp_mes['gbtud'] = grp_mes['pc_total'] / (grp_mes['mes'].dt.days_in_month * 1000)
        grp_mes['mes_str'] = grp_mes['mes'].dt.strftime('%Y-%m')

        # ── Gráfico por año (promedio de GBTUD mensuales) ─────────────────
        grp_mes['anio'] = grp_mes['mes'].dt.year
        grp_anio = grp_mes.groupby(['anio', 'periodo'], observed=True)['gbtud'].mean().reset_index()

        fig_anio = go.Figure()
        for per in sorted(grp_anio['periodo'].astype(str).unique()):
            data_per = grp_anio[grp_anio['periodo'].astype(str) == per]
            fig_anio.add_trace(go.Scatter(
                name=per, x=data_per['anio'], y=data_per['gbtud'],
                mode='lines+markers', fill='tozeroy'
            ))
        fig_anio.update_layout(
            title=f'Potencial de Producción (GBTUD) al año según declaratoria — {dp_variable}',
            xaxis_title='Año', yaxis_title='GBTUD', height=400,
            legend=dict(orientation='v', x=1.02)
        )
        st.plotly_chart(fig_anio, use_container_width=True)

        # ── Gráfico por mes ───────────────────────────────────────────────
        fig_mes = go.Figure()
        for per in sorted(grp_mes['periodo'].astype(str).unique()):
            data_per = grp_mes[grp_mes['periodo'].astype(str) == per]
            fig_mes.add_trace(go.Scatter(
                name=per, x=data_per['mes_str'], y=data_per['gbtud'],
                mode='lines+markers', fill='tozeroy'
            ))
        fig_mes.update_layout(
            title=f'Potencial de Producción (GBTUD) al mes según declaratoria — {dp_variable}',
            xaxis_title='Mes', yaxis_title='GBTUD', height=400,
            legend=dict(orientation='v', x=1.02)
        )
        st.plotly_chart(fig_mes, use_container_width=True)

    with tab_dp2:
            st.subheader("PP, Producción Contratada y PTDV (GBTUD)")

            # ── Filtros (mismos que tab1) ──────────────────────────────────────
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                dp2_declaratoria = st.multiselect("Declaratoria",
                    sorted(df_pot['periodo'].dropna().astype(str).unique().tolist()),
                    default=['2026-2035'], key="dp2_declaratoria")
            with col2:
                dp2_campo = st.multiselect("Campo",
                    sorted(df_pot['campo'].dropna().astype(str).unique().tolist()),
                    placeholder="Todos", key="dp2_campo")
            with col3:
                dp2_operador = st.multiselect("Operador",
                    sorted(df_pot['razon_social'].dropna().astype(str).unique().tolist()),
                    placeholder="Todos", key="dp2_operador")
            with col4:
                anios_disponibles2 = sorted(df_pot['mes'].dt.year.unique().tolist())
                dp2_anios = st.multiselect("Año", anios_disponibles2,
                    default=[a for a in anios_disponibles2 if a >= 2026],
                    key="dp2_anios")
            with col5:
                dp2_gran = st.selectbox("Ver por", ["Mensual", "Anual"], key="dp2_gran")

            # ── Filtrar ───────────────────────────────────────────────────────
            dfp2 = df_pot.copy()
            if dp2_declaratoria:
                dfp2 = dfp2[dfp2['periodo'].astype(str).isin(dp2_declaratoria)]
            if dp2_campo:
                dfp2 = dfp2[dfp2['campo'].astype(str).isin(dp2_campo)]
            if dp2_operador:
                dfp2 = dfp2[dfp2['razon_social'].astype(str).isin(dp2_operador)]
            if dp2_anios:
                dfp2 = dfp2[dfp2['mes'].dt.year.isin(dp2_anios)]

            # ── Agrupar ───────────────────────────────────────────────────────
            grp2 = dfp2.groupby('mes')[['pp', 'pc_consumo_interno', 'ptdv']].sum().reset_index()
            grp2['gbtud_pp'] = grp2['pp'] / 1000
            grp2['gbtud_pc'] = grp2['pc_consumo_interno'] / 1000
            grp2['gbtud_ptdv'] = grp2['ptdv'] / 1000
            grp2['mes_str'] = grp2['mes'].dt.strftime('%Y-%m')

            if dp2_gran == "Anual":
                grp2['anio'] = grp2['mes'].dt.year
                grp2 = grp2.groupby('anio')[['gbtud_pp', 'gbtud_pc', 'gbtud_ptdv']].mean().reset_index()
                eje_x = grp2['anio']
                x_title = 'Año'
            else:
                eje_x = grp2['mes_str']
                x_title = 'Mes'

            # ── Gráfico ───────────────────────────────────────────────────────
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name='Potencial de Producción (PP)', x=eje_x, y=grp2['gbtud_pp'],
                mode='lines', fill='tozeroy', line=dict(color='steelblue'),
                fillcolor='rgba(70,130,180,0.3)'
            ))
            fig2.add_trace(go.Scatter(
                name='PC - Consumo Interno', x=eje_x, y=grp2['gbtud_pc'],
                mode='lines', fill='tozeroy', line=dict(color='green'),
                fillcolor='rgba(0,128,0,0.3)'
            ))
            fig2.add_trace(go.Scatter(
                name='PTDV', x=eje_x, y=grp2['gbtud_ptdv'],
                mode='lines', fill='tozeroy', line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.3)'
            ))
            fig2.update_layout(
                title='Potencial de Producción, Producción Contratada y PTDV (GBTUD)',
                xaxis_title=x_title, yaxis_title='GBTUD', height=500,
                legend=dict(orientation='h', y=1.08)
            )
            st.plotly_chart(fig2, use_container_width=True)  