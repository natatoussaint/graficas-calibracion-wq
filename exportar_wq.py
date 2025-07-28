import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm
from adjustText import adjust_text  # importar adjustText

# --- 1. Rutas y carga de archivos ---
base_dir = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
output_dir = base_dir / 'output'
output_dir.mkdir(exist_ok=True)

df_sim  = pd.read_csv(base_dir / 'Results.csv',      encoding='latin-1')
df_obs  = pd.read_csv(base_dir / 'OB.csv',           encoding='latin-1')
df_map  = pd.read_csv(base_dir / 'Mapeo.csv',        encoding='latin-1')
df_vert = pd.read_csv(base_dir / 'Vertimientos.csv', encoding='latin-1')

# --- 2. Limpiar encabezados y descartar índices inútiles ---
for df in (df_sim, df_obs, df_map, df_vert):
    df.columns = df.columns.str.strip()
df_sim = df_sim.loc[:, ~df_sim.columns.str.startswith('Unnamed')]

# --- 3. Renombrar RS y vert_name ---
df_obs.rename(columns={'XS': 'RS'}, inplace=True)
df_vert.rename(columns={'XS': 'RS', 'ID': 'vert_name'}, inplace=True)

# --- 4. Extraer RS numérico con decimales ---
for df in (df_sim, df_obs, df_vert):
    if 'RS' in df.columns:
        df['RS'] = df['RS'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

# --- 5. Calcular distancia acumulada ---
df_sim['Distance'] = df_sim['WQ Cell Length (m)'].cumsum().shift(fill_value=0)
dist_ref = df_sim[['RS', 'Distance']].drop_duplicates()
df_obs  = df_obs.merge(dist_ref, on='RS', how='left')
df_vert = df_vert.merge(dist_ref, on='RS', how='left')

# --- 6. Funciones de limpieza y normalización de nombres ---
def clean_key(s):
    return re.sub(r'\W+', '', str(s)).lower().strip()

def normalize_filename(s):
    name = re.sub(r'\W+', '_', str(s)).strip('_').lower()
    return name

# --- 7. Mapeos clean → original de columnas ---
sim_cols = {clean_key(c): c for c in df_sim.columns}
obs_cols = {clean_key(c): c for c in df_obs.columns}

# --- 8. Fuente ---
font_path = base_dir / "nunito-sans.extrabold.ttf"
font_prop = (fm.FontProperties(fname=str(font_path))
             if font_path.exists()
             else fm.FontProperties(weight='bold'))
TITLE_SIZE = 15
LABEL_SIZE = 11

# --- 9. Graficar y calcular métricas ---
metrics = []

for _, row in df_map.iterrows():
    code_raw     = row['Nombre HEC RAS']
    display_name = row['Nombre']
    unit         = row['Unidades']
    key          = clean_key(code_raw)

    sim_col = sim_cols.get(key)
    obs_col = obs_cols.get(key)
    if sim_col is None or obs_col is None:
        print(f"⚠️ Saltando '{code_raw}': columna no encontrada.")
        continue

    sim_series  = df_sim[sim_col]
    obs_series  = df_obs[obs_col]
    dist_series = df_obs['Distance']

    mask = sim_series.notna() & obs_series.notna() & dist_series.notna()
    if not mask.any():
        print(f"⚠️ Sin datos válidos para '{code_raw}', se omite.")
        continue

    sim_series  = sim_series[mask]
    obs_series  = obs_series[mask]
    dist_series = dist_series[mask]

    # Calcular métricas
    rmse  = np.sqrt(mean_squared_error(obs_series, sim_series))
    rmscv = rmse / obs_series.mean() * 100
    r2    = r2_score(obs_series, sim_series)
    metrics.append([display_name, rmse, rmscv, r2])

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_sim['Distance'], df_sim[sim_col],
            label='Simulado', color='royalblue', lw=2)
    ax.scatter(dist_series, obs_series,
               edgecolor='black', facecolor='white',
               label='Observado', s=60, zorder=3)

    # Reservar espacio extra arriba para las etiquetas
    ymax = df_sim[sim_col].max()
    ymin = df_sim[sim_col].min()
    ax.set_ylim(bottom=ymin, top=ymax * 1.2)

    # Posición inicial de los textos
    y0 = ymax * 1.05

    # Dibujar líneas y añadir textos (clip_on=False para que salgan)
    texts = []
    for _, v in df_vert.dropna(subset=['Distance']).iterrows():
        x = v['Distance']
        ax.axvline(x, color='red', linestyle='--', lw=1)
        txt = ax.text(
            x, y0, v['vert_name'],
            rotation=90,
            ha='center', va='bottom',
            color='red',
            fontsize=9,
            fontproperties=font_prop,
            clip_on=False
        )
        texts.append(txt)

    # Ajuste automático solo entre textos (no considera curvas ni puntos)
    adjust_text(
        texts,
        only_move={'text': 'y'},
        add_objects=[],             # ignora líneas y puntos en el ajuste
        expand_text=(2.0, 2.0),      # más separación entre textos
        force_text=0.8,             # fuerza para separar textos
        lim=100                     # límite de iteraciones
    )

    # Leyenda y estilo
    ax.plot([], [], linestyle='--', color='red', label='Vertimientos')
    ax.set_xlim(left=0)
    ax.set_title(display_name, fontsize=TITLE_SIZE, fontproperties=font_prop)
    ax.set_xlabel('Abscisa (m)', fontsize=LABEL_SIZE, fontproperties=font_prop)
    ax.set_ylabel(f'{display_name} [{unit}]', fontsize=LABEL_SIZE, fontproperties=font_prop)
    ax.legend(prop=font_prop)
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()

    # Guardar
    fname = f"graf_{normalize_filename(display_name)}.png"
    fig.savefig(output_dir / fname, dpi=300)
    plt.close(fig)
    print(f"✅ Gráfico generado: {fname}")

# --- 10. Exportar tabla de métricas ---
if metrics:
    df_m = pd.DataFrame(metrics, columns=['Parámetro', 'RMSE', 'RMSCV (%)', 'R²'])
    df_m = df_m.round(2)

    colors = []
    for _, r in df_m.iterrows():
        row_cols = ['white'] * len(df_m.columns)
        if r['RMSCV (%)'] > 30:
            idx = df_m.columns.get_loc('RMSCV (%)')
            row_cols[idx] = 'lightcoral'
        colors.append(row_cols)

    fig, ax = plt.subplots(figsize=(10, 0.5 + len(df_m) * 0.35))
    ax.axis('off')
    tabla = ax.table(
        cellText=df_m.values,
        colLabels=df_m.columns,
        cellColours=colors,
        cellLoc='center',
        loc='center'
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    fig.tight_layout()

    tabla_path = output_dir / 'metrics_table.png'
    fig.savefig(tabla_path, dpi=300)
    plt.close(fig)
    print(f"📊 Tabla de métricas generada: {tabla_path.name}")

print("✅ Proceso completo. Revisa la carpeta 'output' para los PNG generados.")
