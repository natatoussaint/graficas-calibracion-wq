import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Configuración de rutas ---
if '__file__' in globals():
    base_dir = Path(__file__).resolve().parent
else:
    # Para entornos interactivos como Jupyter
    base_dir = Path.cwd()

results_file = base_dir / 'Results.csv'
obs_file     = base_dir / 'OB.csv'
map_file     = base_dir / 'Mapeo.csv'
vert_file    = base_dir / 'Vertimientos.csv'
out_dir      = base_dir / 'output'
out_dir.mkdir(exist_ok=True)

print(f"Leyendo datos desde: {base_dir}")

# --- 2. Función para limpiar claves ---
def clean(name: str) -> str:
    return re.sub(r'\W+', '', str(name)).lower().strip()

# --- 3. Carga de datos (Latin-1 para tildes, µ, etc.) ---
df_sim  = pd.read_csv(results_file, encoding='latin-1')
df_obs  = pd.read_csv(obs_file,     encoding='latin-1')
df_map  = pd.read_csv(map_file,     encoding='latin-1')
df_vert = pd.read_csv(vert_file,    encoding='latin-1')

# --- 4. Limpiar espacios en encabezados ---
for df in (df_sim, df_obs, df_map, df_vert):
    df.columns = df.columns.str.strip()

# --- 5. Renombrar columna llave XS → RS ---
for df in (df_sim, df_obs):
    if 'XS' in df.columns:
        df.rename(columns={'XS': 'RS'}, inplace=True)

# --- 6. Renombrar vertimientos: XS → RS y ID → vert_name ---
if 'XS' in df_vert.columns:
    df_vert.rename(columns={'XS': 'RS'}, inplace=True)
if 'ID' in df_vert.columns:
    df_vert.rename(columns={'ID': 'vert_name'}, inplace=True)

# --- 7. Asegurar que RS sea string en todos los DataFrames ---
for df in (df_sim, df_obs, df_vert):
    if 'RS' in df.columns:
        df['RS'] = df['RS'].astype(str).str.strip()

# --- 8. Calcular abscisa acumulada ---
if 'WQ Cell Length (m)' in df_sim.columns:
    df_sim['Distance'] = df_sim['WQ Cell Length (m)'].cumsum().shift(fill_value=0)
else:
    df_sim['Distance'] = np.arange(len(df_sim))

# --- 9. Merge simulados + observados ---
df_all = df_sim.merge(df_obs, on='RS', how='left')

# --- 10. Preparar vertimientos con su abscisa ---
if 'Distance' in df_sim.columns and 'RS' in df_vert.columns:
    df_vert = df_vert.merge(df_sim[['RS','Distance']], on='RS', how='left')

# --- 11. Crear índices limpios de columnas para sim/obs ---
sim_idx = { clean(col): col for col in df_all.columns }
obs_idx = { clean(col): col for col in df_all.columns }

# --- 12. Loop por cada fila de Mapeo.csv ---
metrics = []
for _, row in df_map.iterrows():
    raw_code     = row['Nombre HEC RAS']
    display_name = row['Nombre']
    unit         = row['Unidades']
    key          = clean(raw_code)

    sim_col = sim_idx.get(key)
    obs_col = obs_idx.get(key)
    if sim_col is None or obs_col is None:
        print(f"⚠️ Saltando '{raw_code}' (columna no hallada).")
        continue

    sim = df_all[sim_col]
    obs = df_all[obs_col]

    # 12a. filtrar NaN antes de métricas
    mask = sim.notna() & obs.notna()
    if mask.sum() == 0:
        print(f"⚠️ Sin datos válidos para '{raw_code}', se omite.")
        continue
    sim, obs = sim[mask], obs[mask]

    # 12b. Calcular métricas
    rmse  = np.sqrt(mean_squared_error(obs, sim))
    rmscv = rmse / obs.mean() * 100
    r2    = r2_score(obs, sim)
    metrics.append([display_name, rmse, rmscv, r2])

    # 12c. Generar gráfica
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_all['Distance'],      df_all[sim_col], label='Simulado')
    ax.scatter(df_all['Distance'][mask], obs, edgecolor='k', s=40, label='Observado')
    if 'vert_name' in df_vert.columns:
        for _, vr in df_vert.iterrows():
            ax.axvline(vr['Distance'], ls='--', color='gray')
            ax.text(vr['Distance'], max(sim.max(),obs.max())*0.9,
                    vr['vert_name'], rotation=90, va='top', ha='right')
    ax.set_xlabel('Abscisa (m)')
    ax.set_ylabel(f"{display_name} [{unit}]")
    ax.set_title(display_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"graf_{key}.png", dpi=300)
    plt.close(fig)

# --- 13. Exportar tabla de métricas con resaltado condicional ---
df_met = pd.DataFrame(metrics, columns=['Parámetro','RMSE','RMSCV (%)','R²'])
fig, ax = plt.subplots(figsize=(10, 0.5 + 0.35*len(df_met)))
ax.axis('off')
tbl = ax.table(cellText=df_met.values,
               colLabels=df_met.columns,
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for i, r in df_met.iterrows():
    if r['RMSCV (%)'] > 30:
        tbl[i+1,2].set_facecolor('lightcoral')
    if r['R²'] < 0.8:
        tbl[i+1,3].set_facecolor('lightcoral')
fig.tight_layout()
fig.savefig(out_dir / 'metrics_table.png', dpi=300)
plt.close(fig)

print(f"✅ ¡Proceso completo! Revisa '{out_dir}' para tus gráficos y tabla.")