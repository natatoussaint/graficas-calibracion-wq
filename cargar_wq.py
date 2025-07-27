import pandas as pd

def transformar_csv_a_diccionario(ruta_csv):
    df = pd.read_csv(ruta_csv)

    columnas_const = [col for col in df.columns if col not in ['ID', 'XS_WQ']]

    df_melted = df.melt(
        id_vars='XS_WQ',
        value_vars=columnas_const,
        var_name='Constituent',
        value_name='ConstantValue'
    )

    df_melted["BC"] = df_melted["XS_WQ"].apply(
        lambda x: f"BC=RioRancheria,TramoPrincipal,{int(x)},,,,,,,"
    )

    # Normaliza nombres de Constituent y BC para evitar errores por mayúsculas o espacios
    df_melted["Constituent"] = df_melted["Constituent"].str.replace(" ", "").str.strip()

    valores = {
        (row["Constituent"].lower(), row["BC"].strip().lower()): row["ConstantValue"]
        for _, row in df_melted.iterrows()
    }

    return valores

def modificar_archivo_wqx(ruta_entrada, valores):
    with open(ruta_entrada, 'r') as f:
        lines = f.readlines()

    new_lines = []
    current_constituyente = None
    last_bc = None
    insertados = 0
    dentro_serie = False
    insertar_luego = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith("Constituent="):
            current_constituyente = stripped.split("=")[1].replace(" ", "").strip().lower()
            last_bc = None

        elif stripped.startswith("BC="):
            last_bc = stripped.strip().lower()

        elif stripped == "BC Time Series":
            dentro_serie = True
            insertar_luego = True

        elif stripped.startswith("End Time Series Data"):
            dentro_serie = False
            insertar_luego = False

        elif dentro_serie and stripped.startswith("Observed Data=") and insertar_luego:
            new_lines.append(line)  # dejar Observed Data
            if current_constituyente and last_bc:
                clave = (current_constituyente, last_bc)
                if clave in valores:
                    valor = valores[clave]
                    new_lines.append(f"Constant Value={valor}\n")
                    new_lines.append("Constant Units=1\n")
                    insertados += 1
            continue

        new_lines.append(line)

    with open(ruta_entrada, 'w') as f:
        f.writelines(new_lines)

    print(f"✅ Se sobrescribió '{ruta_entrada}' con {insertados} constantes agregadas.")

# USO
archivo_csv = "BC.csv"
archivo_w01 = "Rio_Rancheria.w01"
valores_constantes = transformar_csv_a_diccionario(archivo_csv)
modificar_archivo_wqx(archivo_w01, valores_constantes)
