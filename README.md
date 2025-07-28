# Gráficas de calibración

Ejecuta `exportar_wq.py` desde el mismo directorio donde están los archivos CSV.
El script buscará `Results.csv`, `OB.csv`, `Mapeo.csv` y `Vertimientos.csv`
y generará gráficos y métricas en la carpeta `output`.

Los títulos y etiquetas usan la fuente **Nunito Sans** incluida en el repositorio.
Puedes cambiar el tamaño modificando las variables `TITLE_SIZE` y `LABEL_SIZE` en
`exportar_wq.py`.

Las marcas de vertimientos se etiquetan de forma que las leyendas cercanas no se
solapen y la tabla de métricas se exporta con dos decimales, resaltando en rojo
los valores de *RMSCV* mayores al 30 %.

Si lo usas desde un entorno interactivo como Jupyter, asegúrate de definir el
directorio actual al de este repositorio para que los archivos se carguen
correctamente.
