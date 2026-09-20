# Datos

Para reproducir el análisis a partir de `results/logits/` **no hace falta descargar nada**.
Lo de aquí abajo solo es necesario para reentrenar los modelos o regenerar la Figura 1.

## 1. Dataset base (no incluido, 23 GB)

Grim y Gradvohl (2024), CC BY 4.0:

- Zenodo: https://doi.org/10.5281/zenodo.10246577 — archivo `SFF_Magnetogram_Seq_MViTs.zip`
- Código original: https://github.com/lfgrim/SFF_MagSeq_MViTs

Contiene 73 810 magnetogramas en JPEG de 8 bits (HMI/SDO), los archivos de secuencia de
16 fotogramas, las etiquetas a 24 y 48 horas y la partición cronológica en 5 folds más el
test fijo. De ese dataset este trabajo reutiliza **solo la partición y las etiquetas**.

Tras descomprimirlo:

```bash
export SFMM_DATA=/ruta/a/SFF_MagSeq_MViTs
```

La estructura que esperan los scripts es la original del dataset:

```
$SFMM_DATA/
├── Seq_Magnetogram/
│   ├── M24/{Seqs16/, Seq16_flare_Mclass_24h_TrainVal{0..4}.txt, ..._Test.txt}
│   └── M48/{Seqs16/, Seq16_flare_Mclass_48h_TrainVal{0..4}.txt, ..._Test.txt}
├── magnetogram_jpg/
└── para_flare_21params.txt        <- se añade desde aquí (punto 3)
```

## 2. Magnetogramas en FITS (no incluidos, cientos de GB)

El JPEG de 8 bits descarta la escala física del campo en Gauss y su rango dinámico. La rama
visual de este trabajo parte de los FITS originales del Joint Science Operations Center:

```bash
python scripts/download_jsoc_fits.py
python scripts/preprocess_fits_to_npy.py
```

Cada fotograma se recorta a la escala estándar de saturación de HMI/SHARP (+-500 G), se
normaliza a [-1, 1], se replica a 3 canales y se redimensiona a 224 x 224.

## 3. Complemento de parámetros SHARP (incluido)

`para_flare_21params.txt.gz` (6.2 MB comprimido, 19 MB sin comprimir).

El dataset base trae 10 atributos numéricos y **omite el predictor más influyente, TOTUSJH**,
además de TOTUSJZ y R_VALUE. Este archivo se descargó de la serie `hmi.sharp_720s` del JSOC
para completar 21 parámetros, de los cuales el paper usa 17.

Formato: 73 810 filas, una por magnetograma. Primera columna el nombre del fichero FITS,
seguida de los 21 parámetros en este orden:

```
TOTUSJH  TOTPOT   TOTUSJZ  USFLUX   USFLUXL
MEANGBH  MEANGBL  MEANGBT  MEANGBZ
MEANJZH  MEANJZD  ABSNJZH  SAVNCPP  MEANALP
MEANPOT  MEANSHR  SHRGT45  AREA_ACR NACR
MEANGAM  R_VALUE
```

Instalación:

```bash
gunzip -k data/para_flare_21params.txt.gz
mv data/para_flare_21params.txt "$SFMM_DATA"/
```

Para regenerarlo desde cero: `python scripts/download_jsoc_sharp.py` (horas de consultas al JSOC).

TOTBSQ y MEANGBR no están disponibles en esa serie, por eso el conjunto completo es de 21 y
no de 23.
