# SF-MM: complementariedad imagen-física en la predicción de fulguraciones solares

Código, resultados y estudios de ablación de:

> G. D. Muñoz Curi y R. Cardenas Talavera.
> *Image-Physics Complementarity for M- and X-Class Solar Flare Prediction: A Multimodal Study with Swin3D and BiLSTM.*
> SIMBig 2026. Springer, Communications in Computer and Information Science (CCIS).

El trabajo compara, en igualdad de condiciones, dos modalidades para predecir fulguraciones de clase M o superior a 24 y 48 horas:

- **Rama visual**: Swin3D-T sobre secuencias de 16 magnetogramas FITS.
- **Rama física**: BiLSTM sobre 17 parámetros SHARP.
- **Fusión tardía**: ensemble ponderado de probabilidades y meta-clasificador (stacking).

El resultado principal es negativo y está documentado aquí de punta a punta: ninguna estrategia de fusión tardía supera a la mejor rama individual.

---

## Qué contiene este repositorio

| Carpeta | Contenido |
|---|---|
| `scripts/` | Descarga de datos, datasets, entrenamiento y extracción de logits del pipeline principal |
| `analysis/` | Métricas, bootstrap, fusión, calibración y generación de las figuras del paper |
| `configs/` | Configuraciones YAML de los modelos publicados (5 folds x 2 horizontes) |
| `studies/` | Los cuatro estudios de ablación, cada uno con su README, su código y sus resultados |
| `results/logits/` | Logits y etiquetas de todos los modelos entrenados (126 archivos, 4.6 MB) |
| `results/reports/` | Informes de texto con los números publicados |
| `data/` | Complemento de parámetros SHARP e instrucciones para obtener el dataset base |

**Qué no contiene**: los checkpoints de los modelos (unos 31 GB) ni las imágenes del dataset (23 GB). Ambos se regeneran o se descargan siguiendo las instrucciones de abajo.

**Por qué sí están los logits**: son 4.6 MB y permiten recalcular **todas** las tablas y figuras del paper sin GPU y sin volver a entrenar. Si solo quiere verificar los resultados, es el único insumo que necesita.

---

## Instalación

```bash
git clone <url-del-repositorio> sf-multimodal
cd sf-multimodal
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Para reproducir únicamente el análisis basta con el primer bloque de `requirements.txt` (numpy, scipy, scikit-learn, matplotlib, PyYAML, tqdm). PyTorch solo hace falta para reentrenar.

Probado con Python 3.13 en Linux.

---

## Reproducir los resultados sin GPU

Esta es la vía rápida. Parte de los logits incluidos y no necesita el dataset.

```bash
# Métricas de cada rama e intervalos de confianza (bootstrap estratificado, B=10000)
python analysis/bootstrap_ci.py

# Comparaciones pareadas entre ramas y estrategias de fusión
python analysis/bootstrap_paired.py

# Reglas de fusión fijas (media, producto, máximo, mínimo)
python analysis/late_fusion_rules.py

# Stacking con meta-MLP, 5 folds
python analysis/stacking_5fold.py

# Calibración y Brier Skill Score
python analysis/bss_calibration.py

# Techo estructural de la fusión a nivel de decisión
python analysis/diversidad_ramas.py

# Tabla completa de métricas de todas las variantes
python analysis/metricas_variantes.py
```

Los informes se escriben en `results/reports/`. Varios scripts llevan **controles incorporados**: comparan su salida con los valores publicados e imprimen `OK` o `DIFIERE`. Por ejemplo, `metricas_variantes.py` verifica que el BiLSTM-17 a 48 h reproduzca POD 0.903, FAR 0.653, HSS 0.464 y F1 0.499.

Las figuras del paper:

```bash
python analysis/fig_sharp_importance_simbig55.py
python analysis/fig_calibration_simbig55.py
python analysis/fig_roc_2panel_simbig55.py
python analysis/fig_pipeline_simbig55.py     # requiere el dataset (ver abajo)
```

El bootstrap con B=10000 tarda varios minutos por script.

---

## Obtener los datos

Hacen falta tres piezas. Solo la tercera está en este repositorio.

### 1. Dataset base (particiones, etiquetas y secuencias)

De Grim y Gradvohl (2024), publicado con licencia CC BY 4.0:

- Zenodo: https://doi.org/10.5281/zenodo.10246577 (archivo `SFF_Magnetogram_Seq_MViTs.zip`, 23.1 GB)
- Código original: https://github.com/lfgrim/SFF_MagSeq_MViTs

Aporta la partición cronológica en 5 folds, las etiquetas a 24 y 48 h, los archivos de secuencia de 16 fotogramas y los magnetogramas en JPEG de 8 bits.

Descomprímalo y apunte la variable de entorno al directorio resultante:

```bash
export SFMM_DATA=/ruta/a/SFF_MagSeq_MViTs
```

Todos los scripts y configuraciones usan `${SFMM_DATA}`; no hay rutas absolutas en el código.

### 2. Magnetogramas en FITS

El dataset base trae los magnetogramas en JPEG de 8 bits, lo que descarta la escala física en Gauss. Este trabajo los vuelve a descargar del Joint Science Operations Center en formato FITS:

```bash
python scripts/download_jsoc_fits.py
python scripts/preprocess_fits_to_npy.py
```

La descarga es larga (decenas de horas según la cola del JSOC) y ocupa cientos de GB.

### 3. Complemento de parámetros SHARP (incluido)

El dataset base trae 10 atributos numéricos y omite el predictor más influyente, TOTUSJH, además de TOTUSJZ y R_VALUE. Este trabajo descargó del JSOC la serie `hmi.sharp_720s` para completar 21 parámetros, de los que se usan 17.

Ese archivo **sí está incluido**, comprimido, porque es el aporte de este trabajo sobre el dataset base y regenerarlo exige horas de consultas al JSOC:

```bash
gunzip -k data/para_flare_21params.txt.gz
mv data/para_flare_21params.txt "$SFMM_DATA"/
```

Son 73 810 filas, una por magnetograma, con los 21 parámetros SHARP.

Si prefiere regenerarlo: `python scripts/download_jsoc_sharp.py`.

---

## Reentrenar desde cero

Requiere GPU. Los modelos publicados se entrenaron en una NVIDIA RTX 3060 de 6 GB.

```bash
export SFMM_DATA=/ruta/a/SFF_MagSeq_MViTs

# Rama visual (Swin3D-T sobre FITS)
python scripts/train_swin3d_standalone_fits.py --config configs/swin3d_standalone_fits_48h.yaml

# Rama física (BiLSTM sobre 17 SHARP), un fold
python scripts/train_lstm_standalone.py --config configs/lstm_cv_48h_k0.yaml

# Extraer logits de todos los folds y horizontes
python scripts/save_logits_swin3d.py
python scripts/save_logits_lstm.py
```

Añada `--smoke 5` a cualquier script de entrenamiento para una pasada corta de comprobación.

El protocolo es validación cruzada cronológica de 5 folds con un conjunto de test fijo de 9384 secuencias (el bloque más reciente). El umbral de decisión se ajusta en el fold de validación y se aplica al test sin reajustar.

---

## Estudios de ablación

Cada estudio es autocontenido y tiene su propio README con la pregunta que responde, cómo ejecutarlo y qué se concluyó.

| Estudio | Pregunta |
|---|---|
| [`01_recurrent_architecture`](studies/01_recurrent_architecture/) | ¿Está limitada la rama física por la arquitectura recurrente? 120 entrenamientos variando ancho, profundidad y tipo de celda (BiLSTM frente a BiGRU) |
| [`02_sharp_provenance`](studies/02_sharp_provenance/) | ¿De dónde salen los parámetros SHARP? Cuáles se calculan sobre el mismo magnetograma que ve la rama visual y cuáles necesitan el campo vectorial |
| [`03_los_derived_ablation`](studies/03_los_derived_ablation/) | Si se quitan los parámetros derivados del magnetograma de línea de visión, ¿aporta algo la fusión? Variantes BiLSTM-21, BiLSTM-16 y BiLSTM-12 |
| [`04_feature_selection_leakage`](studies/04_feature_selection_leakage/) | La preselección de variables se hizo sobre el dataset completo. ¿Transporta información de los periodos de evaluación? |

---

## Resultados incluidos

`results/logits/` contiene logits y etiquetas de validación y test para cada modelo, horizonte y fold, con el formato `{modelo}_{horizonte}h_k{fold}_{split}.npz`:

| Modelo | Descripción |
|---|---|
| `swin3d` | Rama visual sobre FITS |
| `swin3d_jpeg` | Rama visual sobre los JPEG del dataset base |
| `lstm` | Rama física, 17 parámetros SHARP (el del paper) |
| `lstm21` | Los 21 parámetros, sin preselección |
| `lstm16` | Los 16 vectoriales, sin los derivados del LoS |
| `lstm12` | Los 12 vectoriales de la selección original |
| `lstm16b` | Los 17 del paper sin MEANGBZ |

Cada archivo guarda `logits` y `labels`. Para cargarlos:

```python
import numpy as np
d = np.load("results/logits/lstm_48h_k0_test.npz")
d["logits"], d["labels"]
```

---

## Variables de entorno

| Variable | Para qué | Por defecto |
|---|---|---|
| `SFMM_DATA` | Directorio del dataset base descomprimido | sin valor (obligatoria para entrenar) |
| `SFMM_ROOT` | Raíz del repositorio | se deduce de la ubicación del archivo |
| `SFMM_LOGITS` | Directorio de logits | `results/logits` |
| `SFMM_OUT` | Dónde escribir los informes | `results/reports` |

---

## Cómo citar

```bibtex
@inproceedings{MunozCuri2026SFMM,
  author    = {Mu\~noz Curi, Giomar D. and Cardenas Talavera, Rolando},
  title     = {Image--Physics Complementarity for {M}- and {X}-Class Solar Flare
               Prediction: A Multimodal Study with {Swin3D} and {BiLSTM}},
  booktitle = {Information Management and Big Data (SIMBig 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  year      = {2026}
}
```

Si usa el dataset, cite también el trabajo original:

```bibtex
@article{Grim2024,
  author  = {Grim, L. F. L. and Gradvohl, A. L. S.},
  title   = {Solar flare forecasting based on magnetogram sequences learning with
             multiscale vision transformers and data augmentation techniques},
  journal = {Solar Physics},
  volume  = {299},
  number  = {3},
  pages   = {33},
  year    = {2024}
}
```

---

## Licencia

MIT, ver [LICENSE](LICENSE). El dataset base de Grim y Gradvohl se distribuye bajo CC BY 4.0 y mantiene su propia licencia.
