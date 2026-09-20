# Estudio 03: ablación de los parámetros derivados del magnetograma LoS

## Pregunta

El estudio 02 identificó 5 parámetros que se calculan sobre el mismo magnetograma de línea de
visión que ve la rama visual. Si son ellos los que provocan la redundancia entre ramas,
quitarlos debería **abrir espacio** a que la fusión aporte algo.

## Qué se hizo

Tres variantes de la rama física, todas con la misma arquitectura, protocolo, splits e
hiperparámetros que el modelo del paper. Lo único que cambia es el conjunto de parámetros:

| Variante | Parámetros | Dataset |
|---|---|---|
| BiLSTM-17 | Los del paper: 12 vectoriales + 5 del LoS | `dataset_temporal_v3.py` |
| BiLSTM-21 | Los 21, sin ninguna preselección | `dataset_temporal_v5.py` |
| BiLSTM-16 | Los 16 vectoriales (quita los 5 del LoS, reincorpora los 4 de baja d) | `dataset_temporal_v6.py` |
| BiLSTM-12 | Solo los 12 vectoriales de la selección original | `dataset_temporal_v7.py` |

El **BiLSTM-12 es la ablación pura**: es exactamente el BiLSTM-17 menos los 5 parámetros del
LoS, sin cambiar nada más. El BiLSTM-16 mezclaba dos cambios a la vez y no permitía atribuir
el efecto.

## Cómo ejecutarlo

Reentrenar (requiere GPU, unos 20 minutos por variante y horizonte):

```bash
export SFMM_DATA=/ruta/a/SFF_MagSeq_MViTs
bash studies/03_los_derived_ablation/train_run_lstm12_cv.sh
```

Analizar (sin GPU, con los logits incluidos):

```bash
python studies/03_los_derived_ablation/comparar_lstm_variantes.py   # las tres variantes
python studies/03_los_derived_ablation/ablacion_hijos.py            # ablación pura
python analysis/metricas_variantes.py                               # métricas completas
python analysis/fusion_variantes.py                                 # los 6 métodos de fusión
```

## Resultado

Quitar los 5 parámetros **empeora** la rama física, de forma significativa:

| | 24 h | 48 h |
|---|---|---|
| BiLSTM-17 | 0.758 | 0.841 |
| BiLSTM-12 | 0.721 | 0.771 |

La pérdida se concentra casi toda en **falsas alarmas** (48 h: 621 a 1359 falsos positivos),
mientras que POD y AUC apenas se mueven.

Y lo decisivo: el techo de la fusión **no se abre**. A 48 h el margen pasa de +0.0004 a +0.0012,
y los seis métodos de fusión evaluados (ensemble ponderado, stacking y las reglas media,
producto, máximo y mínimo) siguen siendo significativamente peores que la mejor rama
individual.

A 24 h las reglas media y producto superan a ambas ramas en el ensemble de folds (+0.008 y
+0.009, significativos), pero solo en 2 de los 5 folds y sin alcanzar lo que el BiLSTM-17 ya
daba por sí solo (0.758). Es un indicio débil, no un resultado.

## Conclusión

La redundancia entre ramas **no viene de que unos parámetros se deriven de la imagen**. Viene
de que ambas modalidades miden manifestaciones de la misma causa física. Quitar los
parámetros compartidos no libera información complementaria; solo empeora la rama física.

La conclusión del paper se mantiene.

Informes: `results/comparar_lstm_variantes.txt`, `results/ablacion_hijos.txt`,
`results/metricas_variantes.txt`, `results/fusion_variantes.txt`
