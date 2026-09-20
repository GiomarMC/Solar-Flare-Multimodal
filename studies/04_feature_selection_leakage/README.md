# Estudio 04: ¿filtró información la preselección de variables?

## Pregunta

El paper descarta 4 de los 21 parámetros SHARP (MEANGBH, MEANJZH, MEANJZD, MEANALP) por su
baja d de Cohen, **calculada una sola vez sobre el dataset completo**.

Un revisor señaló que esa d debería calcularse dentro de cada fold de entrenamiento. De lo
contrario la selección usa, indirectamente, información de los periodos de validación y test.

## Qué se hizo

Dos pasos, uno barato y otro caro.

**Paso 1, sin reentrenar.** Recalcular la d de Cohen usando **solo los folds de entrenamiento**
de cada partición y comprobar si la selección resultante cambia.

Antes hubo que calibrar la metodología, porque el paper no dejó el script: la agregación
correcta es la **media de los 16 fotogramas** de cada secuencia. Reproduce los valores citados
(MEANGBH 0.136 frente a 0.14; MEANJZD 0.49-0.54 frente a 0.53). Tomar solo el último fotograma
no los reproduce.

**Paso 2, reentrenando.** El paso 1 dejó un caso en la frontera, así que se entrenó la variante
sin ese parámetro: BiLSTM-16b, que es exactamente el BiLSTM-17 del paper **menos MEANGBZ**
(`dataset_temporal_v8.py`). 10 entrenamientos, 5 folds por 2 horizontes.

## Cómo ejecutarlo

Paso 1, sin GPU pero con el dataset base (lee los parámetros SHARP crudos):

```bash
export SFMM_DATA=/ruta/a/SFF_MagSeq_MViTs
python studies/04_feature_selection_leakage/cohen_d_por_fold.py
```

Paso 2, reentrenar (GPU, unos 15 minutos) y analizar:

```bash
bash studies/04_feature_selection_leakage/train_run_lstm16b_cv.sh
python studies/04_feature_selection_leakage/verificacion_meangbz.py
```

## Resultado

**La selección es estable.** En los 5 folds y en los dos horizontes se descartan y se retienen
exactamente los mismos parámetros. Ninguno cambia de lado según qué fold se deje fuera, así
que la preselección no depende del periodo excluido.

Los cuatro descartados lo son con margen cómodo en las 10 combinaciones fold-horizonte
(d entre 0.01 y 0.52, frente al umbral de 0.6).

**Con una excepción en la frontera: MEANGBZ.**

| | Dataset completo | Solo folds de entrenamiento |
|---|---|---|
| 48 h | 0.634 (se conserva) | 0.494 - 0.580 (se descartaría) |
| 24 h | 0.589 | 0.484 - 0.558 |

Un criterio estrictamente interno al fold también lo habría descartado, dejando 16 parámetros
en lugar de 17.

**Y quitarlo empeora**, en contra de lo esperado:

| | 24 h | 48 h |
|---|---|---|
| BiLSTM-17 | 0.7583 | 0.8412 |
| BiLSTM-16b | 0.7225 | 0.8177 |

Ambas diferencias son significativas. Pero el mecanismo importa: el **AUC apenas cambia**
(0.9200 a 0.9174 y 0.9627 a 0.9579) y con umbral óptimo la rama física conserva su ventaja a
24 h. Lo que se rompe es la transferencia del umbral de validación a test: la desviación entre
folds se triplica, de 0.013 a 0.040.

MEANGBZ no aporta capacidad de discriminación, aporta **estabilidad**.

## Conclusión

A **48 horas**, que es el resultado principal del paper, no cambia nada: la rama visual sigue
siendo significativamente superior y ninguna fusión la mejora.

A **24 horas** desaparece la ventaja de la rama física en media de validación cruzada
(0.7096 frente a 0.7140 de la visual), una diferencia que el test pareado ya reportaba como no
significativa. El paper lo declara explícitamente.

Informes: `results/cohen_d_por_fold.txt`, `results/verificacion_meangbz.txt`
