# Estudio 01: ¿está limitada la rama física por su arquitectura?

## Pregunta

A 48 horas la rama física queda por debajo de la visual. ¿Es un límite real de la información
que aportan los parámetros SHARP, o simplemente una BiLSTM demasiado pequeña?

Si el modelo estuviera infradimensionado, la comparación entre modalidades del paper sería
injusta.

## Qué se hizo

Barrido completo de la rama física: tamaños ocultos {32, 64, 128} y profundidades {1, 2}, o
sea seis configuraciones, entre 19.6 K y 695 K parámetros. Todo el barrido se repitió
sustituyendo la celda BiLSTM por una BiGRU, en los dos horizontes y los 5 folds.

En total **120 entrenamientos**.

## Cómo ejecutarlo

Requiere GPU y el dataset base (ver `data/README.md`).

```bash
export SFMM_DATA=/ruta/a/SFF_MagSeq_MViTs

python studies/01_recurrent_architecture/train_lstm_ablation.py        # BiLSTM, 48 h
python studies/01_recurrent_architecture/train_lstm_ablation_24h.py    # BiLSTM, 24 h
python studies/01_recurrent_architecture/train_gru_ablation.py         # BiGRU, 48 h
python studies/01_recurrent_architecture/train_gru_ablation_24h.py     # BiGRU, 24 h

python studies/01_recurrent_architecture/collect_ablation_bilstm.py    # tabla resumen
```

## Resultado

El rendimiento es notablemente plano:

| Horizonte | BiLSTM | BiGRU |
|---|---|---|
| 48 h | TSS 0.806 - 0.815 | 0.787 - 0.818 |
| 24 h | TSS 0.717 - 0.745 | 0.717 - 0.747 |

Las desviaciones se solapan en todo el barrido. Ni la profundidad, ni el ancho, ni el tipo de
celda mejoran la configuración base (64 unidades, 1 capa), y los modelos más pequeños son
algo más estables.

El fold k=1 es el más bajo en **todas** las configuraciones, lo que indica que su variabilidad
es una propiedad de la partición cronológica y no del modelo.

## Conclusión

La rama física necesita poca capacidad y no está limitada por la arquitectura. Su distancia
con la rama visual a 48 horas no se explica por un modelo infradimensionado.

Informe: `results/ablation_bilstm_48h.txt`
