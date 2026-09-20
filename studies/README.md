# Estudios de ablación

Cuatro estudios independientes. Cada carpeta contiene su propio README, el código que le es
propio, sus configuraciones y los resultados que produjo.

| Estudio | Pregunta | Veredicto |
|---|---|---|
| `01_recurrent_architecture` | ¿Limita la arquitectura recurrente a la rama física? | No. El rendimiento es plano en 120 configuraciones |
| `02_sharp_provenance` | ¿Qué parámetros SHARP salen del mismo magnetograma que ve la rama visual? | 5 de 17. Los otros 12 necesitan el campo vectorial |
| `03_los_derived_ablation` | Quitando esos 5, ¿aporta algo la fusión? | No. La redundancia es por causa física común, no por derivación |
| `04_feature_selection_leakage` | ¿Filtró información la preselección de variables? | La selección es estable entre folds, con una excepción al borde |

Todos parten de los logits ya incluidos en `results/logits/`, así que los análisis se pueden
repetir sin GPU. Reentrenar las variantes sí requiere GPU y el dataset base.
