# Estudio 02: procedencia de los parámetros SHARP

## Pregunta

Si los parámetros SHARP se derivan de las mismas imágenes que ve la rama visual, la fusión no
podría aportar nada: sería combinar un dato con una función de ese mismo dato.

La objeción es seria, así que había que resolverla con las fuentes primarias del instrumento,
no con intuiciones.

## Qué se hizo

Se rastreó el procesamiento desde el instrumento, con la documentación de HMI/SDO
(Bobra et al. 2014, Hoeksema et al. 2014, Bobra et al. 2021).

La bifurcación ocurre en el **vector de Stokes**, antes de que exista ninguna imagen:

- **Camino de línea de visión (LoS)**: solo la polarización circular (I+-V), algoritmo tipo MDI,
  produce `hmi.M_720s`, se recorta al HARP y da el segmento `magnetogram`. **Esas son las
  imágenes FITS que ve la rama visual.**
- **Camino vectorial**: el Stokes completo (I, Q, U, V), incluidas las polarizaciones lineales
  Q y U que miden el campo transversal, pasa por la inversión Milne-Eddington (VFISV), la
  desambiguación de azimut a 180 grados y el remapeo CEA. Sobre ese campo vectorial el módulo
  SHARP calcula los índices.

Clasificación de los 17 parámetros que usa el paper:

- **5 derivados del magnetograma LoS**: USFLUXL, MEANGBL, R_VALUE, AREA_ACR, NACR
- **12 derivados del campo vectorial**: el resto

Los 12 vectoriales necesitan Q y U, es decir información que **no está** en el magnetograma de
línea de visión.

## Cómo ejecutarlo

```bash
# Correlación entre cada parámetro SHARP y estadísticos de la imagen
python studies/02_sharp_provenance/redundancia_sharp_imagen.py

# Figura del recorrido del dato desde el instrumento hasta las dos ramas
python studies/02_sharp_provenance/fig_procedencia_datos.py
```

## Resultado

La redundancia existe, pero **no viene de la derivación**. Los parámetros vectoriales
*extensivos* (sumas sobre la región) correlacionan fuerte con la imagen de todos modos,
porque ambos dependen del tamaño y el flujo de la región activa:

| Parámetro | Correlación con la imagen | Tipo |
|---|---|---|
| TOTUSJZ | 0.930 | vectorial, extensivo |
| TOTUSJH | 0.925 | vectorial, extensivo |
| MEANJZH | 0.038 | vectorial, intensivo |
| MEANALP | 0.059 | vectorial, intensivo |

Los *intensivos* (promedios) son casi independientes de la imagen. Y son justamente los que la
preselección por d de Cohen descartó, por su baja capacidad discriminativa individual.

## Conclusión

La premisa "los SHARP salen de las imágenes" es falsa para 12 de los 17 parámetros. Pero la
conclusión que se derivaba de ella resulta ser cierta por otro motivo: la redundancia entre
las dos ramas nace de una **causa física común**, el tamaño y el flujo de la región activa, no
de una relación de derivación.

El estudio 03 comprueba esto experimentalmente.

Informe: `results/redundancia_sharp_imagen.txt`
