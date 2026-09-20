#!/usr/bin/env bash
# Cross-validation completa: BiLSTM-12 standalone — 24h y 48h, 5 folds cada uno.
#
# Ablación PURA de los parámetros derivados del magnetograma LoS: los 17 SHARP del
# paper (dataset_temporal_v3) menos USFLUXL, MEANGBL, R_VALUE, AREA_ACR y NACR
# -> 12 vectoriales (dataset_temporal_v7). Mismos hiperparámetros, protocolo y
# splits que el BiLSTM-17: la única diferencia son esos 5 parámetros.
#
# Al terminar: guarda logits, recalcula el techo (diversidad_ramas) y genera el
# informe de la ablación (graficos/ablacion_hijos.txt): métricas completas, techo
# de decisión y los 6 métodos de fusión contra la mejor rama.
#
# Reanuda automáticamente: si ya hay checkpoints para un fold, salta el entrenamiento.
# Bloquea suspensión/cierre de tapa mientras corre.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ "${SFMM_INHIBITED:-0}" != "1" ]]; then
  exec env SFMM_INHIBITED=1 systemd-inhibit \
      --what=sleep:idle:handle-lid-switch \
      --why="CV BiLSTM-12 en curso" \
      --who="sfmm-lstm12" \
      --mode=block \
      bash "$0" "$@"
fi

RESULTS_FILE="outputs/cv_lstm12_results.txt"
mkdir -p outputs

{
  echo "BiLSTM-12 (17 del paper sin los 5 derivados del LoS) — Cross-validation 5 folds, 24h y 48h"
  echo "$(date)"
  echo "========================================"
} > "$RESULTS_FILE"

for H in 24 48; do
  for k in 0 1 2 3 4; do
    CFG="configs/lstm12_cv_${H}h_k${k}.yaml"
    CKPT_DIR="outputs/checkpoints_lstm12_cv_${H}h_k${k}"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  BiLSTM-12  ${H}h  FOLD k=${k}   $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════"

    if compgen -G "${CKPT_DIR}/*.ckpt" > /dev/null; then
      echo "  Ya hay checkpoints en ${CKPT_DIR} — se salta el entrenamiento."
    else
      python "$REPO"/scripts/train_lstm12_standalone.py --config "$CFG"
    fi

    echo "── ${H}h k=${k} entrenado" >> "$RESULTS_FILE"
  done
done

echo ""
echo "Entrenamiento completo. Guardando logits..."
python "$REPO"/scripts/save_logits_lstm12.py | tee -a "$RESULTS_FILE"

echo ""
echo "Recalculando el techo de fusión con la rama física de 12 params..."
LSTM_MODEL=lstm12 python graficos/diversidad_ramas.py > /dev/null

echo ""
echo "Informe de la ablación pura (métricas + techo + 6 métodos de fusión)..."
python graficos/ablacion_hijos.py 2>&1 | grep -v -i warning | tee -a "$RESULTS_FILE"

echo ""
echo "Listo $(date)."
echo "  graficos/ablacion_hijos.txt"
echo "  graficos/diversidad_ramas_lstm12_{24,48}h.txt"
