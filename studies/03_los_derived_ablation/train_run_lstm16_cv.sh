#!/usr/bin/env bash
# Cross-validation completa: BiLSTM-16 standalone — 24h y 48h, 5 folds cada uno.
#
# Variante de 16 parámetros SHARP vectoriales (dataset_temporal_v6): los 21 menos
# los 5 que se calculan sobre el mismo magnetograma LoS que ve la rama visual
# (USFLUXL, MEANGBL, R_VALUE, AREA_ACR, NACR).
# Objetivo: responder la objeción "padre-hijo" — si sin esos 5 el techo de fusión
# sigue en ~0, la redundancia es por causa física común, no por derivación.
#
# Al terminar: guarda logits, recalcula el techo (diversidad_ramas) y compara las
# tres variantes de la rama física (17 / 21 / 16).
#
# Reanuda automáticamente: si ya hay checkpoints para un fold, salta el entrenamiento.
# Bloquea suspensión/cierre de tapa mientras corre.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ "${SFMM_INHIBITED:-0}" != "1" ]]; then
  exec env SFMM_INHIBITED=1 systemd-inhibit \
      --what=sleep:idle:handle-lid-switch \
      --why="CV BiLSTM-16 en curso" \
      --who="sfmm-lstm16" \
      --mode=block \
      bash "$0" "$@"
fi

RESULTS_FILE="outputs/cv_lstm16_results.txt"
mkdir -p outputs

{
  echo "BiLSTM-16 (16 params SHARP vectoriales) — Cross-validation 5 folds, 24h y 48h"
  echo "$(date)"
  echo "========================================"
} > "$RESULTS_FILE"

for H in 24 48; do
  for k in 0 1 2 3 4; do
    CFG="configs/lstm16_cv_${H}h_k${k}.yaml"
    CKPT_DIR="outputs/checkpoints_lstm16_cv_${H}h_k${k}"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  BiLSTM-16  ${H}h  FOLD k=${k}   $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════"

    if compgen -G "${CKPT_DIR}/*.ckpt" > /dev/null; then
      echo "  Ya hay checkpoints en ${CKPT_DIR} — se salta el entrenamiento."
    else
      python "$REPO"/scripts/train_lstm16_standalone.py --config "$CFG"
    fi

    echo "── ${H}h k=${k} entrenado" >> "$RESULTS_FILE"
  done
done

echo ""
echo "Entrenamiento completo. Guardando logits..."
python "$REPO"/scripts/save_logits_lstm16.py | tee -a "$RESULTS_FILE"

echo ""
echo "Recalculando el techo de fusión con la rama física de 16 params..."
LSTM_MODEL=lstm16 python graficos/diversidad_ramas.py > /dev/null

echo ""
echo "Comparando las tres variantes de la rama física..."
python graficos/comparar_lstm_variantes.py | tee -a "$RESULTS_FILE"

echo ""
echo "Listo $(date)."
echo "  graficos/diversidad_ramas_lstm16_{24,48}h.txt"
echo "  graficos/comparar_lstm_variantes.txt"
