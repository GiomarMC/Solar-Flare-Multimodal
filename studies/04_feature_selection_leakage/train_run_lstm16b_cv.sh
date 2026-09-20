#!/usr/bin/env bash
# Cross-validation completa: BiLSTM-16b standalone — 24h y 48h, 5 folds cada uno.
#
# Respuesta al Revisor #1 de SIMBig. Los 17 SHARP del paper (dataset_temporal_v3)
# menos MEANGBZ -> 16 (dataset_temporal_v8). Mismos hiperparámetros, protocolo y
# splits que el BiLSTM-17: la única diferencia es ese parámetro.
#
# Motivo: al recalcular la d de Cohen solo con los folds de entrenamiento
# (graficos/cohen_d_por_fold.py), MEANGBZ cae bajo el umbral de 0.6 (0.49-0.58 por
# fold vs 0.63 en el dataset completo), asi que un criterio sin fuga tambien lo
# habria descartado. Si el TSS no cambia, la objecion queda cerrada.
#
# Al terminar: guarda logits y genera graficos/verificacion_meangbz.txt
# (metricas completas + bootstrap pareado vs BiLSTM-17 + los 6 metodos de fusion).
#
# Reanuda automáticamente: si ya hay checkpoints para un fold, salta el entrenamiento.
# Bloquea suspensión/cierre de tapa mientras corre.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

if [[ "${SFMM_INHIBITED:-0}" != "1" ]]; then
  exec env SFMM_INHIBITED=1 systemd-inhibit \
      --what=sleep:idle:handle-lid-switch \
      --why="CV BiLSTM-16b (sin MEANGBZ) en curso" \
      --who="sfmm-lstm16b" \
      --mode=block \
      bash "$0" "$@"
fi

RESULTS_FILE="outputs/cv_lstm16b_results.txt"
mkdir -p outputs

{
  echo "BiLSTM-16b (los 17 del paper sin MEANGBZ) — Cross-validation 5 folds, 24h y 48h"
  echo "$(date)"
  echo "========================================"
} > "$RESULTS_FILE"

for H in 24 48; do
  for k in 0 1 2 3 4; do
    CFG="configs/lstm16b_cv_${H}h_k${k}.yaml"
    CKPT_DIR="outputs/checkpoints_lstm16b_cv_${H}h_k${k}"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  BiLSTM-16b  ${H}h  FOLD k=${k}   $(date '+%H:%M:%S')"
    echo "════════════════════════════════════════════════"

    if compgen -G "${CKPT_DIR}/*.ckpt" > /dev/null; then
      echo "  Ya hay checkpoints en ${CKPT_DIR} — se salta el entrenamiento."
    else
      python "$REPO"/scripts/train_lstm16b_standalone.py --config "$CFG"
    fi

    echo "── ${H}h k=${k} entrenado" >> "$RESULTS_FILE"
  done
done

echo ""
echo "Entrenamiento completo. Guardando logits..."
python "$REPO"/scripts/save_logits_lstm16b.py | tee -a "$RESULTS_FILE"

echo ""
echo "Informe de la verificación (métricas + bootstrap pareado + 6 métodos de fusión)..."
python graficos/verificacion_meangbz.py 2>&1 | grep -v -i warning | tee -a "$RESULTS_FILE"

echo ""
echo "Listo $(date)."
echo "  graficos/verificacion_meangbz.txt"
