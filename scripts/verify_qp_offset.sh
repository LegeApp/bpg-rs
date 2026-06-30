#!/usr/bin/env bash
# Lock the bpgenc-vs-still265 QP-mapping assumption.
#
# bpgenc `-qN`  -> actual HEVC QP (N - 3)   (offset inside x265 placebo+tune-ssim)
# still265 `-qN` -> actual HEVC QP N         (1:1)
#
# Verified by back-solving the DC dequant scale of the top-left 32x32 luma TU from
# a bit-exact decode. bpgenc -q{N+3} and still265 -q{N} must produce the SAME
# actual HEVC QP (and bit-identical DC levels). Exits non-zero on mismatch.
set -euo pipefail
cd "$(dirname "$0")/.."

IMG="${1:-test-set/test-set-4mp/20260606_201839_4mp.png}"
TOOLS=target/release/bpg-tools
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
OFFSET=3

# Infer actual HEVC QP from the DC dequant/level ratio of the (0,0) 32x32 TU.
infer_qp() {
  BPG_STILL_DUMP=0,0,5,0 "$TOOLS" decode "$1" -o /dev/null 2>&1 | python3 -c '
import sys
lvl=deq=None; sec=None; r=0
for line in sys.stdin:
    line=line.rstrip()
    if line=="[LEVELS]": sec="L"; r=0; continue
    if line=="[DEQUANT]": sec="D"; r=0; continue
    if line.startswith("["): sec=None; continue
    if sec=="L" and r==0: lvl=int(line.split()[0])
    if sec=="D" and r==0: deq=int(line.split()[0]); break
    if sec in ("L","D"): r+=1
if not lvl: print("ERR"); sys.exit(0)
ratio=deq/lvl; ls=[40,45,51,57,64,72]; best=None
for qp in range(0,52):
    exp=(ls[qp%6]<<(qp//6))/16.0
    if best is None or abs(exp-ratio)<best[1]: best=(qp,abs(exp-ratio))
print(f"{best[0]} {lvl}")'
}

fail=0
echo "image: $IMG   (bpgenc -q{N+$OFFSET}  ==  still265 -q{N})"
printf "%-10s %-22s %-22s %s\n" "actual QP" "bpgenc(nominal/QP/DC)" "still265(nominal/QP/DC)" "match"
for AQ in 17 21 24 27 29; do
  CQ=$((AQ + OFFSET))
  bpgenc -e x265 -m 9 -q "$CQ" -f 420 -o "$TMP/c.bpg" "$IMG" 2>/dev/null
  "$TOOLS" encode "$IMG" -o "$TMP/r.bpg" --effort best -q "$AQ" 2>/dev/null
  read -r cqp cdc <<<"$(infer_qp "$TMP/c.bpg")"
  read -r rqp rdc <<<"$(infer_qp "$TMP/r.bpg")"
  ok="OK"
  if [ "$cqp" != "$AQ" ] || [ "$rqp" != "$AQ" ] || [ "$cdc" != "$rdc" ]; then ok="MISMATCH"; fail=1; fi
  printf "%-10s %-22s %-22s %s\n" "$AQ" "q$CQ/QP$cqp/DC$cdc" "q$AQ/QP$rqp/DC$rdc" "$ok"
done
if [ "$fail" -ne 0 ]; then echo "FAIL: QP-offset assumption (==$OFFSET) violated"; exit 1; fi
echo "PASS: bpgenc -q maps to HEVC QP (q-$OFFSET); still265 -q is 1:1; quant bit-exact at equal actual QP."
