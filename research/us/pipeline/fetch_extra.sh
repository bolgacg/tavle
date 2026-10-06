#!/bin/bash
# Second-wave downloads for the New York study, same skip-if-present pattern as fetch.sh.
#  1. NYISO outSched monthly zips (daily snapshots of scheduled transmission outages), Jan 2020 to Sep 2026.
#  2. Open-Meteo previous-runs archive, GFS temperature_2m_previous_day2, one point per load zone,
#     2021 to 2026-09-30, one JSON per point per year (fetch_gfs.py).
# Validity check uses python zipfile (gene has no unzip binary).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RAW=~/nyiso-us/raw
cd "$RAW"
ok_zip() { python3 -I -c "import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); sys.exit(1 if z.testzip() or not z.namelist() else 0)" "$1" 2>/dev/null; }
for ym in $(python3 -c "
y,m=2020,1
while (y,m)<=(2026,9):
    print(f'{y}{m:02d}01'); m+=1
    if m==13: y,m=y+1,1"); do
  out=${ym}outSched_csv.zip
  [ -s "$out" ] && ok_zip "$out" && continue
  if curl -s -f -m 300 --retry 3 -o "$out.part" "http://mis.nyiso.com/public/csv/outSched/${out}" && ok_zip "$out.part"; then
    mv "$out.part" "$out"; echo "ok $out"
  else
    rm -f "$out.part"; echo "FAIL $out"
  fi
  sleep 1
done
echo "OUTSCHED DONE $(ls *outSched_csv.zip | wc -l) files"
python3 -I "$HERE/fetch_gfs.py"
