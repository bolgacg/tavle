#!/bin/bash
# v2: NYISO outSched monthly zips Jan 2010 to Dec 2019, same skip-if-present pattern as v1 fetch_extra.sh.
set -u
cd ~/nyiso-us/raw
ok_zip() { python3 -I -c "import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); sys.exit(1 if z.testzip() or not z.namelist() else 0)" "$1" 2>/dev/null; }
for ym in $(python3 -c "
y,m=2010,1
while (y,m)<=(2019,12):
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
echo "OUTSCHED DONE $(ls 201[0-9]*outSched_csv.zip | wc -l) files"
