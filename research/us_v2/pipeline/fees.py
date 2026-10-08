"""v2 fees: v1 model/fees.py extended back to 2010. The v1 module is executed into this namespace, then the
yearly tables gain 2010-2019 and RATES / SUPPLY_RATES are recomputed. fee(), fee_for(), supply_fee_for(),
cost_for() are v1's functions and read the extended tables. Put this folder before research/us/model on
sys.path (features_v2.py does) so `import fees` resolves here.

RS1 virtual (non-physical) rate, USD per cleared virtual MWh (OATT Rate Schedule 1, 6.1.2.4). Virtual
transactions paid no Schedule 1 charge before 1 Jan 2010 (BAWG deck, Jan 2010: "These charges became
effective on January 1, 2010"). URLs are under https://www.nyiso.com/documents/20142/ :
  2010 0.065   BAWG "OATT Schedule 1 (S, SC, & D)", Jan 2010 (1392923/BAWG_Schedule_One_presentation.pdf):
               "Tariff defined rate of $0.065/ MWh". A Nov 2010 draft budget deck shows 0.054; the higher,
               charged value is used.
  2011 0.0871  NOT FOUND; nearest year. 2010 and 2012 are equally near; the higher (2012) is used, which is
               conservative (the Nov 2010 draft 2011 budget deck shows 0.054).
  2012 0.0871  OATT 6.1.2.4.1: "For calendar year 2012, the applicable rate shall be $0.0871 per cleared MWh
               of Virtual Transactions"
  2013 0.0805  BPWG "2013 Budget vs. Actual", 13 Feb 2014 (1401423): "$0.0805/ Cleared MWh"
  2014 0.0976  BPWG "2014 Budget vs. Actual", 24 Oct 2014 (1408237): "$0.0976/ Cleared MWh"
  2015 0.1046  BPWG "2015 Budget vs. Actual", 13 Jul 2015 (1411330): "$0.1046/ Cleared MWh"
  2016 0.0850  "Schedule 1 Rates for 2016" posting (2959493): "Virtual Resources $0.0850 per cleared MWh"
  2017 0.0649  BPWG "2017 Budget vs. Actual", 26 Apr 2017 (1410270): "Virtual Trading $0.0649/ Cleared MWh"
  2018 0.0636  BPWG "Sept 2018 Budget vs. Actual", 31 Oct 2018 (3690320): "Virtual Trading ... $0.0636/ Cleared MWh"
  2019 0.0795  BPWG "Budget vs. Actual", 30 Oct 2019 (8948507): "Virtual Trading $2.4M $0.0795/ Cleared MWh"
  The 2013 to 2015 slides carry a mislabelled caption; the value was matched to the virtual row by magnitude
  and revenue requirement (research note, 7 Oct 2026). Not yet confirmed against yearly postings.
FERC annual charge per virtual MWh 2010-2019: no per-MWh figure found (2010 posting folds FERC fees into
  the physical rate; 2016 shows none; 2020 "billed separately"). Nearest year used: 2020's 0.010.
Forecast-pass BPCG uplift upper bound on virtual supply 2010-2019: not researched for v2; nearest year
  used: 2020's 0.007. Both nearest-year values are small next to RS1 and the 0.50 / 1.00 stress costs.
"""
from __future__ import annotations

from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_V1 = next(p for p in (_HERE.parent.parent / "us" / "model", _Path.home() / "nyiso-us" / "model")
           if (p / "fees.py").exists())
exec(compile((_V1 / "fees.py").read_text(), str(_V1 / "fees.py"), "exec"), globals())

RS1_V2 = {2010: 0.065, 2011: 0.0871, 2012: 0.0871, 2013: 0.0805, 2014: 0.0976, 2015: 0.1046, 2016: 0.0850,
          2017: 0.0649, 2018: 0.0636, 2019: 0.0795}
RS1.update(RS1_V2)
FERC.update({y: FERC[2020] for y in RS1_V2})
UPLIFT_SUPPLY.update({y: UPLIFT_SUPPLY[2020] for y in RS1_V2})
RATES.clear()
RATES.update({y: round(RS1[y] + FERC[y], 6) for y in sorted(RS1)})
SUPPLY_RATES.clear()
SUPPLY_RATES.update({y: round(RATES[y] + UPLIFT_SUPPLY[y], 6) for y in sorted(RS1)})
SOURCE["rs1"].update({2010: "BAWG deck Jan 2010", 2011: "not found; nearest year (2012), conservative",
                      2012: "OATT 6.1.2.4.1", 2013: "BPWG budget vs actual", 2014: "BPWG budget vs actual",
                      2015: "BPWG budget vs actual", 2016: "posting", 2017: "BPWG budget vs actual",
                      2018: "BPWG budget vs actual", 2019: "BPWG budget vs actual"})
SOURCE["v2_nearest_year"] = "FERC and uplift 2010-2019 = 2020 values (not found / not researched)"
