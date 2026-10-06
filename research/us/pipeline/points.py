"""One weather point per NYISO load zone (name: zone, lat, lon, primary).
Mostly the airport weather station of the zone's main load centre, because NYISO's own load
forecast is driven by airport stations. NORTH gets two points (Plattsburgh = residential load
centre, primary; Massena = industrial load at the St Lawrence), the rest one each. GFS runs on a
~0.25 degree grid (~25 km), so the three downstate points near NYC (N.Y.C., DUNWOD, MILLWD) may
fall in neighbouring or identical grid cells; that is a property of the source, not a bug."""
POINTS = {
    "buffalo":      ("WEST",   42.94, -78.74, True),   # BUF airport
    "rochester":    ("GENESE", 43.12, -77.68, True),   # ROC airport
    "syracuse":     ("CENTRL", 43.11, -76.10, True),   # SYR airport
    "plattsburgh":  ("NORTH",  44.65, -73.47, True),   # PBG airport
    "massena":      ("NORTH",  44.94, -74.85, False),  # MSS airport, secondary
    "utica":        ("MHK VL", 43.10, -75.23, True),   # Utica city
    "albany":       ("CAPITL", 42.75, -73.80, True),   # ALB airport
    "poughkeepsie": ("HUD VL", 41.63, -73.88, True),   # POU airport
    "millwood":     ("MILLWD", 41.19, -73.80, True),   # Millwood / Ossining, northern Westchester
    "white_plains": ("DUNWOD", 41.07, -73.71, True),   # HPN airport, southern Westchester
    "nyc":          ("N.Y.C.", 40.71, -74.01, True),   # lower Manhattan (URL verified 6 Oct)
    "islip":        ("LONGIL", 40.79, -73.10, True),   # ISP airport, central Long Island
}
