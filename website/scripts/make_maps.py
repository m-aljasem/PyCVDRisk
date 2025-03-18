"""Country maps on NASA's satellite mosaic: SCORE2 risk regions (Figure 5) and SCORE (2003) cohort countries.
Run from website/: python scripts/make_maps.py GEO public/images/regions ../reference/uniscore
GEO holds world.jpg (commons.wikimedia.org/wiki/File:Whole_world_-_land_and_oceans.jpg, NASA, public domain)
and ne50.geojson (Natural Earth ne_50m_admin_0_countries, public domain)."""
import csv, json, math, sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
G, OUT, ROUTING = sys.argv[1], sys.argv[2], sys.argv[3]
world = Image.open(f"{G}/world.jpg").convert("RGB")
W, H = world.size
feats = json.load(open(f"{G}/ne50.geojson"))["features"]
def iso(f):
    p = f["properties"]
    return p["ISO_A3"] if p["ISO_A3"] != "-99" else p["ADM0_A3"]

def draw(colours, box, out, lat0=50, outw=1200):
    lon0, lat_top, lon1, lat_bot = box
    px = W / 360
    crop = world.crop((int((lon0 + 180) * px), int((90 - lat_top) * px), int((lon1 + 180) * px), int((90 - lat_bot) * px)))
    sx = math.cos(math.radians(lat0))  # equirectangular with a standard parallel, so Europe is not stretched
    s = outw / crop.width / sx
    base = crop.resize((outw, round(crop.height * s)), Image.LANCZOS)
    xy = lambda lon, lat: ((lon - lon0) * px * s * sx, (lat_top - lat) * px * s)
    fill = Image.new("RGBA", base.size, (0, 0, 0, 0)); fd = ImageDraw.Draw(fill)
    line = Image.new("RGBA", base.size, (0, 0, 0, 0)); ld = ImageDraw.Draw(line)
    for f in feats:
        c = colours.get(iso(f))
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            ring = [xy(*pt) for pt in poly[0]]
            if c:
                fd.polygon(ring, fill=c + (130,))
            ld.line(ring + ring[:1], fill=(255, 255, 255, 200 if c else 70), width=2 if c else 1)
    img = Image.alpha_composite(Image.alpha_composite(base.convert("RGBA"), fill), line).convert("RGB")
    img.save(out, quality=85)
    print(out, img.size)

# ESC SCORE2 chart colours
REG = {"low": (46, 160, 67), "moderate": (240, 200, 30), "high": (240, 120, 30), "very_high": (210, 30, 40)}
sys.path.insert(0, ROUTING); from build_routing import SCORE2_FIG5, ISO3  # SCORE2 Figure 5, as the repo records it
score2 = {ISO3[n]: REG[r] for r, names in SCORE2_FIG5.items() for n in names}
print(len(score2), "SCORE2 countries")
draw(score2, (-25, 72, 62, 25), f"{OUT}/score2_regions.jpg")
# SCORE (Conroy 2003): the 12 cohorts came from these 11 countries (England and Scotland both UK)
cohorts = "FIN RUS NOR GBR DNK SWE BEL DEU ITA FRA ESP".split()
draw({c: (0, 140, 200) for c in cohorts}, (-25, 72, 62, 22), f"{OUT}/score_cohorts.jpg")
