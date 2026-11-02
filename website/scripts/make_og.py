"""Social preview image (Open Graph / Twitter, 1200x630). Run from website/: python scripts/make_og.py"""
import json
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
A, B = (14, 165, 233), (99, 102, 241)  # hero gradient: sky-500 -> indigo-500
img = Image.new("RGB", (W, H)); d = ImageDraw.Draw(img)
for x in range(W):
    t = x / W
    d.line([(x, 0), (x, H)], fill=tuple(round(a + (b - a) * t) for a, b in zip(A, B)))
logo = Image.open("public/PyCVDRisk-Logo.png").convert("RGBA"); logo.thumbnail((290, 290))
img.paste(logo, (W - logo.width - 40, (H - logo.height) // 2), logo)
F = "/usr/share/fonts/truetype/liberation/LiberationSans-"
n = len(json.load(open("data/models.json"))["models"])
d.text((70, 90), "PYCVDRISK", font=ImageFont.truetype(F + "Bold.ttf", 30), fill=(224, 242, 254))
y = 150
for line in ["Cardiovascular risk scores", "from across the world,", "verified, in Python"]:
    d.text((70, y), line, font=ImageFont.truetype(F + "Bold.ttf", 60), fill="white"); y += 74
d.text((70, y + 30), f"{n} published risk equations · one vectorised API", font=ImageFont.truetype(F + "Regular.ttf", 30), fill=(224, 242, 254))
d.text((70, H - 80), "pip install cvd-risk", font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28), fill="white")
img.save("public/og.png", optimize=True)
