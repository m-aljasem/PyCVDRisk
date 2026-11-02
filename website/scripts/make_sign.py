"""A Massachusetts-style green town-line sign for Framingham (illustration; Commons has no photo of one).
Run from website/: python scripts/make_sign.py public/images/regions/framingham_sign.jpg"""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
W, H = 1200, 800
img = Image.new("RGB", (W, H)); d = ImageDraw.Draw(img)
for y in range(H):  # sky gradient
    t = y / H
    d.line([(0, y), (W, y)], fill=(int(70 + 110 * t), int(130 + 90 * t), int(210 + 35 * t)))
d.rectangle([0, 690, W, H], fill=(88, 128, 60))  # grass verge
d.rectangle([0, 740, W, H], fill=(70, 70, 74))  # road
for x in (420, 780):  # posts
    d.rectangle([x - 12, 500, x + 12, 720], fill=(150, 152, 156))
F = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
big, small = ImageFont.truetype(F, 104), ImageFont.truetype(F, 58)
x0, y0, x1, y1 = 150, 230, 1050, 530
shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
ImageDraw.Draw(shadow).rounded_rectangle([x0 + 14, y0 + 16, x1 + 14, y1 + 16], 34, fill=(0, 0, 0, 90))
img.paste(shadow.filter(ImageFilter.GaussianBlur(10)), (0, 0), shadow.filter(ImageFilter.GaussianBlur(10)))
d.rounded_rectangle([x0, y0, x1, y1], 34, fill=(0, 112, 60))
d.rounded_rectangle([x0 + 14, y0 + 14, x1 - 14, y1 - 14], 24, outline="white", width=8)
d.text((W / 2, 350), "FRAMINGHAM", font=big, fill="white", anchor="mm")
d.text((W / 2, 455), "TOWN LINE", font=small, fill="white", anchor="mm")
img.save(sys.argv[1], quality=88)
