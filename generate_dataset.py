from src.plate_generator import PlateGenerator
from pathlib import Path

# مسیر فونت‌ها
fonts = []
font_dir = Path("data/fonts")

if font_dir.exists():
    for font_file in font_dir.glob("*.ttf"):
        fonts.append(str(font_file))

if not fonts:
    print("No fonts found in data/fonts/")
    print("Using default font...")
    fonts = [""]

# مسیر تمپلیت‌ها
templates = []
template_dir = Path("data/templates")

if template_dir.exists():
    for template_file in template_dir.glob("*.png"):
        templates.append(str(template_file))

print(f"Found {len(fonts)} fonts")
print(f"Found {len(templates)} templates")

# تولید دیتاست
generator = PlateGenerator(fonts, templates)
generator.generate_dataset("data/generated_plates", num_samples=1000)

print("Dataset generation completed!")