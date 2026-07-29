import os
from PIL import Image, ImageDraw, ImageFont

def get_fallback_font(size):
    # Try common fonts across Windows, macOS, and Linux
    font_options = ["arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Helvetica.ttc"]
    for font_name in font_options:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()

def create_banner():
    # 1. Setup Canvas and Colors
    width, height = 1280, 400
    bg_color = (13, 17, 23)        # #0D1117 (GitHub Dark Mode)
    border_color = (48, 54, 61)    # #30363D
    card_bg = (22, 27, 34)         # #161B22
    
    banner = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(banner)

    # 2. Draw outer container card
    margin = 20
    draw.rounded_rectangle(
        [margin, margin, width - margin, height - margin],
        radius=16,
        fill=card_bg,
        outline=border_color,
        width=2
    )

    # 3. Add Typography
    title = "Brain MRI Classification"
    subtitle = "End-to-End Medical Deep Learning Pipeline"
    tech_stack = ["PyTorch", "FastAPI", "Docker", "Grad-CAM", "Transfer Learning"]

    font_title = get_fallback_font(52)
    font_sub = get_fallback_font(22)
    font_badge = get_fallback_font(14)

    draw.text((60, 90), title, fill=(255, 255, 255), font=font_title)
    draw.text((60, 160), subtitle, fill=(139, 148, 158), font=font_sub)

    # 4. Draw Tech Badges
    x_offset = 60
    y_offset = 240
    badge_colors = [(238, 76, 44), (0, 150, 136), (36, 150, 237), (245, 158, 11), (168, 85, 247)]

    for i, tech in enumerate(tech_stack):
        # Estimate width based on character count
        badge_width = len(tech) * 9 + 40 
        draw.rounded_rectangle(
            [x_offset, y_offset, x_offset + badge_width, y_offset + 36],
            radius=18,
            fill=(33, 38, 45),
            outline=(48, 54, 61),
            width=1
        )
        # Colored indicator dot
        draw.ellipse([x_offset + 14, y_offset + 14, x_offset + 22, y_offset + 22], fill=badge_colors[i % len(badge_colors)])
        
        # Text inside badge
        draw.text((x_offset + 30, y_offset + 9), tech, fill=(201, 209, 217), font=font_badge)
        x_offset += badge_width + 12

    # 5. Draw Abstract Neural Network Art on the Right Side
    # Base coordinates for network nodes
    nodes = [
        (950, 100), (950, 200), (950, 300),  # Input layer
        (1050, 120), (1050, 190), (1050, 280), # Hidden layer 1
        (1150, 150), (1150, 250)             # Output layer
    ]
    
    # Draw connecting lines (cyan glow effect)
    for i in range(3):
        for j in range(3, 6):
            draw.line([nodes[i], nodes[j]], fill=(13, 110, 253), width=2) # Blue lines
    for j in range(3, 6):
        for k in range(6, 8):
            draw.line([nodes[j], nodes[k]], fill=(56, 189, 248), width=2) # Cyan lines
            
    # Draw nodes (circles)
    for x, y in nodes:
        r = 6
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(22, 27, 34), outline=(56, 189, 248), width=3)

    # 6. Ensure Output Directory Exists and Save
    os.makedirs("assets/readme", exist_ok=True)
    output_path = "assets/banner.png"
    banner.save(output_path)
    print(f"✅ Success! Banner saved to: {output_path}")

if __name__ == "__main__":
    create_banner()