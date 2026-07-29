import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. SETUP & UTILS
# ==========================================

def setup_directory():
    """Ensure the target assets directory exists."""
    os.makedirs("assets", exist_ok=True)
    print("✅ Assets directory ready.")

def get_font(size):
    """Helper to load system fonts cleanly across platforms."""
    font_options = ["Helvetica.ttc", "Arial.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"]
    for font_name in font_options:
        try:
            return ImageFont.truetype(font_name, size)
        except IOError:
            continue
    return ImageFont.load_default()

# ==========================================
# 2. DATA PLOTS (300 DPI Matplotlib)
# ==========================================

def generate_confusion_matrix():
    cm = np.array([
        [364.0,   7.0,   0.0,   0.0],
        [ 11.0, 356.0,   0.0,   8.0],
        [  0.0,   0.0, 200.0,   1.0],
        [  0.0,   2.0,   0.0, 359.0]
    ])
    labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
    plt.figure(figsize=(7, 5.5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='viridis', xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title('Finetune Matrix', fontsize=12)
    plt.xlabel('Predicted label', fontsize=10)
    plt.ylabel('True label', fontsize=10)
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: assets/confusion_matrix.png")

def generate_training_curve():
    epochs = list(range(20))
    fe_acc = [0.638, 0.722, 0.736, 0.746, 0.748, 0.753, 0.746, 0.748, 0.752, 0.754, 
              0.748, 0.754, 0.743, 0.749, 0.754, 0.754, 0.756, 0.754, 0.753, 0.759]
    ft_acc = [0.696, 0.844, 0.887, 0.914, 0.933, 0.948, 0.960, 0.969, 0.971, 0.979, 
              0.983, 0.983, 0.986, 0.989, 0.992, 0.993, 0.994, 0.993, 0.994, 0.995]
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(epochs, fe_acc, label='FE')
    plt.plot(epochs, ft_acc, label='FT')
    plt.title('Training Accuracy Comparison', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.tight_layout()
    plt.savefig('assets/training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: assets/training_curve.png")

def generate_model_comparison():
    """Recreates the model comparison and test loss charts side-by-side at true 300 DPI."""
    models = ['CNN Baseline V1', 'CNN Baseline V2', 'ResNet18 Feature Extraction', 'ResNet18 Fine-Tuned']
    x = np.arange(len(models))
    width = 0.15

    # Data from your README table
    acc = [0.8112, 0.9641, 0.8180, 0.9778]
    prec = [0.8273, 0.9643, 0.8205, 0.9804]
    rec = [0.8121, 0.9680, 0.8189, 0.9800]
    f1 = [0.8115, 0.9655, 0.8188, 0.9801]
    loss = [0.528, 0.128, 0.458, 0.066]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # Left Plot: Metrics
    axes[0].bar(x - 1.5*width, acc, width, label='accuracy')
    axes[0].bar(x - 0.5*width, prec, width, label='precision')
    axes[0].bar(x + 0.5*width, rec, width, label='recall')
    axes[0].bar(x + 1.5*width, f1, width, label='f1')
    axes[0].set_title('Model Performance Comparison')
    axes[0].set_ylabel('Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()

    # Right Plot: Loss
    axes[1].bar(x, loss, width=0.4, color='#1f77b4')
    axes[1].set_title('Test Loss Comparison')
    axes[1].set_ylabel('Loss')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('assets/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Generated: assets/model_comparison.png")

# ==========================================
# 3. SCREENSHOT STITCHING
# ==========================================

def process_gradcam_images():
    """Combines the two Grad-CAM screenshots vertically and saves at 300 DPI."""
    img_correct_path = "gradcam_1.png"
    img_wrong_path = "gradcam_2.png"
    output_path = "assets/grad_cam.png"

    try:
        img_correct = Image.open(img_correct_path)
        img_wrong = Image.open(img_wrong_path)
        total_width = max(img_correct.width, img_wrong.width)
        total_height = img_correct.height + img_wrong.height
        
        combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        combined_img.paste(img_correct, (0, 0))
        combined_img.paste(img_wrong, (0, img_correct.height))
        
        combined_img.save(output_path, dpi=(300, 300))
        print(f"✅ Generated: {output_path}")
    except FileNotFoundError:
        print(f"⚠️ Skipped Grad-CAM: Ensure {img_correct_path} and {img_wrong_path} exist in the root folder.")

def process_swagger_images():
    """Combines the four FastAPI screenshots vertically into one 300 DPI image."""
    image_paths = [
        "Screenshot 1405-05-07 at 15.40.03.png",
        "Screenshot 1405-05-07 at 15.41.28.png",
        "Screenshot 1405-05-07 at 15.42.43.png",
        "Screenshot 1405-05-07 at 15.42.59.png"
    ]
    output_path = "assets/swagger.png"

    try:
        # Load all images
        images = [Image.open(p) for p in image_paths]
        
        # Calculate canvas dimensions
        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        # Create canvas and paste images consecutively
        combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        y_offset = 0
        for img in images:
            combined_img.paste(img, (0, y_offset))
            y_offset += img.height

        # Save at 300 DPI
        combined_img.save(output_path, dpi=(300, 300))
        print(f"✅ Generated: {output_path}")
    except FileNotFoundError as e:
        print(f"⚠️ Skipped Swagger: Missing screenshot file. {e}")

# ==========================================
# 4. DARK MODE ARCHITECTURE DIAGRAMS (Pillow)
# ==========================================

def draw_flow_diagram(steps, output_path, highlights=[]):
    """Dynamically generates a dark-mode block diagram based on a list of steps."""
    box_width, box_height = 280, 50
    arrow_height = 30
    padding = 40
    
    total_width = box_width + (padding * 2)
    total_height = len(steps) * box_height + (len(steps) - 1) * arrow_height + (padding * 2)

    bg_color = (13, 17, 23)        # #0D1117
    box_bg = (22, 27, 34)          # #161B22
    border_color = (48, 54, 61)    # #30363D
    text_color = (201, 209, 217)   # #C9D1D9
    accent_color = (56, 189, 248)  # Cyan Glow (#38BDF8)
    
    img = Image.new("RGB", (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(img)
    font = get_font(18)

    y_cursor = padding
    center_x = total_width // 2

    for i, step in enumerate(steps):
        left, top = center_x - (box_width // 2), y_cursor
        right, bottom = center_x + (box_width // 2), y_cursor + box_height

        is_highlight = step in highlights
        current_border = accent_color if is_highlight else border_color
        current_text = (255, 255, 255) if is_highlight else text_color

        draw.rounded_rectangle([left, top, right, bottom], radius=10, fill=box_bg, outline=current_border, width=2)

        bbox = draw.textbbox((0, 0), step, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((center_x - (text_w // 2), top + (box_height - text_h) // 2 - 2), step, fill=current_text, font=font)

        y_cursor = bottom

        # Draw down arrow
        if i < len(steps) - 1:
            arr_end = y_cursor + arrow_height - 4
            draw.line([(center_x, y_cursor + 4), (center_x, arr_end)], fill=accent_color, width=2)
            draw.polygon([(center_x - 6, arr_end - 6), (center_x + 6, arr_end - 6), (center_x, arr_end)], fill=accent_color)
            y_cursor += arrow_height

    img.save(output_path, dpi=(300, 300))
    print(f"✅ Generated: {output_path}")

def generate_all_diagrams():
    # 1. Main Architecture
    draw_flow_diagram(
        ["Brain MRI", "DataLoader", "Transforms", "PyTorch Model", "ResNet18", "Prediction", "GradCAM", "FastAPI", "Docker API"],
        "assets/architecture.png", 
        highlights=["ResNet18", "GradCAM", "Docker API"]
    )
    
    # 2. Pipeline
    draw_flow_diagram(
        ["Raw Images", "Train / Validation / Test", "Transforms", "Training", "Checkpoint", "Evaluation", "GradCAM", "Deployment"],
        "assets/pipeline.png",
        highlights=["Training", "Evaluation", "Deployment"]
    )
    
    # 3. CNN V2 & ResNet18 Layer Stacks
    layer_stack = [
        "224×224", "Conv32", "BN", "ReLU", "Pool", 
        "Conv64", "BN", "ReLU", "Pool", 
        "Conv128", "BN", "ReLU", "Pool", 
        "Conv256", "BN", "ReLU", "Pool", 
        "AdaptiveAvgPool", "Dropout", "FC", "4 Classes"
    ]
    draw_flow_diagram(layer_stack, "assets/cnn_v2.png", highlights=["224×224", "4 Classes"])
    draw_flow_diagram(layer_stack, "assets/resnet18.png", highlights=["224×224", "4 Classes"])

# ==========================================
# 5. EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Starting comprehensive asset generation...")
    setup_directory()
    
    # Plots
    generate_confusion_matrix()
    generate_training_curve()
    generate_model_comparison()
    
    # Image Stitching
    process_gradcam_images()
    process_swagger_images()
    
    # Diagrams
    generate_all_diagrams()
    
    print("🎉 All assets generated successfully!")