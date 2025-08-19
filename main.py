"""
make_video_resume.py

Creates a cinematic video resume with VFX and a British male voice (Edge TTS).
No moviepy used. Uses OpenCV + Pillow to render frames and ffmpeg to mux audio.

Requirements:
pip install edge-tts pillow opencv-python numpy mutagen
ffmpeg must be installed and on PATH.
Place vishwajit.jpg, ml.jpg, ai.jpg, datascience.jpg in the same folder as this script.
"""

import os
import math
import random
import subprocess
import asyncio
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import edge_tts

# optional for fallback audio duration
try:
    from mutagen.mp3 import MP3
except Exception:
    MP3 = None

# -----------------------------
# CONFIG
# -----------------------------
OUTPUT_DIR = os.path.abspath(".")
PHOTO_FILE = os.path.join(OUTPUT_DIR, "vishwajit.jpg")
LOGOS = [
    os.path.join(OUTPUT_DIR, "ml.jpg"),
    os.path.join(OUTPUT_DIR, "ai.jpg"),
    os.path.join(OUTPUT_DIR, "datascience.jpg"),
]
VOICEOVER_FILE = os.path.join(OUTPUT_DIR, "voiceover.mp3")
TEMP_VIDEO = os.path.join(OUTPUT_DIR, "temp_video.mp4")   # video only
FINAL_VIDEO = os.path.join(OUTPUT_DIR, "final_intro_video.mp4")

W, H = 1280, 720
FPS = 30

# fonts - adjust if you want a different TTF
FONT_PATHS = [
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def get_font(sz):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()

FONT_TITLE = get_font(56)
FONT_SUB = get_font(28)
FONT_BODY = get_font(22)

# -----------------------------
# YOUR LONG SCRIPT (exact as provided)
# -----------------------------
SCRIPT = """Hello everyone! Welcome to my professional video resume. I am a seasoned data science AI leader with over 15 years of experience, combining deep technical expertise with strategic vision to drive meaningful business outcomes. Currently, I am focused on building a cutting-edge conversation AI product utilizing technologies such as Python, PyTorch, Hugging Face, Large Language Models (LLMs), BERT, Dataiku, Sentence Transformers, word embedding models, matching algorithms, conversational AI, prompt engineering, and SBERT.
Also, have experience in working with Seismic data (.segy format) for data analysis and identifying the bright spot in Oil & gas sector.
Throughout my career, I have successfully delivered high-impact solutions across diverse industries, including manufacturing, healthcare, banking, pharmaceuticals, automotive, insurance, consulting and technology. My passion lies in translating complex data into actionable insights that empower organizations to make informed decisions, foster innovation, and achieve sustainable growth. My technical expertise includes Python, R, SAS, SQL, cloud platforms (AWS, Azure), Big Data technologies, PySpark, deep learning frameworks (PyTorch, Dataiku, Transformers), Agentic AI, MLOps, data visualization tools (Power BI, Tableau), and advanced AI applications including Generative AI, LLMs, Retrieval-Augmented Generation (RAG), Computer vision problems like Image classification using Transfer Learning (VGG16, Resnet etc.), Object detection using Faster R-CNN, YOLO, Image segmentation, Langchain, Llama Index, Matching Algorithms using BERT, SBERT, voice cloning using XTTS_v2 and Hugging Face. Experienced in building and leading high-performing data science teams, cultivating a culture of innovation, and aligning technical solutions with business goals through strong stakeholder collaboration. As a lifelong learner, I am committed to staying at the forefront of the rapidly evolving data science landscape, continually seeking new challenges and opportunities to push the boundaries of what data and AI can achieve.
Coached multiple folks to successfully build their career in the data science and problem-solving using ML/DL algorithms."""
# strip any accidental leading/trailing whitespace
SCRIPT = SCRIPT.strip()

# -----------------------------
# UTIL: synthesize with Edge TTS (British male)
# -----------------------------
async def synthesize_edge_tts(text, voice="en-GB-RyanNeural", outfile=VOICEOVER_FILE):
    # Edge TTS Communicate accepts long text, but if you face issues you can chunk it.
    print("Synthesizing voice with Edge TTS (British male) ->", voice)
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(outfile)
    print("Saved voiceover to:", outfile)

def ensure_voiceover():
    if os.path.exists(VOICEOVER_FILE):
        print("Using existing voiceover:", VOICEOVER_FILE)
        return
    asyncio.run(synthesize_edge_tts(SCRIPT, voice="en-GB-RyanNeural", outfile=VOICEOVER_FILE))

# -----------------------------
# UTIL: get audio duration (ffprobe preferred, fallback mutagen)
# -----------------------------
def get_audio_duration(path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path
        ], stderr=subprocess.DEVNULL)
        return float(out.decode().strip())
    except Exception:
        if MP3 is not None:
            try:
                a = MP3(path)
                return a.info.length
            except Exception:
                pass
    return None

# -----------------------------
# UTIL: overlay with alpha channel
# -----------------------------
def overlay_image_alpha(bg, fg, x, y):
    """Overlay fg (BGR or BGRA) onto bg (BGR) at x,y using fg alpha if present."""
    h, w = fg.shape[:2]
    if fg.shape[2] == 3:
        # simple paste with bounds
        h0 = min(h, bg.shape[0] - y)
        w0 = min(w, bg.shape[1] - x)
        if h0 <= 0 or w0 <= 0:
            return bg
        bg[y:y+h0, x:x+w0] = fg[:h0, :w0]
        return bg
    # BGRA
    alpha = fg[:, :, 3] / 255.0
    fg_rgb = fg[:, :, :3].astype(float)
    bg_region = bg[y:y+h, x:x+w].astype(float)
    h0 = min(h, bg.shape[0] - y)
    w0 = min(w, bg.shape[1] - x)
    if h0 <= 0 or w0 <= 0:
        return bg
    alpha_r = alpha[:h0, :w0][:, :, None]
    comp = (alpha_r * fg_rgb[:h0, :w0] + (1 - alpha_r) * bg_region[:h0, :w0]).astype(np.uint8)
    bg[y:y+h0, x:x+w0] = comp
    return bg

# -----------------------------
# Prepare assets
# -----------------------------
if not os.path.exists(PHOTO_FILE):
    raise FileNotFoundError("Portrait not found: " + PHOTO_FILE)

# load portrait and make it "cover" the frame
def load_and_cover(path, target_w, target_h):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("Unable to read image: " + path)
    ih, iw = img.shape[:2]
    scale = max(target_w / iw, target_h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    x = (nw - target_w) // 2
    y = (nh - target_h) // 2
    return img_resized[y:y+target_h, x:x+target_w]

portrait = load_and_cover(PHOTO_FILE, W, H)

loaded_logos = []
for p in LOGOS:
    if os.path.exists(p):
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is not None:
            # resize logos to reasonable size
            ih, iw = im.shape[:2]
            scale = min(360/iw, 140/ih, 1.0)
            im_s = cv2.resize(im, (int(iw*scale), int(ih*scale)), interpolation=cv2.INTER_AREA)
            loaded_logos.append(im_s)
if not loaded_logos:
    print("No logos found (ml.jpg, ai.jpg, datascience.jpg) — script will continue without logos.")

# particles
NUM_PARTICLES = 120
particles = []
for _ in range(NUM_PARTICLES):
    particles.append([random.uniform(0, W), random.uniform(0, H), random.uniform(8,80), random.uniform(1.0,4.0),
                      (random.randint(90,220), random.randint(90,220), random.randint(90,220))])

# precompute vignette
yy, xx = np.mgrid[0:H, 0:W]
cx, cy = W/2, H/2
dx = (xx - cx) / (W/2)
dy = (yy - cy) / (H/2)
rad = np.sqrt(dx*dx + dy*dy)
vignette = 1.0 - 0.65 * np.clip(rad, 0.0, 1.0)

# -----------------------------
# Synthesize voice (Edge TTS) and measure duration
# -----------------------------
ensure_voiceover()
audio_duration = get_audio_duration(VOICEOVER_FILE)
if audio_duration is None:
    print("Warning: couldn't detect audio duration. Defaulting to 30s.")
    audio_duration = 30.0
print(f"Voiceover duration: {audio_duration:.2f}s")

# TIMELINE (seconds)
NAME_DUR = 2.5
LOGO_CYCLE_DUR = 2.0 if loaded_logos else 0.0
LOGOS_TOTAL = LOGO_CYCLE_DUR * len(loaded_logos)
OUTRO_DUR = 1.5
MAIN_DUR = max(3.0, audio_duration - (NAME_DUR + LOGOS_TOTAL + OUTRO_DUR))
TOTAL_DURATION = NAME_DUR + LOGOS_TOTAL + MAIN_DUR + OUTRO_DUR
TOTAL_FRAMES = int(math.ceil(TOTAL_DURATION * FPS))

print(f"Timeline (s): name={NAME_DUR}, logos={LOGOS_TOTAL}, main={MAIN_DUR}, outro={OUTRO_DUR}")
print("Total frames:", TOTAL_FRAMES)

# subtitles for main section
SUBS = [
    "Python · PyTorch · Hugging Face · LLMs · SBERT, AWS, AZURE, GCP"
    "MLOps · Agentic AI · Retrieval-Augmented Generation",
    "Computer Vision · Transfer Learning (VGG16, ResNet)",
    "Faster R-CNN, YOLO · Image Segmentation · LangChain · LlamaIndex"
    "R, SQL, NLP, LLMs, Machine Learning, Deep Learning"
]
SUB_DUR = 3.5

# setup writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(TEMP_VIDEO, fourcc, FPS, (W, H))

# helper: draw centered text with PIL
def draw_centered_multiline(pil_img, text, font, y_center, max_width, fill=(255,255,255,255)):
    draw = ImageDraw.Draw(pil_img)
    # naive wrap
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        test = cur + (" " if cur else "") + w
        try:
            bbox = draw.textbbox((0,0), test, font=font)
            tw = bbox[2]-bbox[0]
        except Exception:
            tw = font.getsize(test)[0]
        if tw <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    # draw lines centered
    try:
        line_h = draw.textbbox((0,0), "Ay", font=font)[3] - draw.textbbox((0,0), "Ay", font=font)[1]
    except Exception:
        line_h = font.getsize("Ay")[1]
    total_h = line_h * len(lines)
    y0 = int(y_center - total_h/2)
    for i, ln in enumerate(lines):
        try:
            bbox = draw.textbbox((0,0), ln, font=font)
            tw = bbox[2]-bbox[0]; th = bbox[3]-bbox[1]
        except Exception:
            tw, th = font.getsize(ln)
        x = (W - tw)//2
        draw.text((x, y0 + i*line_h), ln, font=font, fill=fill)

# -----------------------------
# RENDER FRAMES
# -----------------------------
print("Rendering frames (this may take several minutes depending on CPU)...")
for frame_idx in range(TOTAL_FRAMES):
    t = frame_idx / FPS

    # base animated gradient (clamped)
    shift = int(36 * math.sin(2 * math.pi * (t / 6.0)))
    r = int(np.clip(18 + shift, 0, 255))
    g = int(np.clip(12 + shift//4, 0, 255))
    b = int(np.clip(30 + shift, 0, 255))
    bg = np.full((H, W, 3), (r, g, b), dtype=np.uint8)

    # make PIL for drawing text overlays
    pil = Image.fromarray(bg)
    draw = ImageDraw.Draw(pil)

    # Scenes: name -> logos -> main portrait -> outro
    if t < NAME_DUR:
        # large name reveal with soft glow
        p = t / max(1e-6, NAME_DUR)
        # background vignette slightly stronger
        frame_np = np.array(pil)
        # glow shadow
        title = "Vishwajit Sen"
        subtitle = "Data Science & AI Leader"
        # draw shadow by multiple offsets
        shadow_col = (8,8,10)
        for ox, oy in [(-6,-6),(6,6),(6,-6),(-6,6),(0,8)]:
            draw.text((W//2+ox, H//2+oy - 20), title, font=FONT_TITLE, anchor="mm", fill=shadow_col)
        draw.text((W//2, H//2 - 20), title, font=FONT_TITLE, anchor="mm", fill=(235,235,245))
        draw.text((W//2, H//2 + 50), subtitle, font=FONT_SUB, anchor="mm", fill=(200,200,210))

        frame = np.array(pil)

    elif t < NAME_DUR + LOGOS_TOTAL and loaded_logos:
        # one-by-one logo showcase
        tt = t - NAME_DUR
        idx = int(tt // LOGO_CYCLE_DUR)
        idx = min(idx, len(loaded_logos)-1)
        local_t = tt - idx * LOGO_CYCLE_DUR
        prog = local_t / LOGO_CYCLE_DUR
        logo = loaded_logos[idx]
        lh, lw = logo.shape[:2]
        # animate: fade+scale+spin (mild)
        scale = 0.5 + 0.7 * prog
        new_w = int(lw*scale); new_h = int(lh*scale)
        logo_r = cv2.resize(logo, (max(1,new_w), max(1,new_h)), interpolation=cv2.INTER_AREA)
        x = (W - logo_r.shape[1]) // 2
        y = (H - logo_r.shape[0]) // 2
        frame_np = np.array(pil)
        frame_np = overlay_image_alpha(frame_np, logo_r, x, y)
        # small caption
        draw = ImageDraw.Draw(Image.fromarray(frame_np))
        # write caption under the logo
        pil2 = Image.fromarray(frame_np)
        draw2 = ImageDraw.Draw(pil2)
        caption = f"Technology · {idx+1}/{len(loaded_logos)}"
        draw2.text((W//2, H - 140), caption, font=FONT_SUB, anchor="mm", fill=(220,220,230))
        frame = np.array(pil2)

    else:
        # MAIN portrait section with Ken-Burns, subtitles, floating logos & particles
        tt = t - (NAME_DUR + LOGOS_TOTAL)
        prog = min(1.0, tt / max(1e-6, MAIN_DUR))

        # Ken-Burns scale from 1.04 -> 1.18
        kb_scale = 1.04 + 0.14 * prog
        ph, pw = portrait.shape[:2]
        sw, sh = int(pw*kb_scale), int(ph*kb_scale)
        big = cv2.resize(portrait, (sw, sh), interpolation=cv2.INTER_LINEAR)
        cx = max(0, (sw - W)//2)
        cy = max(0, (sh - H)//2)
        frame_np = big[cy:cy+H, cx:cx+W].copy()

        # apply vignette
        for c in range(3):
            frame_np[:,:,c] = (frame_np[:,:,c].astype(np.float32) * vignette).astype(np.uint8)

        # floating logos top-right
        for j, lg in enumerate(loaded_logos):
            off_x = int(W - 140 - j*120 + 12*math.sin(2*math.pi*(t*0.6 + j)))
            off_y = int(36 + 12*math.cos(2*math.pi*(t*0.8 + j)))
            frame_np = overlay_image_alpha(frame_np, lg, off_x, off_y)

        # particles overlay
        for p in particles:
            px, py, speed, size, col = p
            py -= speed / FPS
            if py < -30:
                py = H + random.uniform(0, 120)
                px = random.uniform(0, W)
            p[0], p[1] = px, py
            cv2.circle(frame_np, (int(px), int(py)), int(size), col, -1, lineType=cv2.LINE_AA)

        # subtitles cycling near bottom
        if tt > 0.6:
            sub_tt = tt - 0.6
            sub_idx = int((sub_tt // SUB_DUR) % len(SUBS))
            sub_text = SUBS[sub_idx]
            pilf = Image.fromarray(cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB))
            drawf = ImageDraw.Draw(pilf)
            # semi-transparent box
            bw, bh = int(W*0.9), 64
            bx = int(W*0.05); by = H - 140
            rect = Image.new("RGBA", (bw, bh), (10,10,12,180))
            pilf.paste(rect, (bx, by), rect)
            # text
            drawf.text((bx+18, by+10), sub_text, font=FONT_SUB, fill=(245,245,250))
            frame_np = cv2.cvtColor(np.array(pilf), cv2.COLOR_RGB2BGR)

        frame = frame_np

    # outro fade
    if t >= NAME_DUR + LOGOS_TOTAL + MAIN_DUR:
        frag = (t - (NAME_DUR + LOGOS_TOTAL + MAIN_DUR)) / max(1e-6, OUTRO_DUR)
        # overlay a soft fade/thank you text
        pil_frame = Image.fromarray(frame)
        draw_o = ImageDraw.Draw(pil_frame)
        alpha_box = Image.new("RGBA", (W, H), (0,0,0, int(180*frag)))
        pil_frame = Image.alpha_composite(pil_frame.convert("RGBA"), alpha_box).convert("RGB")
        draw_o = ImageDraw.Draw(pil_frame)
        if frag > 0.05:
            draw_o.text((W//2, H//2), "Thank you", font=FONT_TITLE, anchor="mm", fill=(235,235,240))
        frame = np.array(pil_frame)

    # cinematic letterbox bars (top & bottom)
    bar_h = 44
    frame[0:bar_h, :, :] = (frame[0:bar_h, :, :].astype(np.float32) * 0.12).astype(np.uint8)
    frame[H-bar_h:H, :, :] = (frame[H-bar_h:H, :, :].astype(np.float32) * 0.12).astype(np.uint8)

    # film grain
    grain = (np.random.randn(H, W, 1) * 5).astype(np.int16)
    for c in range(3):
        tmp = frame[:,:,c].astype(np.int16) + grain[:,:,0]
        tmp = np.clip(tmp, 0, 255)
        frame[:,:,c] = tmp.astype(np.uint8)

    writer.write(frame)

writer.release()
print("Temp video written ->", TEMP_VIDEO)

# -----------------------------
# MUX audio + video via ffmpeg
# -----------------------------
print("Muxing audio and video with ffmpeg...")
cmd = [
    "ffmpeg", "-y",
    "-i", TEMP_VIDEO,
    "-i", VOICEOVER_FILE,
    "-c:v", "libx264", "-preset", "medium",
    "-c:a", "aac", "-b:a", "192k",
    "-map", "0:v:0", "-map", "1:a:0",
    "-shortest", FINAL_VIDEO
]
subprocess.run(cmd, check=True)
print("Final video saved ->", FINAL_VIDEO)

# optional cleanup: remove temp video
try:
    os.remove(TEMP_VIDEO)
except Exception:
    pass

print("All done — enjoy your cinematic video resume!")
