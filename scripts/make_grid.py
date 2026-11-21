"""Grid video: tiles rendered with `python -m dlo.make_videos_soft ... --plain` are arranged in rows x cols.
usage: make_grid.py tiles_dir index_file out.mp4 [rows cols tile_w tile_h]"""
import sys, numpy as np, imageio
from PIL import Image

tiles, idx_file, out = sys.argv[1:4]
rows, cols, tw, th = (int(x) for x in sys.argv[4:8]) if len(sys.argv) > 4 else (10, 10, 256, 144)
idx = [int(l) for l in open(idx_file).read().split()][:rows * cols]
readers = [imageio.get_reader(f"{tiles}/episode_{e:03d}.mp4") for e in idx]
fps = readers[0].get_meta_data()["fps"]; n = min(r.count_frames() for r in readers); gap = 2
W, H = cols * tw + (cols + 1) * gap, rows * th + (rows + 1) * gap
W += W % 2; H += H % 2
with imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
    for t in range(n):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        for k, r in enumerate(readers):
            im = Image.fromarray(r.get_data(t)).resize((tw, th), Image.LANCZOS)
            canvas.paste(im, (gap + (k % cols) * (tw + gap), gap + (k // cols) * (th + gap)))
        w.append_data(np.asarray(canvas))
print(out, f"{len(idx)} tiles, {n} frames, {W}x{H}")
