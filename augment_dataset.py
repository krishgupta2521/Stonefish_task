#!/usr/bin/env python3
"""Create random augmented copies of images for an object-detection dataset."""
import argparse, random, sys
from pathlib import Path
import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def images_in(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def read_labels(path):
    boxes = []
    if path is None or not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:  # Standard YOLO detection labels only; skip segmentation polygons.
            continue
        try:
            boxes.append([int(float(p[0])), *map(float, p[1:])])
        except ValueError:
            pass
    return boxes

def write_labels(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\\n" for c,x,y,w,h in boxes), encoding="utf-8")

def flip(img, boxes):
    return cv2.flip(img, 1), [[c, 1-x, y, w, h] for c,x,y,w,h in boxes]

def brightness(img, boxes, factor):
    return np.clip(img.astype(np.float32)*factor, 0, 255).astype(np.uint8), boxes.copy()

def contrast(img, boxes, factor):
    mean = img.mean(axis=(0,1), keepdims=True)
    out = np.clip((img.astype(np.float32)-mean)*factor+mean, 0, 255).astype(np.uint8)
    return out, boxes.copy()

def zoom(img, boxes, factor, rng):
    h,w = img.shape[:2]
    cw,ch = max(2,int(w/factor)), max(2,int(h/factor))
    x0,y0 = rng.randint(0,max(0,w-cw)), rng.randint(0,max(0,h-ch))
    out = cv2.resize(img[y0:y0+ch, x0:x0+cw], (w,h), interpolation=cv2.INTER_LINEAR)
    result=[]
    for c,x,y,bw,bh in boxes:
        x1,x2=(x-bw/2)*w,(x+bw/2)*w
        y1,y2=(y-bh/2)*h,(y+bh/2)*h
        a,b=max(0,(x1-x0)/cw),max(0,(y1-y0)/ch)
        d,e=min(1,(x2-x0)/cw),min(1,(y2-y0)/ch)
        if d>a and e>b: result.append([c,(a+d)/2,(b+e)/2,d-a,e-b])
    return out,result

def rotate(img, boxes, angle):
    h,w=img.shape[:2]
    m=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    out=cv2.warpAffine(img,m,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    result=[]
    for c,x,y,bw,bh in boxes:
        x1,x2=(x-bw/2)*w,(x+bw/2)*w
        y1,y2=(y-bh/2)*h,(y+bh/2)*h
        pts=np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]],dtype=np.float32) @ m.T
        a=max(0,float(pts[:,0].min())/w); b=max(0,float(pts[:,1].min())/h)
        d=min(1,float(pts[:,0].max())/w); e=min(1,float(pts[:,1].max())/h)
        if d>a and e>b: result.append([c,(a+d)/2,(b+e)/2,d-a,e-b])
    return out,result

def safe_stem(p):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p.stem) or "image"

def main():
    ap=argparse.ArgumentParser(description="Randomly augment images; optionally update standard YOLO detection labels.")
    ap.add_argument("--input",required=True,help="Folder containing source images (recursive).")
    ap.add_argument("--output",default="augmented_images",help="Output image folder.")
    ap.add_argument("--count",type=int,default=350,help="Total number of augmented images to create.")
    ap.add_argument("--labels",help="Optional folder containing matching YOLO .txt labels.")
    ap.add_argument("--output-labels",help="Output folder for transformed YOLO labels (required with --labels).")
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--allow-upside-down",action="store_true",help="Use 180-degree rotations instead of small rotations.")
    args=ap.parse_args()
    src=Path(args.input).expanduser().resolve(); out=Path(args.output).expanduser().resolve()
    lab=Path(args.labels).expanduser().resolve() if args.labels else None
    outlab=Path(args.output_labels).expanduser().resolve() if args.output_labels else None
    if not src.is_dir(): ap.error(f"Input folder does not exist: {src}")
    if args.count<1: ap.error("--count must be positive")
    if lab and not outlab: ap.error("--output-labels is required with --labels")
    imgs=[p for p in images_in(src) if out not in p.parents]
    if not imgs: ap.error("No source images found.")
    rng=random.Random(args.seed)
    types=["horizontal_flip","brighten","darken","contrast","zoom","rotate_180" if args.allow_upside_down else "small_rotate"]
    out.mkdir(parents=True,exist_ok=True)
    if outlab: outlab.mkdir(parents=True,exist_ok=True)
    counts={t:0 for t in types}; made=0
    print(f"Found {len(imgs)} source images. Generating {args.count} augmented images.")
    for i in range(args.count):
        p=rng.choice(imgs)
        img=cv2.imread(str(p),cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] Cannot read {p}"); continue
        boxes=read_labels(lab/f"{p.stem}.txt") if lab else []
        aug=rng.choice(types)
        if aug=="horizontal_flip": result, newboxes=flip(img,boxes)
        elif aug=="brighten": result,newboxes=brightness(img,boxes,rng.uniform(1.2,1.55))
        elif aug=="darken": result,newboxes=brightness(img,boxes,rng.uniform(.5,.8))
        elif aug=="contrast": result,newboxes=contrast(img,boxes,rng.uniform(.75,1.3))
        elif aug=="zoom": result,newboxes=zoom(img,boxes,rng.uniform(1.1,1.35),rng)
        else: result,newboxes=rotate(img,boxes,180.0 if aug=="rotate_180" else rng.uniform(-12,12))
        made+=1
        name=f"aug_{made:05d}_{safe_stem(p)}_{aug}.jpg"
        if not cv2.imwrite(str(out/name),result,[cv2.IMWRITE_JPEG_QUALITY,95]):
            print(f"[ERROR] Could not write {out/name}"); made-=1; continue
        if outlab: write_labels(outlab/f"{Path(name).stem}.txt",newboxes)
        counts[aug]+=1
        if made%25==0 or made==args.count: print(f"Progress: {made}/{args.count}")
    print(f"Done. Created {made} images in {out}")
    for name,count in counts.items(): print(f"  {name}: {count}")

if __name__=="__main__":
    main()
