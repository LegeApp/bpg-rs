#!/usr/bin/env python3
import subprocess, time, os, csv, sys

BPGENC = "/home/dk/Desktop/testers-linux/BPG/libbpg-0.9.8/bpgenc"
BPGDEC = "/home/dk/Desktop/testers-linux/BPG/libbpg-0.9.8/bpgdec"
SRC = "/home/dk/Desktop/testers-linux/BPG/dusk.png"
OUTDIR = "/home/dk/Desktop/testers-linux/BPG/test_runs"

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def metrics(png_path):
    r = run(["ffmpeg", "-hide_banner", "-i", png_path, "-i", SRC,
             "-lavfi", "ssim;[0:v][1:v]psnr", "-f", "null", "-"])
    out = r.stderr
    psnr = ssim = None
    for line in out.splitlines():
        if "PSNR" in line and "average" in line:
            for tok in line.split():
                if tok.startswith("average:"):
                    psnr = float(tok.split(":")[1])
        if "SSIM" in line and "All" in line:
            for tok in line.split():
                if tok.startswith("All:"):
                    ssim = float(tok.split(":")[1].split("(")[0])
    return psnr, ssim

def encode_one(encoder, qp, cfmt, m=None, extra=None, tag=""):
    name = f"{encoder}_q{qp}_{cfmt}{('_m'+str(m)) if m else ''}{tag}"
    bpg_path = os.path.join(OUTDIR, name + ".bpg")
    png_path = os.path.join(OUTDIR, name + ".png")
    cmd = [BPGENC, "-e", encoder, "-q", str(qp), "-f", cfmt, "-o", bpg_path, SRC]
    if m is not None:
        cmd += ["-m", str(m)]
    if extra:
        cmd += extra
    t0 = time.time()
    r = run(cmd)
    t1 = time.time()
    if r.returncode != 0:
        print("ENCODE FAIL", name, r.stderr, file=sys.stderr)
        return None
    size = os.path.getsize(bpg_path)
    rd = run([BPGDEC, "-o", png_path, bpg_path])
    if rd.returncode != 0:
        print("DECODE FAIL", name, rd.stderr, file=sys.stderr)
        return None
    psnr, ssim = metrics(png_path)
    os.remove(png_path)
    row = dict(name=name, encoder=encoder, qp=qp, cfmt=cfmt, m=m,
               extra=" ".join(extra) if extra else "",
               size_bytes=size, enc_time_s=round(t1-t0, 3),
               psnr=psnr, ssim=ssim)
    print(row, flush=True)
    return row

def main():
    csv_path = os.path.join(OUTDIR, "results.csv")
    fieldnames = ["name","encoder","qp","cfmt","m","extra","size_bytes","enc_time_s","psnr","ssim"]
    f = open(csv_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)

    qps = [24, 32, 40]

    # m=9 sweep at 420 (x265 placebo preset, jctvc TU depth 4)
    for qp in qps:
        for enc in ["x265", "jctvc"]:
            row = encode_one(enc, qp, "420", m=9)
            if row:
                w.writerow(row)
                f.flush()

    # 444 chroma sweep at default m
    for qp in qps:
        for enc in ["x265", "jctvc"]:
            row = encode_one(enc, qp, "444")
            if row:
                w.writerow(row)
                f.flush()

    f.close()

if __name__ == "__main__":
    main()
