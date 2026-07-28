#!/usr/bin/env python3
"""
Roboshelf AI — Kiskereskedelmi bolt demo videó generátor  v3

Használat (repo gyökeréből):
    python tools/make_store_demo.py

Kimenet:
    output/store_demo.mp4  — 1280x720, 30 fps, ~28 másodperc

Függőségek:
    pip install mujoco opencv-python
"""

import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ─── Elérési utak ─────────────────────────────────────────────────────────────
REPO       = Path(__file__).resolve().parents[1]
SCENE_XML  = REPO / "tmp" / "combined_g1_store.xml"
MESHES_DIR = REPO / "unitree_rl_gym" / "resources" / "robots" / "g1_description" / "meshes"
OUTPUT_DIR = REPO / "output"
_ts        = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_MP4 = OUTPUT_DIR / f"store_demo_{_ts}.mp4"

W, H    = 1280, 720
FPS     = 30
TOTAL_S = 28.0


# ─── Keyframe-ek ──────────────────────────────────────────────────────────────
KF_NEUTRAL = {
    "right_shoulder_pitch_joint": -0.10,
    "right_shoulder_roll_joint":  -0.35,
    "right_shoulder_yaw_joint":    0.00,
    "right_elbow_joint":           0.40,
    "left_shoulder_pitch_joint":  -0.10,
    "left_shoulder_roll_joint":    0.35,
    "left_shoulder_yaw_joint":     0.00,
    "left_elbow_joint":            0.40,
}

KF_REACH_START = {
    "right_shoulder_pitch_joint":  0.60,
    "right_shoulder_roll_joint":  -0.80,
    "right_shoulder_yaw_joint":   -0.20,
    "right_elbow_joint":           0.90,
    "left_shoulder_pitch_joint":  -0.10,
    "left_shoulder_roll_joint":    0.35,
    "left_shoulder_yaw_joint":     0.00,
    "left_elbow_joint":            0.40,
}

KF_REACH_EXTENDED = {
    "right_shoulder_pitch_joint":  0.75,
    "right_shoulder_roll_joint":  -0.90,
    "right_shoulder_yaw_joint":   -0.15,
    "right_elbow_joint":           0.30,
    "left_shoulder_pitch_joint":  -0.10,
    "left_shoulder_roll_joint":    0.35,
    "left_shoulder_yaw_joint":     0.00,
    "left_elbow_joint":            0.40,
}

KF_PUSH = {
    "right_shoulder_pitch_joint":  0.85,
    "right_shoulder_roll_joint":  -0.70,
    "right_shoulder_yaw_joint":   -0.10,
    "right_elbow_joint":           0.10,
    "left_shoulder_pitch_joint":  -0.10,
    "left_shoulder_roll_joint":    0.35,
    "left_shoulder_yaw_joint":     0.00,
    "left_elbow_joint":            0.40,
}

KF_RETRACT = {
    "right_shoulder_pitch_joint":  0.20,
    "right_shoulder_roll_joint":  -0.40,
    "right_shoulder_yaw_joint":    0.00,
    "right_elbow_joint":           0.65,
    "left_shoulder_pitch_joint":  -0.10,
    "left_shoulder_roll_joint":    0.35,
    "left_shoulder_yaw_joint":     0.00,
    "left_elbow_joint":            0.40,
}

# ─── Fázisok ──────────────────────────────────────────────────────────────────
PHASES = [
    (KF_NEUTRAL,        KF_NEUTRAL,        4.0, "shoulder_cam", "Roboshelf AI - Retail Stocking Demo"),
    (KF_NEUTRAL,        KF_REACH_START,    4.0, "behind_cam",   "Approaching target shelf position"),
    (KF_REACH_START,    KF_REACH_EXTENDED, 5.0, "behind_cam",   "Extending arm to product"),
    (KF_REACH_EXTENDED, KF_PUSH,           5.0, "shoulder_cam", "Pushing product onto shelf"),
    (KF_PUSH,           KF_RETRACT,        4.0, "behind_cam",   "Task complete - retracting arm"),
    (KF_RETRACT,        KF_NEUTRAL,        4.0, "top_cam",      "UnifoLM-VLA-0  |  80% Success Rate"),
    (KF_NEUTRAL,        KF_NEUTRAL,        2.0, "shoulder_cam", "Roboshelf AI  |  roboshelfai.com"),
]

# ─── Kamerák (robot-relatív, geometriailag ellenőrzött) ───────────────────────
# Robot: x=1.35, y=1.30, z=0.79; arc: +y irány (polc felé)
#
# behind_cam:   cam=(1.35, 0.30, 1.50) — robot MÖGÖTT, hátulról látjuk a kart + polcot
#               lookat=[1.35,1.80,0.80], az=0, el=-25, d=1.65
#
# shoulder_cam: cam=(0.69, 0.80, 1.59) — bal váll felett, oldalról, profil + polc
#               lookat=[1.35,1.90,0.70], az=-31, el=-35, d=1.56
#
# top_cam:      cam=(1.35, 0.81, 2.00) — robot felett, enyhén hátulról, madártávlat
#               lookat=[1.35,1.65,0.70], az=0, el=-57, d=1.55
# Valódi formula (sandbox-ban mérve): cam = lookat - d*[cos(el)*cos(az), cos(el)*sin(az), sin(el)]
# Ellenőrzött pozíciók (mjv_updateScene-nel mérve):
#   behind_cam:   (1.32, 0.31, 1.50) — robot mögött, hátulról+fentről, kar+polc látszik
#   shoulder_cam: (0.66, 0.82, 1.60) — bal váll felett, oldalról, profil+polc
#   top_cam:      (1.32, 0.80, 2.01) — felülről hátulról, robot+polc madártávlat
CAMERAS = {
    "behind_cam":   dict(lookat=[1.35, 1.80, 0.80], azimuth= 90.0, elevation=-25.0, distance=1.65),
    "shoulder_cam": dict(lookat=[1.35, 1.90, 0.70], azimuth= 59.0, elevation=-35.0, distance=1.56),
    "top_cam":      dict(lookat=[1.35, 1.65, 0.70], azimuth= 90.0, elevation=-57.0, distance=1.56),
}


def lerp_kf(kf_a, kf_b, t):
    keys = set(kf_a) | set(kf_b)
    return {k: kf_a.get(k, 0.0) + (kf_b.get(k, 0.0) - kf_a.get(k, 0.0)) * t for k in keys}


def smooth(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def add_overlay(frame_bgr, text, time_s, total_s):
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    bar_h = 52
    overlay = out.copy()
    overlay[h - bar_h:, :] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, text,           (16, h - bar_h + 32), font, 0.65, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(out, "ROBOSHELF AI", (w - 170, h - bar_h + 32), font, 0.55, (255,255,255), 1, cv2.LINE_AA)
    prog_w = int(w * time_s / total_s)
    cv2.rectangle(out, (0, h - 3), (prog_w, h), (233, 165, 14), -1)
    return out


def main():
    import mujoco

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Roboshelf AI — Store Demo Video Generator  v3")
    print(f"{'='*60}")
    print(f"  Output: {OUTPUT_MP4}")
    print(f"  Res:    {W}x{H}  |  {FPS} fps  |  ~{TOTAL_S:.0f}s")
    print(f"{'='*60}\n")

    if not SCENE_XML.exists():
        print(f"[ERROR] {SCENE_XML}"); sys.exit(1)
    if not MESHES_DIR.exists():
        print(f"[ERROR] {MESHES_DIR}"); sys.exit(1)

    # ── 1. XML betöltés: meshdir patch + offscreen framebuffer méret ──────────
    print("[1/5] Loading model...")
    xml = SCENE_XML.read_text()
    xml = xml.replace('meshdir="assets"', f'meshdir="{MESHES_DIR}"')
    xml = xml.replace(
        '<option integrator="implicitfast" />',
        '<option integrator="implicitfast" />\n'
        '  <visual><global offwidth="1280" offheight="720"/></visual>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)
    print(f"      {model.nbody} body, {model.njnt} joint, {model.nmesh} mesh")

    # ── 2. Robot pozicionálása Gondola A elé ──────────────────────────────────
    print("[2/5] Positioning robot...")
    fb_jid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    fb_qaddr = model.jnt_qposadr[fb_jid]

    # Gondola B (x=+1.35) a korridor felé nyílik — termékek y=1.95-nél, robot y=1.30-nál
    # Gondola A háttal néz a korridornak, ezért B-t használjuk
    angle = math.pi / 2.0   # robot +y irányba néz (polc felé)
    PELVIS_QPOS = [1.35, 1.30, 0.793,
                   math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)]
    data.qpos[fb_qaddr:fb_qaddr+7] = PELVIS_QPOS
    mujoco.mj_forward(model, data)
    print(f"      pelvis @ {data.xpos[model.body('pelvis').id]}")

    # ── 3. Joint map ──────────────────────────────────────────────────────────
    all_keys = set()
    for kf in [KF_NEUTRAL, KF_REACH_START, KF_REACH_EXTENDED, KF_PUSH, KF_RETRACT]:
        all_keys |= set(kf)
    jnt_map = {}
    for name in all_keys:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            jnt_map[name] = model.jnt_qposadr[jid]

    def apply_kf(kf):
        # Pelvis visszaállítása (nehogy a freejoint elmásszon)
        data.qpos[fb_qaddr:fb_qaddr+7] = PELVIS_QPOS
        # Kar joints beállítása
        for name, qaddr in jnt_map.items():
            if name in kf:
                data.qpos[qaddr] = kf[name]
        # NINCS mj_step — csak kinematikai frissítés, robot nem dülöng
        mujoco.mj_forward(model, data)

    # ── 4. Renderer + VideoWriter ─────────────────────────────────────────────
    print("[3/5] Setup renderer + writer...")
    renderer = mujoco.Renderer(model, height=H, width=W)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, FPS, (W, H))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_MP4), fourcc, FPS, (W, H))
    if not writer.isOpened():
        print("[ERROR] Nincs működő videocodec (avc1/mp4v)"); sys.exit(1)
    print(f"      OK")

    # ── 5. Render loop ────────────────────────────────────────────────────────
    print("[4/5] Rendering...")
    global_time = 0.0
    frame_count = 0

    for idx, (kf_a, kf_b, dur, cam_name, caption) in enumerate(PHASES):
        n_frames = int(dur * FPS)
        print(f"  [{idx+1}/{len(PHASES)}] {caption[:45]:<45} {n_frames}fr")

        cfg = CAMERAS[cam_name]
        cam = mujoco.MjvCamera()
        cam.lookat[:]  = cfg["lookat"]
        cam.distance   = cfg["distance"]
        cam.azimuth    = cfg["azimuth"]
        cam.elevation  = cfg["elevation"]

        for fi in range(n_frames):
            t = smooth(fi / max(n_frames - 1, 1))
            apply_kf(lerp_kf(kf_a, kf_b, t))
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = add_overlay(bgr, caption, global_time, TOTAL_S)
            writer.write(bgr)
            global_time += 1.0 / FPS
            frame_count += 1

    renderer.close()
    writer.release()

    size_kb = OUTPUT_MP4.stat().st_size // 1024
    print(f"\n[5/5] Done — {frame_count} frame, {frame_count/FPS:.1f}s, {size_kb} KB")
    print(f"  open {OUTPUT_MP4}\n")


if __name__ == "__main__":
    main()
