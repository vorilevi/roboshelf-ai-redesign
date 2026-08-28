"""
shelflife_render_env.py — renderelés fejlesztői környezetben is

    import shelflife_render_env   # importáláskor beállít mindent, ha kell

────────────────────────────────────────────────────────────────────────────
MIÉRT
────────────────────────────────────────────────────────────────────────────
A fejlesztői sandboxban nincs ablakkezelő, ezért a MuJoCo renderelése eddig
`FatalError: an OpenGL platform library has not been loaded` hibával szállt el.
Minden vizuális ellenőrzést a felhasználó gépére kellett küldeni — és ez
mérhetően lassította a munkát: a fogási hiba gyökerét (a doboz a kéz MELLETT
van, nem benne) EGY videókocka azonnal megmutatta, miközben számokból sok
körön át kerestem.

A megoldás nem hiányzó csomag, hanem hiányzó KIJELZŐ:

    libGL.so.1  →  megvan  (aarch64)
    Xvfb        →  megvan
    DISPLAY     →  nincs   ← ez volt a hiány

Egy virtuális X-kijelzővel (`Xvfb :99`) és `MUJOCO_GL=glx`-szel a szoftveres
renderelés működik. Mérve: 240×240-es próbakép hibátlanul.

macOS-en (a felhasználó gépén) mindez felesleges — ott natív a grafika —, ezért
a modul csak Linuxon és csak akkor lép működésbe, ha nincs `DISPLAY`.

────────────────────────────────────────────────────────────────────────────
KÖVETKEZMÉNY A MUNKAMÓDSZERRE
────────────────────────────────────────────────────────────────────────────
Munkaszabály lett belőle (l. projektterv 6/5): ha egy hibát két körnél tovább
keresek számokból, rendereljek képet. A durva diagnózist a kép adja, a finomat
a mérés — eddig csak az egyiket használtuk.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time

DISPLAY_ID = ":99"
_READY = False


def ensure(width: int = 1280, height: int = 1024) -> bool:
    """Gondoskodik róla, hogy legyen renderelhető környezet. True, ha van."""
    global _READY
    if _READY:
        return True
    if platform.system() == "Darwin":                 # macOS: natív
        _READY = True
        return True
    if os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "glx")
        _READY = True
        return True
    if not shutil.which("Xvfb"):
        return False

    running = subprocess.run(["pgrep", "-f", f"Xvfb {DISPLAY_ID}"],
                             capture_output=True).returncode == 0
    if not running:
        subprocess.Popen(
            ["Xvfb", DISPLAY_ID, "-screen", "0", f"{width}x{height}x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        time.sleep(2.0)
    os.environ["DISPLAY"] = DISPLAY_ID
    os.environ["MUJOCO_GL"] = "glx"
    _READY = True
    return True


def selftest() -> bool:
    """Rendereltünk-e ténylegesen? Nem elég a beállítás, látni is kell."""
    if not ensure():
        return False
    import mujoco
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 2"/>'
        '<geom type="box" size=".1 .1 .1" rgba=".2 .6 .9 1"/>'
        '</worldbody></mujoco>')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 64, 64)
    r.update_scene(d)
    img = r.render()
    r.close()
    return bool(img is not None and img.std() > 1.0)   # ne legyen üres kép


ensure()


if __name__ == "__main__":
    ok = selftest()
    print(f"renderelés: {'✅ működik' if ok else '❌ nem elérhető'} · "
          f"DISPLAY={os.environ.get('DISPLAY', '—')} · "
          f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '—')}")
    raise SystemExit(0 if ok else 1)
