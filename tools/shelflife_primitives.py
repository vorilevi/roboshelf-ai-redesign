"""
shelflife_primitives.py — a Shelf Life PRIMITÍV-rétege (GENE.01)

A Waddle-féle code-as-policy architektúra alsó szintje:

    Primitívek   ← EZ A FÁJL. Fix, platform által adott. Az ügynök NEM írja.
    Készségek    ← az ügynök hozza létre ezekből (grasp_box, orient_to_face, ...)
    Program      ← az ügynök végső szkriptje a feladatra

A primitívek feladata, hogy elrejtsék a vezérlés nehéz részét, és az ügynök
szemantikai szinten dolgozhasson. Ami itt van, az MÉRT és validált; amit az
ügynök ír, azt az eval validálja.

────────────────────────────────────────────────────────────────────────────
KÉT MÉRT PROBLÉMA, AMIT EZ A RÉTEG OLD MEG
────────────────────────────────────────────────────────────────────────────
1. NYÍLT HURKÚ POZICIONÁLÁS PONTATLAN.
   Kinematikailag a tenyér 4 mm-re odaér, PD-vel csak 47 mm-re — a kar
   megereszkedik a gravitációban. A `kp` emelése nem javít (30000-nél a hiba
   0.09 rad, a qacc viszont 2786-ra ugrik).
   → `move_palm_to()` ZÁRT HURKOT használ: parancs → beállás → mérés →
     Jacobian-korrekció. Ugyanaz az elv, mint a GR1T1 v2-nél: a beállt pózra
     tervezünk, nem a parancsoltra.

2. AZ EGYENES VONALÚ KÖZELÍTÉS LELÖKI A TERMÉKET.
   Mérve: naiv közelítésnél a termék 1135 mm-t mozdult — leesett a polcról.
   → `approach_product()` waypointokon megy: előbb a polc elé, a termék
     magasságában, csak utána befelé. A G1 APPROACH → PUSH mintája.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover
    raise SystemExit("pip install mujoco")

_REPO = Path(__file__).resolve().parent.parent
SCENE = _REPO / "src/envs/assets/shelflife_scene_gene01_v1.xml"
TEX   = _REPO / "src/envs/assets/shelflife_textures"

ARM_JOINTS = ["r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
              "r_elbow", "r_wrist_yaw", "r_wrist_pitch", "r_wrist_roll"]
NECK_JOINTS = ["neck_pitch", "neck_roll", "neck_yaw"]
FINGER_TOKENS = ("thumb", "index", "middle", "ring", "little")

# A kéz nyitott és zárt póza. A 21 ujj-ízületet EGY skalár vezérli — ez a
# primitív-réteg dolga, nem az ügynöké (vö. G1: 1 policy DoF → 7 ujj-ízület).
FINGER_OPEN, FINGER_CLOSED = 0.0, 1.05      # rad, minden flex-ízületre

# A "ready" póz tenyér-célpontja: a polc síkja (x=0.34) ELŐTT, a jobb kar
# komfortzónájában. Mérve: erre a pózra menet a termék 0 mm-t mozdul.
READY_PALM_XYZ = np.array([0.15, -0.30, 1.20])

SETTLE_STEPS = 250          # egy parancs beállásához (mérve: 250 elég)
IK_MAX_ITERS = 25
IK_TOL_M     = 0.005        # 5 mm — ennél pontosabb pozicionálás nem kell
IK_DAMPING   = 0.05         # csillapított legkisebb négyzetek (rosszul
                            # kondicionált Jacobiannál stabilabb, mint lstsq)


@dataclass
class StepResult:
    ok: bool
    detail: str = ""
    data: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


class ShelfLifeRobot:
    """GENE.01 primitívek egy bolti polc előtt.

    Az ügynök CSAK ezeket a metódusokat hívja. Ami itt nincs, azt a
    készség-szinten kell megírnia belőlük.
    """

    # ── életciklus ──────────────────────────────────────────────────────────

    def __init__(self, scene: Optional[Path] = None, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_path(str(scene or SCENE))
        self.data = mujoco.MjData(self.model)
        # KÜLÖN állapot a kinematikai számításokhoz.
        # BUG, amibe belefutottunk: ha az IK az élő `self.data`-t írja, a
        # `qpos[:]=0` a TERMÉK freejointját is nullázza → a termék a világ
        # origójába teleportál, és onnan folytatódik a szimuláció. Egy mérésünk
        # emiatt hamis 1152 mm-es „elmozdulást" mutatott. Soha ne számolj FK-t
        # az élő állapoton.
        self._scratch = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self._renderer = None

        jid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        bid = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)

        self._arm_q = [self.model.jnt_qposadr[jid(n)] for n in ARM_JOINTS]
        self._arm_v = [self.model.jnt_dofadr[jid(n)] for n in ARM_JOINTS]
        self._arm_a = [aid(f"act_{n}") for n in ARM_JOINTS]
        self._arm_range = np.array([self.model.jnt_range[jid(n)] for n in ARM_JOINTS])
        self._neck_a = [aid(f"act_{n}") for n in NECK_JOINTS]

        self._finger_a, self._finger_flex = [], []
        for i in range(self.model.nu):
            nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
            if nm.startswith("act_r_") and any(t in nm for t in FINGER_TOKENS):
                self._finger_a.append(i)
                self._finger_flex.append("flex" in nm or nm.endswith(("_2", "_3")))

        self._palm = bid("r_wrist_3")
        self._head = bid("head")
        self._products = [i for i in range(self.model.nbody)
                          if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
                          .startswith("product_")]
        self._manifest = {}
        mf = TEX / "manifest.json"
        if mf.exists():
            self._manifest = json.loads(mf.read_text())
        self.reset()

    def reset(self, seed: Optional[int] = None) -> StepResult:
        """Alaphelyzet. A termék pozíciója/orientációja randomizálható."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._cmd = np.zeros(len(ARM_JOINTS))
        mujoco.mj_forward(self.model, self.data)
        self.step(120)                       # a passzív ízületek beállása
        return StepResult(True, "reset")

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    # ── érzékelés ───────────────────────────────────────────────────────────

    def palm_pose(self) -> np.ndarray:
        return self.data.xpos[self._palm].copy()

    def product_pose(self, idx: int = 0) -> np.ndarray:
        return self.data.xpos[self._products[idx]].copy()

    def product_quat(self, idx: int = 0) -> np.ndarray:
        return self.data.xquat[self._products[idx]].copy()

    def n_products(self) -> int:
        return len(self._products)

    def touching_product(self, idx: int = 0) -> bool:
        """Van-e kontaktus a jobb kéz és a termék között."""
        pb = self._products[idx]
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            for a, b in ((b1, b2), (b2, b1)):
                if b == pb:
                    nm = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, a) or ""
                    if nm.startswith("r_") and any(
                            t in nm for t in FINGER_TOKENS + ("wrist", "hand")):
                        return True
        return False

    def camera_pose(self, camera: str = "rgb_camera"):
        """(pozíció, előre-irány) a megadott kamerára, világkoordinátákban."""
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if i < 0:
            raise ValueError(f"nincs ilyen kamera: {camera}")
        pos = self.data.cam_xpos[i].copy()
        fwd = -self.data.cam_xmat[i].reshape(3, 3)[:, 2]   # MuJoCo: -z a nézet
        return pos, fwd

    def in_view(self, point: Sequence[float],
                camera: str = "rgb_camera") -> StepResult:
        """Benne van-e a pont a kamera látóterében — GEOMETRIAILAG, render nélkül.

        Az ügynöknek ez olcsó ellenőrzés a ReAct-hurokban: mielőtt képet kérne
        és a VLM-et hívná, megnézheti, hogy egyáltalán látszik-e a tárgy.
        Ezzel derült ki, hogy a termék a polcon 32°-kal kilóg a mellkasi kamera
        22.5°-os fél-látószögéből — ezért voltak üresek az első renderek.
        """
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        pos, fwd = self.camera_pose(camera)
        mat = self.data.cam_xmat[i].reshape(3, 3)
        right, up = mat[:, 0], mat[:, 1]
        v = np.asarray(point, float) - pos
        dist = float(np.linalg.norm(v))
        vn = v / max(dist, 1e-9)
        half = float(self.model.cam_fovy[i]) / 2.0
        ah = float(np.degrees(np.arctan2(vn @ right, vn @ fwd)))
        av = float(np.degrees(np.arctan2(vn @ up, vn @ fwd)))
        ok = (vn @ fwd) > 0 and abs(ah) < half and abs(av) < half
        return StepResult(ok,
                          f"{camera}: {'látszik' if ok else 'KILÓG'} "
                          f"(vízsz {ah:+.1f}°, függ {av:+.1f}°, fél-FOV {half:.1f}°, "
                          f"táv {dist:.2f} m)",
                          {"h_deg": ah, "v_deg": av, "half_fov": half,
                           "dist_m": dist, "visible": ok})

    def inspect_pose(self, camera: str = "rgb_camera",
                     distance: float = 0.18,
                     held_offset: float = 0.07) -> StepResult:
        """A KÉZBEN tartott terméket a kamera elé viszi, olvasható méretben.

        MIÉRT KELL: a polcon álló terméket egyik gyári kamera sem látja
        (32°-kal kilóg a mellkasiéból, a fejkamerák 44°-kal fölötte néznek el).
        A megoldás ugyanaz, mint embernél: a tárgyat a szem elé emeljük.

        Mérve, 0.18 m-en: a termék a kép 54%-át tölti ki → a 4 mm-es nyomtatott
        dátum ~17 px 640-es felbontáson, bőven a kimért ~7 px-es olvashatósági
        küszöb felett. A tenyér IK-hibája erre a pontra 2 mm.
        """
        pos, fwd = self.camera_pose(camera)
        product_target = pos + fwd * distance
        palm_target = product_target - fwd * held_offset
        r = self.move_palm_to(palm_target, slices=6)
        vis = self.in_view(self.product_pose(0), camera)
        return StepResult(bool(r) and vis.ok,
                          f"{r.detail} · {vis.detail}",
                          {**r.data, **vis.data})

    def render_view(self, camera: str = "rgb_camera", res: int = 640) -> np.ndarray:
        """Kamerakép a VLM-nek.

        FIGYELEM: OpenGL kell hozzá. Fejlesztői sandboxban általában NINCS —
        ilyenkor beszédes hibát dobunk, mert a néma fekete kép sokkal
        rosszabb lenne (a modell magabiztosan félreolvasna egy üres képet).
        """
        try:
            # BUG volt: a renderert egyszer hoztuk létre és utána a `res`
            # paramétert csendben figyelmen kívül hagytuk — a felbontás-sorozat
            # emiatt öt AZONOS képet adott. Felbontásváltáskor újra kell építeni.
            if self._renderer is None or self._renderer.height != res \
                    or self._renderer.width != res:
                self._renderer = mujoco.Renderer(self.model, res, res)
            self._renderer.update_scene(self.data, camera=camera)
            return self._renderer.render()
        except Exception as e:
            self._renderer = None          # ne maradjon félkész objektum
            msg = str(e)
            if "framebuffer" in msg or "offwidth" in msg:
                # NEM OpenGL-hiba: a scene offscreen buffere kisebb a kértnél.
                raise RuntimeError(
                    f"A kért felbontás ({res}px) nagyobb a scene offscreen "
                    f"bufferénél.\nJavítás: a builderben OFFSCREEN_MAX >= {res}, "
                    f"majd futtasd újra a shelflife_build_scene.py-t.\n"
                    f"Eredeti hiba: {msg}") from e
            raise RuntimeError(
                f"A renderelés nem indul ({type(e).__name__}: {msg}).\n"
                "Ha 'OpenGL platform library has not been loaded' szerepel benne, "
                "akkor hiányzik a grafikus backend (a fejlesztői sandboxban nincs) "
                "— futtasd a saját gépeden. macOS-en az offscreen render sima "
                "python3-mal is megy; csak az interaktív viewer igényel mjpythont "
                "(lásd docs/known_issues.md #1 és #28)."
            ) from e

    # ── mozgás ──────────────────────────────────────────────────────────────

    def _fk_palm(self, q: np.ndarray) -> np.ndarray:
        """Tenyérpozíció adott kar-konfigurációra — a SCRATCH állapoton."""
        mujoco.mj_resetData(self.model, self._scratch)
        for a, v in zip(self._arm_q, q):
            self._scratch.qpos[a] = v
        mujoco.mj_forward(self.model, self._scratch)
        return self._scratch.xpos[self._palm].copy()

    def ik_seed(self, target: Sequence[float]) -> np.ndarray:
        """Kinematikai kar-konfiguráció a célponthoz (durva rács + finomítás).

        Ez csak KIINDULÓPONT: a gravitációs megereszkedés miatt a parancsolt
        póz nem egyenlő a beállttal, ezért utána mindig kell zárt hurkú
        korrekció. Mérve: a kinematikai megoldás 0 mm, a beállt hiba 27 mm.
        """
        target = np.asarray(target, float)
        lo = self._arm_range[:, 0] + 0.10 * (self._arm_range[:, 1] - self._arm_range[:, 0])
        hi = self._arm_range[:, 1] - 0.10 * (self._arm_range[:, 1] - self._arm_range[:, 0])
        best = (1e9, None)
        import itertools
        for q5 in itertools.product(*[np.linspace(lo[i], hi[i], 7) for i in range(5)]):
            q = np.array(list(q5) + [0.0, 0.0])
            e = float(np.linalg.norm(self._fk_palm(q) - target))
            if e < best[0]:
                best = (e, q)
        q, step = best[1].copy(), 0.2
        for _ in range(6):
            improved = True
            while improved:
                improved = False
                for i in range(7):
                    for s in (1, -1):
                        q2 = q.copy()
                        q2[i] = float(np.clip(q2[i] + s * step, lo[i], hi[i]))
                        e = float(np.linalg.norm(self._fk_palm(q2) - target))
                        if e < best[0] - 1e-5:
                            best = (e, q2.copy()); q = q2; improved = True
            step /= 2
        return best[1]

    def ready_pose(self) -> StepResult:
        """A kart a polc SÍKJA ELÉ emeli, szabad térbe.

        MIÉRT KELL: a lógó alaphelyzetből közvetlenül a polcra mozdulva a kar
        átsöpör a terméken és lelöki (mérve: 80–250 mm elmozdulás). Erre a
        köztes pózra menet a termék MÉRHETETLENÜL keveset mozdul (0 mm).
        """
        q = self.ik_seed(READY_PALM_XYZ)
        for a, v in zip(self._arm_a, q):
            self.data.ctrl[a] = v
        self.step(500)
        self._cmd = q.copy()
        return StepResult(True, f"ready, tenyér={np.round(self.palm_pose(), 3)}")

    def move_palm_to(self, target: Sequence[float],
                     tol: float = IK_TOL_M,
                     max_iters: int = IK_MAX_ITERS,
                     slices: int = 1) -> StepResult:
        """A tenyeret a megadott világkoordinátára viszi, ZÁRT HUROKBAN.

        Miért nem nyílt hurokban: a kar gravitációs megereszkedése miatt a
        parancsolt és a beállt póz között ~47 mm eltérés van. Itt a parancsot
        korrigáljuk a MÉRT hiba alapján, amíg a tényleges pozíció be nem áll.

        `slices>1`: a pálya Cartesian szeletekre bontva. Ez akkor kell, ha a
        közelben törékeny/elmozdítható tárgy van — a joint-térbeli ugrás ívben
        halad és átsöpörhet rajta. Mérve: 10 szelettel a termék elmozdulása
        0.0 mm, a végső tenyér-hiba 1.0 mm.
        """
        target = np.asarray(target, float)
        if slices > 1:
            start = self.palm_pose().copy()
            last = None
            for k in range(1, slices + 1):
                wp = start + (target - start) * (k / slices)
                last = self.move_palm_to(wp, tol=max(tol, 0.008),
                                         max_iters=4, slices=1)
            return self.move_palm_to(target, tol=tol, max_iters=max_iters, slices=1)
        hist = []
        for _ in range(max_iters):
            for a, v in zip(self._arm_a, self._cmd):
                self.data.ctrl[a] = v
            self.step(SETTLE_STEPS)
            err = target - self.palm_pose()
            e = float(np.linalg.norm(err))
            hist.append(e)
            if e < tol:
                return StepResult(True, f"cél elérve {e*1000:.1f} mm",
                                  {"iters": len(hist), "err_mm": e * 1000,
                                   "history_mm": [round(h*1000, 1) for h in hist]})
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self._palm)
            Jm = jacp[:, self._arm_v]
            # csillapított legkisebb négyzetek: J^T (J J^T + λ²I)^-1 e
            JJt = Jm @ Jm.T + (IK_DAMPING ** 2) * np.eye(3)
            dq = Jm.T @ np.linalg.solve(JJt, err)
            dq = np.clip(dq, -0.30, 0.30)
            self._cmd = np.clip(self._cmd + dq,
                                self._arm_range[:, 0] + 0.05,
                                self._arm_range[:, 1] - 0.05)
        return StepResult(False, f"nem konvergált, {hist[-1]*1000:.1f} mm",
                          {"iters": len(hist), "err_mm": hist[-1] * 1000,
                           "history_mm": [round(h*1000, 1) for h in hist]})

    def approach_product(self, idx: int = 0,
                         standoff: float = 0.16,
                         grasp_gap: float = 0.07) -> StepResult:
        """Ütközésmentes közelítés a termékhez, waypointokon.

        MIÉRT waypoint: egyenes vonalon a kar átmegy a terméken és lelöki a
        polcról (mérve: 1135 mm elmozdulás). Ezért előbb a polc ELÉ állunk a
        termék magasságában, és csak onnan megyünk befelé.
        """
        start = self.product_pose(idx).copy()
        pre = start + np.array([-standoff, 0.0, 0.02])

        # 1. lépés: ready póz — a polc síkja elé, szabad térbe
        self.ready_pose()
        moved = float(np.linalg.norm(self.product_pose(idx) - start))
        if moved > 0.02:
            return StepResult(False, f"a termék már a ready póznál elmozdult "
                                     f"({moved*1000:.0f} mm)", {"moved_mm": moved*1000})

        # 2. lépés: Cartesian szeletekben a pre-grasp pontig
        r2 = self.move_palm_to(pre, slices=10)
        moved = float(np.linalg.norm(self.product_pose(idx) - start))
        ok = bool(r2) and moved < 0.02
        return StepResult(ok,
                          f"{r2.detail} · termék elmozdulás {moved*1000:.1f} mm",
                          {**r2.data, "moved_mm": moved * 1000,
                           "grasp_gap": grasp_gap})

    def grasp(self, amount: float = 1.0, settle: int = SETTLE_STEPS) -> StepResult:
        """A 21 ujj-ízület EGY skalárral: 0.0 = nyitott, 1.0 = zárt.

        Ez szándékosan primitív, nem ügynök-írta: a fogás finomhangolása
        vezérléstechnika, nem szemantika.
        """
        a = float(np.clip(amount, 0.0, 1.0))
        val = FINGER_OPEN + a * (FINGER_CLOSED - FINGER_OPEN)
        for act, is_flex in zip(self._finger_a, self._finger_flex):
            self.data.ctrl[act] = val if is_flex else 0.0
        self.step(settle)
        return StepResult(True, f"grasp={a:.2f}", {"amount": a})

    def look_at(self, target: Sequence[float], settle: int = 150) -> StepResult:
        """A fejet a megadott pont felé fordítja (nyak pitch/roll/yaw)."""
        t = np.asarray(target, float) - self.data.xpos[self._head]
        yaw = float(np.arctan2(t[1], t[0]))
        pitch = float(-np.arctan2(t[2], np.linalg.norm(t[:2])))
        self.data.ctrl[self._neck_a[0]] = np.clip(pitch, -0.6, 0.6)
        self.data.ctrl[self._neck_a[1]] = 0.0
        self.data.ctrl[self._neck_a[2]] = np.clip(yaw, -1.0, 1.0)
        self.step(settle)
        return StepResult(True, f"look_at yaw={yaw:.2f} pitch={pitch:.2f}")

    def lift(self, dz: float = 0.08) -> StepResult:
        """A tenyeret dz-vel megemeli az aktuális pozícióból."""
        return self.move_palm_to(self.palm_pose() + np.array([0.0, 0.0, dz]))

    def retreat(self, dx: float = 0.15) -> StepResult:
        """Kihúzza a kezet a polcból (-x irányba)."""
        return self.move_palm_to(self.palm_pose() + np.array([-dx, 0.0, 0.0]))

    # ── ground truth (csak az evalhoz, az ügynök NEM hívhatja) ──────────────

    def _ground_truth(self, idx: int = 0) -> dict:
        items = self._manifest.get("items", [])
        return items[idx] if idx < len(items) else {}


if __name__ == "__main__":
    r = ShelfLifeRobot()
    print(f"termékek: {r.n_products()}  ujj-aktuátorok: {len(r._finger_a)}")
    print(f"tenyér: {np.round(r.palm_pose(), 3)}")
    print(f"termék: {np.round(r.product_pose(), 3)}")
