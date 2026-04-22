"""
VLA A/B/C teszt — közös kiértékelő script mindhárom modellhez.

Használat:
    # Egy modell kiértékelése:
    python scripts/eval_vla_abc.py --model wall-oss --episodes 50

    # Mind a három egymás után (teljes A/B/C teszt):
    python scripts/eval_vla_abc.py --model all --episodes 50

    # Gyors smoke test:
    python scripts/eval_vla_abc.py --model wall-oss --episodes 5 --stub

Kimenet:
    results/vla_abc/<model>/metrics.json   — EIbench-kompatibilis metrikák
    results/vla_abc/<model>/episodes.csv   — epizód-szintű adatok
    results/vla_abc/summary.csv            — összesítő táblázat (composite score)

Composite score = success_rate × (1 / normalized_latency_ms)
Döntési szabály: legmagasabb composite score nyer.
Ha <5% különbség → WALL-OSS preferált (geopolitikai diverzifikáció).

Phase: 030 — F4-ben futtatjuk Vast.ai A100-on, de lokálisan is futtatható stub_mode-ban.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

# roboshelf_common importok
sys.path.insert(0, str(Path(__file__).parent.parent / "roboshelf-common"))
try:
    from roboshelf_common.vla_client import VLAClient, VLAModel
    from roboshelf_common.heis_adapter import HEISAdapter
    from roboshelf_common.product_intelligence_layer import ProductIntelligenceDB
except ImportError:
    # Ha editable install van:
    from vla_client import VLAClient, VLAModel
    from heis_adapter import HEISAdapter
    from product_intelligence_layer import ProductIntelligenceDB


# ---------------------------------------------------------------------------
# Adatstruktúrák
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode_id: int
    model: str
    success: bool
    completion_time_s: float
    energy_efficiency: float
    inference_latency_ms: float
    min_human_distance_m: float = float("inf")


@dataclass
class ModelResult:
    model: str
    n_episodes: int
    success_rate: float
    avg_completion_time_s: float
    avg_energy_efficiency: float
    avg_inference_latency_ms: float
    composite_score: float
    episodes: list[EpisodeResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Kiértékelő
# ---------------------------------------------------------------------------

class VLAEvaluator:
    """Egy VLA modellt értékel ki N epizódon."""

    def __init__(
        self,
        model: str,
        n_episodes: int,
        stub_mode: bool = True,
        output_dir: Path = Path("results/vla_abc"),
        track: str = "humanoid",
    ):
        self.model = model
        self.n_episodes = n_episodes
        self.stub_mode = stub_mode
        self.output_dir = output_dir / model.replace("-", "_")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.track = track

        self.heis_adapter = HEISAdapter(track=track)
        self.pil_db = ProductIntelligenceDB()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.vla_client = VLAClient(
                model=model,
                device="cuda" if not stub_mode else "cpu",
                dtype="bfloat16",
                stub_mode=stub_mode,
            )
        self.vla_client.load()

    def run_episode(self, episode_id: int) -> EpisodeResult:
        """Egyetlen epizód futtatása.

        Stub módban szimulált eredmények — valós futtatáshoz az
        unitree_rl_mjlab G1 pick-place environment kell (F1 után).
        """
        if self.stub_mode:
            return self._stub_episode(episode_id)
        else:
            return self._real_episode(episode_id)

    def _stub_episode(self, episode_id: int) -> EpisodeResult:
        """Szimulált epizód stub módban — modell-specifikus véletlen eredmények.

        A stub értékek a dokumentált modell-karakterisztikák alapján:
          WALL-OSS:    magasabb success rate, alacsonyabb latencia (flow-matching)
          UnifoLM:     közepes success rate, G1-specifikus
          GR00T N1.6:  stabil baseline, magasabb latencia (diffúziós transzformer)
        """
        import random
        rng = random.Random(episode_id * 1000 + hash(self.model) % 10000)

        # Modell-specifikus valószínűség eloszlások (stub — tényleges A/B/C tesztnél felülírja)
        profiles = {
            VLAModel.WALL_OSS.value:     {"success_p": 0.74, "latency_mu": 45, "latency_sd": 8},
            VLAModel.UNIFOLM_VLA0.value: {"success_p": 0.68, "latency_mu": 55, "latency_sd": 10},
            VLAModel.GROOT_N16.value:    {"success_p": 0.71, "latency_mu": 72, "latency_sd": 12},
        }
        p = profiles.get(self.model, {"success_p": 0.5, "latency_mu": 60, "latency_sd": 15})

        success = rng.random() < p["success_p"]
        latency = max(10.0, rng.gauss(p["latency_mu"], p["latency_sd"]))
        completion_time = rng.uniform(4.5, 9.0) if success else rng.uniform(8.0, 15.0)
        energy = rng.uniform(0.8, 2.5)

        # VLA predict hívás latencia mérése (stub)
        t0 = time.perf_counter()
        _ = self.vla_client.predict(
            observation={"image": None, "joint_pos": [0.0] * 12},
            language_instruction="place product on shelf",
        )
        measured_latency = (time.perf_counter() - t0) * 1000

        return EpisodeResult(
            episode_id=episode_id,
            model=self.model,
            success=success,
            completion_time_s=completion_time,
            energy_efficiency=energy,
            inference_latency_ms=measured_latency if not self.stub_mode else latency,
            min_human_distance_m=rng.uniform(0.6, 2.5),
        )

    def _real_episode(self, episode_id: int) -> EpisodeResult:
        """Valós epizód — unitree_rl_mjlab G1 pick-place env-ben.

        TODO (F4): implementálni a következő lépésekkel:
          1. env = G1PickPlaceEnv(scene="retail_v1", seed=episode_id)
          2. obs, _ = env.reset()
          3. while not done:
               vla_result = self.vla_client.predict(obs, instruction)
               action = vla_result["action"]
               obs, reward, done, _, info = env.step(action)
          4. EIbench metrikák kiszámítása az epizód logból
        """
        raise NotImplementedError(
            "Valós epizód futtatás F4-ben implementálni. "
            "Addig használd: --stub flag"
        )

    def evaluate(self) -> ModelResult:
        """N epizód futtatása és összesítő ModelResult visszaadása."""
        print(f"\n[{self.model}] {self.n_episodes} epizód kiértékelése...")
        episodes: list[EpisodeResult] = []

        for i in range(self.n_episodes):
            ep = self.run_episode(i)
            episodes.append(ep)
            status = "✅" if ep.success else "❌"
            print(
                f"  ep {i+1:3d}/{self.n_episodes} {status} "
                f"t={ep.completion_time_s:.1f}s  "
                f"lat={ep.inference_latency_ms:.1f}ms"
            )

        # Összesítők
        n = len(episodes)
        success_rate = sum(e.success for e in episodes) / n
        avg_time = sum(e.completion_time_s for e in episodes) / n
        avg_energy = sum(e.energy_efficiency for e in episodes) / n
        avg_latency = sum(e.inference_latency_ms for e in episodes) / n

        # Composite score: success_rate / normalized_latency
        # Normalizálás: 100ms = 1.0 referencia
        normalized_latency = avg_latency / 100.0
        composite = success_rate / normalized_latency if normalized_latency > 0 else 0.0

        result = ModelResult(
            model=self.model,
            n_episodes=n,
            success_rate=success_rate,
            avg_completion_time_s=avg_time,
            avg_energy_efficiency=avg_energy,
            avg_inference_latency_ms=avg_latency,
            composite_score=composite,
            episodes=episodes,
        )

        self._save_results(result)
        return result

    def _save_results(self, result: ModelResult) -> None:
        # JSON metrikák
        metrics = {
            "model": result.model,
            "n_episodes": result.n_episodes,
            "success_rate": result.success_rate,
            "avg_completion_time_s": result.avg_completion_time_s,
            "avg_energy_efficiency": result.avg_energy_efficiency,
            "avg_inference_latency_ms": result.avg_inference_latency_ms,
            "composite_score": result.composite_score,
            "heis_version": "1.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )

        # CSV epizód-szint
        csv_path = self.output_dir / "episodes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode_id", "model", "success", "completion_time_s",
                    "energy_efficiency", "inference_latency_ms", "min_human_distance_m",
                ],
            )
            writer.writeheader()
            for ep in result.episodes:
                writer.writerow(asdict(ep))

        print(f"  Eredmények mentve: {self.output_dir}/")


# ---------------------------------------------------------------------------
# Összesítő + döntési logika
# ---------------------------------------------------------------------------

def print_summary(results: list[ModelResult], output_dir: Path) -> str:
    """Összesítő táblázat + VLA döntés."""
    print("\n" + "=" * 65)
    print("VLA A/B/C TESZT — ÖSSZESÍTŐ")
    print("=" * 65)
    print(f"{'Modell':<20} {'Success%':>8} {'Lat(ms)':>8} {'Energy':>8} {'Composite':>10}")
    print("-" * 65)

    best = max(results, key=lambda r: r.composite_score)

    for r in sorted(results, key=lambda r: r.composite_score, reverse=True):
        marker = " ← NYERTES" if r == best else ""
        print(
            f"{r.model:<20} "
            f"{r.success_rate*100:>7.1f}% "
            f"{r.avg_inference_latency_ms:>8.1f} "
            f"{r.avg_energy_efficiency:>8.3f} "
            f"{r.composite_score:>10.3f}"
            f"{marker}"
        )

    # 5%-os küszöb ellenőrzés
    sorted_results = sorted(results, key=lambda r: r.composite_score, reverse=True)
    winner = sorted_results[0]
    second = sorted_results[1] if len(sorted_results) > 1 else None

    print("\n[Döntési logika]")
    if second and (winner.composite_score - second.composite_score) / winner.composite_score < 0.05:
        # Kevesebb mint 5% különbség → WALL-OSS preferált
        wall_oss_result = next((r for r in results if r.model == VLAModel.WALL_OSS.value), None)
        if wall_oss_result:
            final_winner = VLAModel.WALL_OSS.value
            print(f"  <5% különbség a top 2 között → WALL-OSS preferált (geopolitikai diverzifikáció)")
        else:
            final_winner = winner.model
    else:
        final_winner = winner.model
        print(f"  Egyértelmű nyertes: {winner.composite_score:.3f} vs {second.composite_score:.3f}")

    print(f"\n  🏆 KIVÁLASZTOTT VLA: {final_winner}")
    print(f"     → Fine-tune ezt a modellt retail-specifikus adatokon (F5)")
    print(f"     → Frissítsd: roboshelf-common/vla_client/client.py DEFAULT_MODEL")

    # Summary CSV
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "n_episodes", "success_rate", "avg_inference_latency_ms",
                        "avg_energy_efficiency", "composite_score"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r.model,
                "n_episodes": r.n_episodes,
                "success_rate": r.success_rate,
                "avg_inference_latency_ms": r.avg_inference_latency_ms,
                "avg_energy_efficiency": r.avg_energy_efficiency,
                "composite_score": r.composite_score,
            })
    print(f"\n  Összesítő mentve: {summary_path}")

    return final_winner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VLA A/B/C teszt — EIbench kiértékelés")
    p.add_argument(
        "--model",
        choices=["wall-oss", "unifolm-vla-0", "groot-n1.6", "all"],
        default="all",
        help="Kiértékelendő modell (all = mindhárom egymás után)",
    )
    p.add_argument("--episodes", type=int, default=50, help="Epizódok száma modellenként")
    p.add_argument("--stub", action="store_true", default=True,
                   help="Stub mód (F1-F3 között, valós env nélkül)")
    p.add_argument("--no-stub", dest="stub", action="store_false",
                   help="Valós inference (F4-ben, Vast.ai A100-on)")
    p.add_argument("--output", default="results/vla_abc", help="Kimeneti mappa")
    p.add_argument("--track", default="humanoid", choices=["humanoid", "ean"])
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = (
        [VLAModel.WALL_OSS.value, VLAModel.UNIFOLM_VLA0.value, VLAModel.GROOT_N16.value]
        if args.model == "all"
        else [args.model]
    )

    mode = "STUB" if args.stub else "VALÓS"
    print(f"VLA A/B/C teszt — {mode} mód — {args.episodes} epizód/modell")
    if args.stub:
        print("⚠️  Stub mód: szimulált eredmények. Valós tesztre használd: --no-stub")

    results: list[ModelResult] = []
    for model in models:
        evaluator = VLAEvaluator(
            model=model,
            n_episodes=args.episodes,
            stub_mode=args.stub,
            output_dir=output_dir,
            track=args.track,
        )
        result = evaluator.evaluate()
        results.append(result)

    if len(results) > 1:
        winner = print_summary(results, output_dir)
    else:
        r = results[0]
        print(f"\n{r.model}: success={r.success_rate:.1%}, latency={r.avg_inference_latency_ms:.1f}ms, composite={r.composite_score:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
