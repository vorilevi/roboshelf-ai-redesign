"""
Product Intelligence Layer (PIL) adatbázis.

Egységes termék-metaadat forrás mindkét Roboshelf trackhez:
  - humanoid track: MJCF asset metaadat (tömeg, súlypont, megfogási zóna)
  - EAN track:      Isaac Lab scene asset metadata + planogram logika

Backend: SQLite (fejlesztés), Postgres-re migrálható production-ban.
Jelenleg: in-memory dict (stub), SQLite integráció Fázis 3-ban.

HEIS kompatibilitás: schema.json definiálja a mezőket, HEIS 2026 Q1 v1.0 szerint.

Használat:
    from roboshelf_common.product_intelligence_layer import ProductIntelligenceDB

    db = ProductIntelligenceDB()
    product = db.get_by_ean("5900617004284")
    grasp   = db.get_grasp_zone("5900617004284")
    slot    = db.get_planogram_slot("5900617004284", store_id="demo_store")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PILProduct:
    """Egy termék teljes PIL rekordja.

    Minden mező HEIS 2026 Q1 v1.0 schema.json alapján.
    """
    ean: str
    name_hu: str
    mass_kg: float
    center_of_mass_xyz: list[float]          # [x, y, z] méterben
    bounding_box_xyz: list[float]            # [szélesség, mélység, magasság] méterben
    grasp_zone: str                          # ld. schema.json enum
    compliance: float                        # 0.0 (merev) – 1.0 (puha)
    expiry_ocr_position: str                 # ld. schema.json enum
    planogram_slot: str = ""
    sku: str = ""
    fragile: bool = False

    def to_dict(self) -> dict:
        return {
            "ean": self.ean,
            "sku": self.sku,
            "name_hu": self.name_hu,
            "mass_kg": self.mass_kg,
            "center_of_mass_xyz": self.center_of_mass_xyz,
            "bounding_box_xyz": self.bounding_box_xyz,
            "grasp_zone": self.grasp_zone,
            "compliance": self.compliance,
            "expiry_ocr_position": self.expiry_ocr_position,
            "planogram_slot": self.planogram_slot,
            "fragile": self.fragile,
        }


# ---------------------------------------------------------------------------
# Seed adatok — 12 termékkategória (fejlesztési és tesztelési célra)
# EAN-ok: valós formátum, de fiktív adatok
# ---------------------------------------------------------------------------
_SEED_PRODUCTS: list[dict] = [
    # Tejtermékek
    {
        "ean": "5900617004284", "sku": "MILK-1L-UHT",
        "name_hu": "UHT tej 1L",
        "mass_kg": 1.03, "center_of_mass_xyz": [0.0, 0.0, 0.065],
        "bounding_box_xyz": [0.072, 0.072, 0.195],
        "grasp_zone": "upper_cylindrical", "compliance": 0.10,
        "expiry_ocr_position": "top_lid", "planogram_slot": "DAIRY-SHELF-A-TOP",
        "fragile": False,
    },
    {
        "ean": "5900617004291", "sku": "YOGURT-150G",
        "name_hu": "Joghurt 150g",
        "mass_kg": 0.16, "center_of_mass_xyz": [0.0, 0.0, 0.04],
        "bounding_box_xyz": [0.08, 0.08, 0.07],
        "grasp_zone": "flat_top", "compliance": 0.35,
        "expiry_ocr_position": "top_lid", "planogram_slot": "DAIRY-SHELF-B-MID",
        "fragile": False,
    },
    # Üdítők
    {
        "ean": "5449000214911", "sku": "COLA-500ML",
        "name_hu": "Cola 0.5L PET",
        "mass_kg": 0.54, "center_of_mass_xyz": [0.0, 0.0, 0.11],
        "bounding_box_xyz": [0.065, 0.065, 0.22],
        "grasp_zone": "upper_cylindrical", "compliance": 0.05,
        "expiry_ocr_position": "bottom", "planogram_slot": "DRINKS-SHELF-A-MID",
        "fragile": False,
    },
    {
        "ean": "5449000214928", "sku": "COLA-1.5L",
        "name_hu": "Cola 1.5L PET",
        "mass_kg": 1.62, "center_of_mass_xyz": [0.0, 0.0, 0.17],
        "bounding_box_xyz": [0.085, 0.085, 0.34],
        "grasp_zone": "upper_cylindrical", "compliance": 0.05,
        "expiry_ocr_position": "bottom", "planogram_slot": "DRINKS-SHELF-A-LOW",
        "fragile": False,
    },
    # Konzervek
    {
        "ean": "5010021002258", "sku": "TUNA-CAN-185G",
        "name_hu": "Tonhal konzerv 185g",
        "mass_kg": 0.22, "center_of_mass_xyz": [0.0, 0.0, 0.025],
        "bounding_box_xyz": [0.085, 0.085, 0.05],
        "grasp_zone": "side_grip", "compliance": 0.0,
        "expiry_ocr_position": "bottom", "planogram_slot": "CANNED-SHELF-B-MID",
        "fragile": False,
    },
    # Kenyér / pékáru
    {
        "ean": "5998900001113", "sku": "BREAD-TOAST-500G",
        "name_hu": "Toast kenyér 500g",
        "mass_kg": 0.51, "center_of_mass_xyz": [0.0, 0.0, 0.06],
        "bounding_box_xyz": [0.12, 0.23, 0.12],
        "grasp_zone": "side_grip", "compliance": 0.75,
        "expiry_ocr_position": "side_left", "planogram_slot": "BAKERY-SHELF-A-MID",
        "fragile": False,
    },
    # Tojás
    {
        "ean": "5998901112204", "sku": "EGGS-10PC",
        "name_hu": "Tojás 10 db",
        "mass_kg": 0.62, "center_of_mass_xyz": [0.0, 0.0, 0.045],
        "bounding_box_xyz": [0.195, 0.12, 0.09],
        "grasp_zone": "flat_top", "compliance": 0.0,
        "expiry_ocr_position": "side_right", "planogram_slot": "DAIRY-SHELF-C-LOW",
        "fragile": True,
    },
    # Tisztítószerek
    {
        "ean": "8710908015229", "sku": "DETERGENT-1L",
        "name_hu": "Folyékony mosószer 1L",
        "mass_kg": 1.08, "center_of_mass_xyz": [0.0, 0.0, 0.12],
        "bounding_box_xyz": [0.09, 0.065, 0.24],
        "grasp_zone": "upper_cylindrical", "compliance": 0.12,
        "expiry_ocr_position": "back_bottom", "planogram_slot": "HOUSEHOLD-SHELF-A-TOP",
        "fragile": False,
    },
    # Chips / snack
    {
        "ean": "5900259030011", "sku": "CHIPS-100G",
        "name_hu": "Chips 100g",
        "mass_kg": 0.11, "center_of_mass_xyz": [0.0, 0.0, 0.08],
        "bounding_box_xyz": [0.175, 0.045, 0.235],
        "grasp_zone": "side_grip", "compliance": 0.85,
        "expiry_ocr_position": "back_bottom", "planogram_slot": "SNACK-SHELF-B-TOP",
        "fragile": False,
    },
    # Befőtt / üveges
    {
        "ean": "5998876501112", "sku": "JAM-STRAWBERRY-400G",
        "name_hu": "Eper lekvár 400g üveg",
        "mass_kg": 0.48, "center_of_mass_xyz": [0.0, 0.0, 0.055],
        "bounding_box_xyz": [0.075, 0.075, 0.11],
        "grasp_zone": "side_grip", "compliance": 0.0,
        "expiry_ocr_position": "top_lid", "planogram_slot": "JAM-SHELF-A-MID",
        "fragile": True,
    },
    # Kávé
    {
        "ean": "8711000375761", "sku": "COFFEE-250G",
        "name_hu": "Instant kávé 250g",
        "mass_kg": 0.28, "center_of_mass_xyz": [0.0, 0.0, 0.07],
        "bounding_box_xyz": [0.085, 0.085, 0.14],
        "grasp_zone": "side_grip", "compliance": 0.0,
        "expiry_ocr_position": "bottom", "planogram_slot": "COFFEE-SHELF-A-TOP",
        "fragile": False,
    },
    # Rizs
    {
        "ean": "5998876502003", "sku": "RICE-1KG",
        "name_hu": "Hosszúszemű rizs 1kg",
        "mass_kg": 1.02, "center_of_mass_xyz": [0.0, 0.0, 0.055],
        "bounding_box_xyz": [0.135, 0.065, 0.245],
        "grasp_zone": "side_grip", "compliance": 0.20,
        "expiry_ocr_position": "back_bottom", "planogram_slot": "DRY-SHELF-B-LOW",
        "fragile": False,
    },
]


class ProductIntelligenceDB:
    """PIL adatbázis — in-memory stub, SQLite backend Fázis 3-ban.

    Mindkét track ugyanebből olvas:
      - humanoid MJCF: tömeg, súlypont, megfogási zóna, compliance
      - EAN Isaac Lab: planogram slot, EAN, fragile flag
    """

    def __init__(self, backend: str = "memory") -> None:
        """
        Args:
            backend: "memory" (stub) | "sqlite" (Fázis 3-ban implementálni)
        """
        self.backend = backend
        self._data: dict[str, PILProduct] = {}
        self._load_seed_data()

    def _load_seed_data(self) -> None:
        for row in _SEED_PRODUCTS:
            product = PILProduct(
                ean=row["ean"],
                sku=row.get("sku", ""),
                name_hu=row["name_hu"],
                mass_kg=row["mass_kg"],
                center_of_mass_xyz=row["center_of_mass_xyz"],
                bounding_box_xyz=row["bounding_box_xyz"],
                grasp_zone=row["grasp_zone"],
                compliance=row["compliance"],
                expiry_ocr_position=row["expiry_ocr_position"],
                planogram_slot=row.get("planogram_slot", ""),
                fragile=row.get("fragile", False),
            )
            self._data[product.ean] = product

    # ------------------------------------------------------------------
    # Lekérdező metódusok
    # ------------------------------------------------------------------

    def get_by_ean(self, ean: str) -> Optional[PILProduct]:
        """Teljes PIL rekord EAN alapján."""
        return self._data.get(ean)

    def get_mass(self, ean: str) -> float:
        """Tömeg kg-ban (MJCF inertia számításhoz)."""
        p = self._data.get(ean)
        return p.mass_kg if p else 0.1  # default 100g ha ismeretlen

    def get_grasp_zone(self, ean: str) -> str:
        """Optimális megfogási zóna (manipulation policy reward shaping-hez)."""
        p = self._data.get(ean)
        return p.grasp_zone if p else "side_grip"

    def get_compliance(self, ean: str) -> float:
        """Rugalmassági faktor (0=merev, 1=puha) — megfogási erő skálázásához."""
        p = self._data.get(ean)
        return p.compliance if p else 0.1

    def get_center_of_mass(self, ean: str) -> list[float]:
        """Súlypont a termék koordinátarendszerében (m)."""
        p = self._data.get(ean)
        return p.center_of_mass_xyz if p else [0.0, 0.0, 0.05]

    def get_planogram_slot(self, ean: str, store_id: str = "default") -> str:
        """Planogram slot kód (EAN track: Isaac Lab scene builder-hez).

        Args:
            ean:      termék EAN
            store_id: bolt azonosító (Fázis 3-ban store-specifikus planogramok)
        """
        p = self._data.get(ean)
        # TODO (Fázis 3): store_id alapján különböző planogramok
        return p.planogram_slot if p else "UNKNOWN-SLOT"

    def is_fragile(self, ean: str) -> bool:
        """Törékeny termék flag (Human-Safe Nav extra óvatossági mód)."""
        p = self._data.get(ean)
        return p.fragile if p else False

    def list_all_eans(self) -> list[str]:
        """Az adatbázisban lévő összes EAN listája."""
        return list(self._data.keys())

    def add_product(self, product: PILProduct) -> None:
        """Új termék hozzáadása (fejlesztési/tesztelési célra)."""
        self._data[product.ean] = product

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ProductIntelligenceDB(backend={self.backend!r}, products={len(self)})"
