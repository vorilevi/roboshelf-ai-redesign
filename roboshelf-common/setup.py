"""
setup.py — roboshelf-common telepítés

Fejlesztési módban (editable install):
    pip install -e roboshelf-common/ --break-system-packages

Ezután mindkét trackből importálható:
    from roboshelf_common.heis_adapter import HEISAdapter
    from roboshelf_common.vla_client import VLAClient
    from roboshelf_common.product_intelligence_layer import ProductIntelligenceDB
"""

from setuptools import setup, find_packages

# A mappa neve 'roboshelf-common' (kötőjel), de Python package névként
# 'roboshelf_common' (aláhúzás) kell. A package_dir mapping ezt oldja meg:
# a 'roboshelf_common' csomag gyökere a saját mappa (ahol a __init__.py van).
setup(
    name="roboshelf-common",
    version="0.1.0",
    description="Közös komponensek a Roboshelf AI két trackjéhez (Phase 030)",
    packages=[
        "roboshelf_common",
        "roboshelf_common.heis_adapter",
        "roboshelf_common.vla_client",
        "roboshelf_common.product_intelligence_layer",
        "roboshelf_common.lerobot_pipeline",
    ],
    package_dir={"roboshelf_common": "."},
    python_requires=">=3.10",
    install_requires=[],
)
