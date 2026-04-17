"""Helpers for building combined MuJoCo XML scenes."""

from pathlib import Path
import copy
import xml.etree.ElementTree as ET


def _read_xml(path: Path) -> ET.ElementTree:
    return ET.parse(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _ensure_child(root: ET.Element, tag: str) -> ET.Element:
    child = root.find(tag)
    if child is None:
        child = ET.Element(tag)
        root.append(child)
    return child


def build_combined_scene_xml(g1_xml_path: str, store_xml_path: str) -> str:
    g1_path = Path(g1_xml_path).expanduser().resolve()
    store_path = Path(store_xml_path).expanduser().resolve()

    g1_tree = _read_xml(g1_path)
    store_tree = _read_xml(store_path)

    g1_root = g1_tree.getroot()
    store_root = store_tree.getroot()

    g1_default = _ensure_child(g1_root, "default")
    g1_asset = _ensure_child(g1_root, "asset")
    g1_worldbody = _ensure_child(g1_root, "worldbody")

    store_default = store_root.find("default")
    store_asset = store_root.find("asset")
    store_worldbody = store_root.find("worldbody")

    if store_worldbody is None:
        raise ValueError("Store XML: missing <worldbody>")

    if store_default is not None:
        for child in list(store_default):
            g1_default.append(copy.deepcopy(child))

    if store_asset is not None:
        for child in list(store_asset):
            g1_asset.append(copy.deepcopy(child))

    for child in list(store_worldbody):
        g1_worldbody.append(copy.deepcopy(child))

    return ET.tostring(g1_root, encoding="unicode")


def write_combined_scene_xml(g1_xml_path: str, store_xml_path: str, output_xml_path: str) -> str:
    output_path = Path(output_xml_path).expanduser().resolve()
    xml = build_combined_scene_xml(g1_xml_path, store_xml_path)
    _write_text(output_path, xml)
    return str(output_path)
