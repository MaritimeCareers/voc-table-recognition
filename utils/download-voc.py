#!/usr/bin/env python3

import os
import sys
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

# Namespaces and URLs
XML_NAMESPACE = {'ead': 'urn:isbn:1-931666-00-8'}
METS_NAMESPACE = {'mets': 'http://www.loc.gov/METS/', 'xlink': 'http://www.w3.org/1999/xlink'}
XML_DOWNLOAD_URL = "https://www.nationaalarchief.nl/onderzoeken/archief/1.04.02/download/xml"
XML_FILENAME = "1.04.02.xml"

def ensure_1_04_02_xml():
    # Make sure 1.04.02.xml is present, otherwise download
    if not Path(XML_FILENAME).exists():
        print(f"Downloading {XML_FILENAME}...")
        r = requests.get(XML_DOWNLOAD_URL)
        r.raise_for_status()
        with open(XML_FILENAME, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {XML_FILENAME}.")

def parse_unitid_mets(file_path: str) -> Dict[str, str]:
    # Parse the big EAD file and collect unitid -> METS URL
    t = ET.parse(file_path)
    r = t.getroot()
    out = {}
    for did in r.findall(".//did", XML_NAMESPACE):
        uid = did.find("unitid", XML_NAMESPACE)
        mets = did.find("dao", XML_NAMESPACE)
        if uid is not None and mets is not None:
            out[uid.text] = mets.attrib.get('href')
    return out

def extract_inventory_number(filename: str) -> str:
    # Get the inventory number from filename
    try:
        return filename.split("1.04.02_")[1].split("_")[0]
    except IndexError:
        raise ValueError(f"Invalid filename format: {filename}")

def download_mets(url: str, path: Path):
    # Download one METS XML
    r = requests.get(url)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded METS XML to {path}")

def find_image_url(mets_file: Path, label_without_ext: str) -> Optional[str]:
    # Look for the correct LABEL in the METS div elements
    t = ET.parse(mets_file)
    r = t.getroot()
    for ext in (".tif", ".jpg"):
        label = label_without_ext + ext
        div_el = r.find(f".//mets:div[@LABEL='{label}']", METS_NAMESPACE)
        if div_el is not None:
            div_id = div_el.attrib.get("ID")
            if div_id:
                file_el = r.find(f".//mets:file[@ID='{div_id}DEF']", METS_NAMESPACE)
                if file_el is not None:
                    flocat = file_el.find("mets:FLocat", METS_NAMESPACE)
                    if flocat is not None:
                        return flocat.attrib.get("{http://www.w3.org/1999/xlink}href")
    return None

def rename_files(source_dir: Path) -> list[Path]:
    # Remove 4-digit prefix + underscore from filenames
    renamed = []
    for f in source_dir.glob("*.xml"):
        if f.name[:4].isdigit() and f.name[4] == "_":
            new_name = f.name[5:]  # Remove the first 5 characters (4 digits + '_')
            new_path = source_dir / new_name
            f.rename(new_path)
            renamed.append(new_path)
            print(f"Renamed {f.name} => {new_name}")
        else:
            renamed.append(f)
    return renamed

def process_file(xml_file: Path, unitid_mets: Dict[str, str], target_dir: Path, processed_mets: set):
    # Download image for one XML file
    inv_no = extract_inventory_number(xml_file.name)
    mets_url = unitid_mets.get(inv_no)
    if not mets_url:
        print(f"Skipping {xml_file.name}: no METS URL for inv. no. {inv_no}.")
        return

    mets_file = target_dir / f"{inv_no}.xml"
    if inv_no not in processed_mets:
        print(f"Downloading METS XML for {inv_no}...")
        download_mets(mets_url, mets_file)
        processed_mets.add(inv_no)

    label = xml_file.stem
    try:
        image_url = find_image_url(mets_file, label)
        if not image_url:
            print(f"Error for {xml_file.name}: no METS div matches LABEL '{label}'.")
            return

        image_path = target_dir / f"{label}.jpg"
        if image_path.exists():
            print(f"Already exists: {image_path}")
            return

        print(f"Downloading image for {xml_file.name}...")
        r = requests.get(image_url)
        r.raise_for_status()
        with open(image_path, "wb") as f:
            f.write(r.content)
        print(f"Saved image to {image_path}")
    except ValueError as e:
        print(f"Error for {xml_file.name}: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: download-voc.py source_dir/ target_dir/")
        sys.exit(1)

    source_dir = Path(sys.argv[1])
    target_dir = Path(sys.argv[2])

    if not source_dir.is_dir():
        print(f"Source directory does not exist: {source_dir}")
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)

    # First rename incorrectly named files
    renamed_files = rename_files(source_dir)

    # Make sure 1.04.02.xml is present
    ensure_1_04_02_xml()
    unitid_mets = parse_unitid_mets(XML_FILENAME)

    processed_mets = set()
    for xml_file in renamed_files:
        process_file(xml_file, unitid_mets, target_dir, processed_mets)

    # Remove downloaded METS files
    for mf in target_dir.glob("*.xml"):
        mf.unlink()
        print(f"Deleted METS file: {mf}")

if __name__ == "__main__":
    main()
