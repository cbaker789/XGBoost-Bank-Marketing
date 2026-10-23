"""Download the original UCI archive; requires only Python's standard library."""
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib
import json

ROOT = Path(__file__).resolve().parent
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def main():
    destination = ROOT / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    (destination / "bank-marketing.zip").write_bytes(payload)
    with ZipFile(BytesIO(payload)) as outer:
        member = next(name for name in outer.namelist() if name.endswith("bank-additional.zip"))
        with ZipFile(BytesIO(outer.read(member))) as inner:
            for filename in ("bank-additional-full.csv", "bank-additional-names.txt"):
                source = next(name for name in inner.namelist() if name.endswith("/" + filename) or name == filename)
                (destination / filename).write_bytes(inner.read(source))
    metadata = {
        "source": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "download_url": URL,
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_file": "bank-additional-full.csv",
        "license": "CC BY 4.0",
        "citation": "Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306",
    }
    (destination / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Downloaded dataset and original data dictionary to {destination}")


if __name__ == "__main__":
    main()
