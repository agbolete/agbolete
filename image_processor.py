"""EasyOCR preprocessing and row parser for Virtual Football screenshots."""

from __future__ import annotations

import re
from functools import lru_cache
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
SUPPORTED_TEAM_CODES = frozenset(
    "MNC LIV ARS CHE MUN TOT NEW AST BHA WHU WOL FUL CRY BOU EVE NFO BRE LUT BUR SHU "
    "RMA BAR ATM SEV VIL SOC BIL VAL MIL INT JUV NAP ROM LAZ ATA FIO DOR LEI LEV FRA WOB".split()
)
TEAM_CODE_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{3,4}(?![A-Z0-9])")
DECIMAL_ODDS_RE = re.compile(r"(?<![\w.])\d{1,3}[.,]\d{1,3}(?![\w.])")
NON_TEAM_CODES = {
    "THE", "VS", "AND", "WIN", "ODD", "HOM", "AWY", "DRW", "OVR", "UND",
    "FOR", "BET", "NOW", "HTH", "FTS", "HAL", "COR", "FT", "HT", "OVER",
}
LEAGUE_HEADINGS = {
    "england": ("premier league", "england", "english league"),
    "spain": ("la liga", "laliga", "spain", "spanish league"),
    "italy": ("serie a", "italy", "italian league"),
    "germany": ("bundesliga", "germany", "german league"),
}


class ScreenshotProcessingError(ValueError):
    """Raised when a screenshot is invalid or OCR cannot run."""


def _to_bytes(image_buffer: Any) -> bytes:
    if isinstance(image_buffer, bytes):
        data = image_buffer
    elif isinstance(image_buffer, bytearray):
        data = bytes(image_buffer)
    elif hasattr(image_buffer, "getvalue"):
        data = image_buffer.getvalue()
    elif hasattr(image_buffer, "read"):
        data = image_buffer.read()
    else:
        raise ScreenshotProcessingError("Upload a PNG, JPEG, or WebP image.")

    if not data:
        raise ScreenshotProcessingError("The uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ScreenshotProcessingError("Screenshot must be 10 MB or smaller.")
    return data


def preprocess_image(image_buffer: Any) -> Image.Image:
    """Decode, orient, grayscale, enhance, and threshold a screenshot."""
    data = _to_bytes(image_buffer)
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise ScreenshotProcessingError("Upload a PNG, JPEG, or WebP screenshot.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ScreenshotProcessingError("Screenshot dimensions are too large to process safely.")
            image = ImageOps.exif_transpose(source).convert("L")
            image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ScreenshotProcessingError("The uploaded file is not a readable image.") from exc

    image = ImageOps.autocontrast(image, cutoff=1)
    image = ImageEnhance.Contrast(image).enhance(1.7)
    if max(image.size) < 1600:
        scale = min(2.5, 1600 / max(image.size))
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return image.point(lambda value: 255 if value >= 150 else 0, mode="L")


@lru_cache(maxsize=1)
def _get_reader() -> Any:
    try:
        import easyocr

        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception as exc:
        raise ScreenshotProcessingError(
            "EasyOCR could not load its English model. Check network access on first run "
            "and confirm easyocr is installed."
        ) from exc


def _group_into_rows(
    detections: list[tuple[list[list[float]], str, float]], image_height: int
) -> list[list[tuple[float, str, float]]]:
    tokens: list[tuple[float, float, str, float]] = []
    for box, raw_text, raw_confidence in detections:
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            confidence = float(raw_confidence)
            center_x = sum(float(point[0]) for point in box) / len(box)
            center_y = sum(float(point[1]) for point in box) / len(box)
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
        if confidence < 0.12:
            continue
        tokens.append((center_y, center_x, text, confidence))

    tolerance = max(12.0, image_height * 0.014)
    rows: list[dict[str, Any]] = []
    for center_y, center_x, text, confidence in sorted(tokens):
        closest = min(rows, key=lambda row: abs(row["center_y"] - center_y), default=None)
        if closest is None or abs(closest["center_y"] - center_y) > tolerance:
            rows.append({"center_y": center_y, "tokens": [(center_x, text, confidence)]})
            continue
        closest["tokens"].append((center_x, text, confidence))
        count = len(closest["tokens"])
        closest["center_y"] += (center_y - closest["center_y"]) / count

    return [sorted(row["tokens"]) for row in sorted(rows, key=lambda row: row["center_y"])]


def _extract_row(
    tokens: list[tuple[float, str, float]],
    league: str | None = None,
) -> dict[str, Any] | None:
    team_codes: list[tuple[float, str, float]] = []
    odds_tokens: list[tuple[float, float, str]] = []
    raw_text_parts: list[str] = []

    for center_x, text, confidence in tokens:
        raw_text_parts.append(text)
        for code in TEAM_CODE_RE.findall(text.upper()):
            if code not in NON_TEAM_CODES and (code in SUPPORTED_TEAM_CODES or len(code) == 4):
                team_codes.append((center_x, code, confidence))
        for match in DECIMAL_ODDS_RE.finditer(text):
            try:
                value = float(match.group(0).replace(",", "."))
            except ValueError:
                continue
            if 1.01 <= value <= 100:
                odds_tokens.append((center_x, value, text))

    team_codes.sort()
    odds_tokens.sort()
    if len(team_codes) < 2 or len(odds_tokens) < 3:
        return None

    home_team, away_team = team_codes[0][1], team_codes[1][1]
    values = [entry[1] for entry in odds_tokens[:3]]
    if home_team == away_team:
        return None
    fixture = {
        "home_team": home_team,
        "away_team": away_team,
        "odds": {"home": values[0], "draw": values[1], "away": values[2]},
        "market_odds": {},
        "ocr_confidence": round(sum(token[2] for token in tokens) / len(tokens), 3),
        "raw_line": " ".join(raw_text_parts),
        "source": "uploaded screenshot OCR",
    }
    if league:
        fixture["league"] = league
    return fixture


def _league_from_text(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text.lower())
    for league, headings in LEAGUE_HEADINGS.items():
        if any(heading in normalized for heading in headings):
            return league
    return None


def extract_matches(image_buffer: Any) -> dict[str, Any]:
    """Return match rows with two team codes and three adjacent 1X2 odds."""
    image = preprocess_image(image_buffer)
    reader = _get_reader()
    try:
        detections = reader.readtext(
            np.asarray(image),
            detail=1,
            paragraph=False,
            batch_size=1,
            decoder="greedy",
        )
    except Exception as exc:
        raise ScreenshotProcessingError("EasyOCR failed to read this screenshot. Try a clearer image.") from exc

    lines = _group_into_rows(detections, image.height)
    fixtures: list[dict[str, Any]] = []
    seen: set[tuple[str, str, float, float, float]] = set()
    active_league: str | None = None
    for line in lines:
        line_text = " ".join(token[1] for token in line)
        detected_league = _league_from_text(line_text)
        if detected_league:
            active_league = detected_league
            continue
        parsed = _extract_row(line, active_league)
        if parsed is None:
            continue
        key = (
            parsed["home_team"],
            parsed["away_team"],
            parsed["odds"]["home"],
            parsed["odds"]["draw"],
            parsed["odds"]["away"],
        )
        if key not in seen:
            seen.add(key)
            fixtures.append(parsed)

    return {
        "fixtures": fixtures,
        "matches": fixtures,
        "fixture_count": len(fixtures),
        "ocr_text": "\n".join(" ".join(token[1] for token in line) for line in lines),
        "parsing_note": (
            "Rows are accepted only when one detected line contains two team codes and at least "
            "three decimal odds. Verify the parsed odds against the screenshot."
        ),
    }