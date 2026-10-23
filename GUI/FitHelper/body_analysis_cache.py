"""Small per-account cache of the latest body-analysis record ID."""
import json
from pathlib import Path

_CACHE_FILE = Path(__file__).resolve().with_name(".body_analysis_cache.json")


def account_key(login_window=None):
    try:
        import login_client
        user = getattr(login_client, "_user", None)
    except ImportError:
        user = None
    if isinstance(user, dict):
        for key in ("user_id", "id", "username", "userID"):
            value = user.get(key)
            if value is not None and str(value).strip():
                return f"{key}:{str(value).strip().lower()}"
    if login_window is not None:
        for name in ("usernameEdit", "login_username"):
            field = getattr(login_window, name, None)
            if field is not None and hasattr(field, "text"):
                value = field.text().strip()
                if value:
                    return f"username:{value.lower()}"
    return None


def load_body_id(login_window=None):
    key = account_key(login_window)
    if not key:
        return None
    try:
        records = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        record = records.get(key, {})
        return record.get("body_id") if isinstance(record, dict) else None
    except (OSError, ValueError, AttributeError):
        return None


def save_body_id(login_window, body_id):
    key = account_key(login_window)
    if not key or not body_id:
        return
    try:
        records = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        records = {}
    if not isinstance(records, dict):
        records = {}
    records[key] = {"body_id": str(body_id)}
    _CACHE_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clear_body_id(login_window=None, body_id=None):
    key = account_key(login_window)
    if not key:
        return
    try:
        records = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    record = records.get(key)
    if not isinstance(record, dict):
        return
    if body_id is not None and str(record.get("body_id")) != str(body_id):
        return
    records.pop(key, None)
    _CACHE_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_AREA_LABELS = {
    "chest_status": "가슴",
    "shoulder_status": "어깨",
    "arm_status": "팔",
    "back_status": "등",
    "leg_status": "하체",
}
_ALIASES = {
    "chest": "가슴", "가슴": "가슴",
    "shoulder": "어깨", "shoulders": "어깨", "어깨": "어깨",
    "arm": "팔", "arms": "팔", "팔": "팔",
    "back": "등", "등": "등",
    "leg": "하체", "legs": "하체", "lower body": "하체", "하체": "하체",
}


def _classify(value):
    if isinstance(value, dict):
        value = value.get("status", value.get("rating", value.get("condition")))
    text = str(value).strip().lower() if value is not None else ""
    if text in {"good", "strength", "strong", "양호", "좋음", "정상", "강점"}:
        return "good"
    if text in {"bad", "weakness", "weak", "poor", "불량", "나쁨", "취약", "약점"}:
        return "bad"
    return "unknown"


def _add_regions(value, destination):
    if isinstance(value, dict):
        entries = value.keys()
    elif isinstance(value, (list, tuple, set)):
        entries = value
    elif isinstance(value, str):
        entries = value.replace(";", ",").replace("\n", ",").split(",")
    else:
        entries = ()
    for entry in entries:
        name = str(entry).strip().lower()
        for alias, label in _ALIASES.items():
            if name == alias or alias in name:
                if label not in destination:
                    destination.append(label)
                break


def area_groups(response):
    data = dict(response or {})
    nested = data.get("result")
    if isinstance(nested, dict):
        data.update(nested)
    good, bad, unknown = [], [], []
    for key, label in _AREA_LABELS.items():
        status = _classify(data.get(key))
        target = good if status == "good" else bad if status == "bad" else unknown
        if label not in target:
            target.append(label)
    _add_regions(data.get("good"), good)
    _add_regions(data.get("bad"), bad)
    for label in good + bad:
        if label in unknown:
            unknown.remove(label)
    return good, bad, list(dict.fromkeys(unknown))


def format_area_report(response):
    good, bad, unknown = area_groups(response)
    def section(title, values):
        items = "\n".join(f"• {name}" for name in values) if values else "• 없음"
        return f"{title}\n{items}"
    parts = [section("Strength:", good), section("Weakness:", bad)]
    if unknown:
        parts.append(section("판정 불가:", unknown))
    return "\n\n".join(parts)


def format_area_summary(response):
    good, bad, unknown = area_groups(response)
    parts = [
        "Strength: " + (", ".join(good) if good else "없음"),
        "Weakness: " + (", ".join(bad) if bad else "없음"),
    ]
    if unknown:
        parts.append("판정 불가: " + ", ".join(unknown))
    return "\n".join(parts)
