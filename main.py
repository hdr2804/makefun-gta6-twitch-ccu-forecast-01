"""gta6-twitch-ccu-forecast-01 entry: a quantile forecast of GTA VI's launch-day
Twitch viewers, fitted at run time from the case's own comparables.

Model: for each forecast hour h, the hour's count is the case's hour-1 print
times a ratio, with a log spread around it:

    q_tau(h) = p1 * ratio(h) * exp(z_tau * spread(h))

The ratios and spreads are not guessed: every comparable launch in the case
contributes its release-day curve of hourly viewers divided by its own hour-1
print, aligned on the UTC release day. Comparables are weighted by how much
they resemble the case — above all by the scale of their own hour-1 print
(a launch whose first UTC hour is a pre-unlock trickle follows a different
curve family from one whose first hour already holds a mass audience), then
by the shape of their pre-release day (the ramp of the regions that unlocked
before the UTC day), a midnight-local unlock, console platforms and a recent
era. The per-hour
ratio is a weighted trimmed mean of the comparables' log ratios; the per-hour
spread is their weighted dispersion, smoothed across neighbouring hours and
floored. Both are calibrated on the practice cases' comparables.

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_TABLE = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}

# --- fit constants (calibrated by leave-one-out over the comparables) ---
SCALE_BANDWIDTH = 0.9    # kernel width, in log units, for hour-1 scale matching
CH_BANDWIDTH = 0.0       # kernel width for hour-1 channels matching (0 = off)
MIDNIGHT_BOOST = 1.3     # weight multiplier for a midnight-local rolling unlock
GLOBAL_CUT = 0.45        # weight multiplier for a single simultaneous global unlock
CONSOLE_BOOST = 1.25     # weight multiplier when consoles are among the platforms
ERA_TAU = 9.0            # years; era weight is exp(-|year gap| / ERA_TAU)
WEEKDAY_BOOST = 1.0      # weight multiplier for a release on the case's weekday (1 = off)
CENTRE_MODE = "trimmean"  # how the per-hour centre is taken: median | trimmean | mean
BLEND_K = 0.0            # blend the kernel centre toward the all-donor centre (0 = off)
SHAPE_BANDWIDTH = 0.75   # weight by pre-day ramp-shape distance (0 = off)
SMOOTH_PASSES = 1        # smoothing passes over neighbouring hours
SPREAD_MODE = "std"      # how the per-hour dispersion is taken: std | mad
SPREAD_ASYM = 0          # 1: fit the spread's up and down sides separately
SPREAD_MULT = 1.08       # global multiplier on the fitted dispersion
SPREAD_FLOOR = 0.12      # the spread never goes below this, in log units
CENTRE_SHRINK = 0.0      # shrink the fitted log-ratio centre toward 0 (flat) by this share
RAMP_TILT = 0.0          # tilt the early hours' centre by the case's pre-day ramp (0 = off)

HOUR = timedelta(hours=1)


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def fmt_ts(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """x rounded to 2 decimals, half up (the house starter's rounding)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def z_for(level):
    if level in Z_TABLE:
        return Z_TABLE[level]
    # Any other level: invert the normal CDF by bisection (stdlib only).
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def weighted_median(values, weights):
    """The weighted median, linearly interpolated on the cumulative-weight axis."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    total = sum(weights)
    if total <= 0:
        return values[0]
    half = total / 2.0
    acc = 0.0
    prev_value = values[order[0]]
    prev_acc = 0.0
    for i in order:
        acc += weights[i]
        if acc >= half:
            if acc == half or weights[i] == 0:
                return values[i]
            # interpolate between the previous point and this one
            frac = (half - prev_acc) / weights[i]
            return prev_value + frac * (values[i] - prev_value)
        prev_value = values[i]
        prev_acc = acc
    return values[order[-1]]


def weighted_std(values, weights, centre):
    total = sum(weights)
    if total <= 0:
        return 0.0
    var = sum(w * (v - centre) ** 2 for v, w in zip(values, weights)) / total
    return math.sqrt(var)


def centre_of(values, weights):
    """The per-hour centre of the donors' log ratios, per CENTRE_MODE."""
    if CENTRE_MODE == "mean":
        total = sum(weights)
        return sum(v * w for v, w in zip(values, weights)) / total if total > 0 else values[0]
    if CENTRE_MODE == "trimmean":
        order = sorted(range(len(values)), key=lambda i: values[i])
        total = sum(weights)
        lo_cut, hi_cut = 0.15 * total, 0.85 * total
        acc = 0.0
        num = 0.0
        den = 0.0
        for i in order:
            w = weights[i]
            keep = min(max(min(acc + w, hi_cut) - max(acc, lo_cut), 0.0), w)
            num += keep * values[i]
            den += keep
            acc += w
        return num / den if den > 0 else weighted_median(values, weights)
    return weighted_median(values, weights)


def dispersion_of(values, weights, centre):
    """The per-hour dispersion of the donors' log ratios, per SPREAD_MODE."""
    if SPREAD_MODE == "mad":
        deviations = [abs(v - centre) for v in values]
        return 1.4826 * weighted_median(deviations, weights)
    return weighted_std(values, weights, centre)


def smooth(curve, keys):
    """[0.25, 0.5, 0.25] passes over neighbouring hours, edges renormalised."""
    for _ in range(SMOOTH_PASSES):
        out = {}
        for h in keys:
            num = 0.5 * curve[h]
            den = 0.5
            for step in (-1, 1):
                if h + step in curve:
                    num += 0.25 * curve[h + step]
                    den += 0.25
            out[h] = num / den
        curve = out
    return curve


def comparable_weight(comp, case_p1, case_year, case_weekday, case_ch1):
    """How much one comparable's curve should count for this case."""
    unlock = (comp.get("unlock") or "").lower()
    platforms = (comp.get("platforms") or "").lower()
    weight = 1.0
    # Scale: the strongest signal. A comparable whose own hour-1 print sits at
    # the case's scale follows the case's curve family.
    y1 = comp["_y1"]
    d = math.log(y1) - math.log(case_p1)
    weight *= math.exp(-(d * d) / (2.0 * SCALE_BANDWIDTH * SCALE_BANDWIDTH))
    # Hour-1 channels: a launch's streamer supply at the checkpoint.
    if CH_BANDWIDTH > 0 and case_ch1 and comp.get("_ch1"):
        dc = math.log(comp["_ch1"]) - math.log(case_ch1)
        weight *= math.exp(-(dc * dc) / (2.0 * CH_BANDWIDTH * CH_BANDWIDTH))
    # Unlock shape: midnight-local rolling resembles GTA VI; a single global
    # unlock time makes a different (explosive) curve family.
    if "midnight" in unlock:
        weight *= MIDNIGHT_BOOST
    if "single global" in unlock or "simultaneous global" in unlock or "single simultaneous" in unlock:
        weight *= GLOBAL_CUT
    # Console presence (GTA VI launches on consoles).
    if any(name in platforms for name in ("ps", "xbox", "switch", "playstation")):
        weight *= CONSOLE_BOOST
    # Era: Twitch's scale and habits drift; recent launches transfer better.
    try:
        moment = parse_ts(comp["day_start"])
        weight *= math.exp(-abs(moment.year - case_year) / ERA_TAU)
        if moment.weekday() == case_weekday:
            weight *= WEEKDAY_BOOST
    except Exception:
        pass
    return weight


def launch_curve(comp):
    """A comparable's release-day log-ratio curve {hour: ln(y_h / y_1)}, or None.

    Skips non-launches (the GTA VI category's own pre-release history) and any
    comparable without a usable release day.
    """
    name = comp.get("name") or ""
    unlock = (comp.get("unlock") or "").lower()
    if "before release" in name or "not a launch" in unlock:
        return None
    try:
        start = parse_ts(comp["day_start"])
    except Exception:
        return None
    by_hour = {}
    for point in comp.get("hourly") or []:
        by_hour[point["from"]] = point
    first_point = by_hour.get(fmt_ts(start))
    first = first_point["viewers"] if first_point else None
    if not first or first <= 0:
        return None
    curve = {}
    for h in range(2, 25):
        point = by_hour.get(fmt_ts(start + (h - 1) * HOUR))
        viewers = point["viewers"] if point else None
        if viewers is not None and viewers > 0:
            curve[h] = math.log(viewers) - math.log(first)
    if len(curve) < 18:
        return None
    comp["_y1"] = first
    comp["_ch1"] = first_point.get("channels")
    preday = []
    for k in range(24, 0, -1):
        point = by_hour.get(fmt_ts(start - k * HOUR))
        if point and point["viewers"] and point["viewers"] > 0:
            preday.append((k, math.log(point["viewers"])))
    comp["_preday"] = preday
    return curve


def preday_shape(case):
    """The case's own pre-day path: {hours before the day: log viewers}, demeaned."""
    try:
        start = parse_ts(case["day_start"])
        path = {}
        for point in case.get("history_hourly") or []:
            moment = parse_ts(point["from"])
            gap = (start - moment).total_seconds() / 3600.0
            if 1 <= gap <= 24 and point["viewers"] and point["viewers"] > 0:
                path[int(gap)] = math.log(point["viewers"])
        if len(path) < 12:
            return None
        mean = sum(path.values()) / len(path)
        return {k: v - mean for k, v in path.items()}
    except Exception:
        return None


def shape_distance(donor_preday, case_shape):
    """The RMS distance between a donor's pre-day path and the case's, or None."""
    if not donor_preday or case_shape is None:
        return None
    donor = {k: v for k, v in donor_preday}
    if len(donor) < 12:
        return None
    mean = sum(donor.values()) / len(donor)
    diffs = []
    for k, v in case_shape.items():
        if k in donor:
            diffs.append((donor[k] - mean) - v)
    if len(diffs) < 12:
        return None
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def case_ramp(case):
    """How steeply the case's own audience was still climbing into its day.

    The mean of the last three whole hours before the day over the mean of the
    nine before those, in log units, clipped: positive when the launch was
    still ramping at the checkpoint (for a midnight-local launch, the hours
    before the UTC day hold the first regions' unlocks).
    """
    try:
        start = parse_ts(case["day_start"])
        points = []
        for point in case.get("history_hourly") or []:
            moment = parse_ts(point["from"])
            if moment < start and point["viewers"] and point["viewers"] > 0:
                points.append((moment, point["viewers"]))
        points.sort()
        if len(points) < 6:
            return 0.0
        last = [v for _m, v in points[-3:]]
        earlier = [v for _m, v in points[-12:-3]]
        if not earlier:
            return 0.0
        ramp = math.log(sum(last) / len(last)) - math.log(sum(earlier) / len(earlier))
        return max(-1.0, min(1.0, ramp))
    except Exception:
        return 0.0


def side_std(values, weights, centre, positive):
    """The weighted root-mean-square deviation on one side of the centre."""
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        dev = v - centre
        if (dev > 0) == positive and dev != 0:
            num += w * dev * dev
            den += w
    return math.sqrt(num / den) if den > 0 else 0.0


def fit(case, p1, case_ch1):
    """The per-hour (ratio, spread down, spread up) fitted from the comparables."""
    hours = [row["hour"] for row in case["forecast"]]
    donors = []
    for comp in case.get("comparables") or []:
        curve = launch_curve(comp)
        if curve is not None:
            donors.append((comp, curve))
    if len(donors) < 4:
        return {h: (1.0, 0.15 * math.sqrt(h - 1), 0.15 * math.sqrt(h - 1)) for h in hours}
    start = parse_ts(case["day_start"])
    case_shape = preday_shape(case) if SHAPE_BANDWIDTH > 0 else None
    weights = []
    for comp, _curve in donors:
        w = comparable_weight(comp, p1, start.year, start.weekday(), case_ch1)
        if case_shape is not None:
            dist = shape_distance(comp.get("_preday"), case_shape)
            if dist is not None:
                w *= math.exp(-(dist * dist) / (2.0 * SHAPE_BANDWIDTH * SHAPE_BANDWIDTH))
        weights.append(w)
    centre = {}
    spread_lo = {}
    spread_hi = {}
    for h in hours:
        values = []
        ws = []
        for (comp, curve), w in zip(donors, weights):
            if h in curve:
                values.append(curve[h])
                ws.append(w)
        if len(values) < 3:
            centre[h] = 0.0
            spread_lo[h] = spread_hi[h] = 0.15 * math.sqrt(h - 1)
            continue
        med = centre_of(values, ws)
        if BLEND_K:
            plain = [1.0] * len(values)
            global_centre = centre_of(values, plain)
            ess = (sum(ws) ** 2) / sum(w * w for w in ws)
            med = (ess * med + BLEND_K * global_centre) / (ess + BLEND_K)
        centre[h] = med * (1.0 - CENTRE_SHRINK)
        if SPREAD_ASYM:
            spread_lo[h] = max(side_std(values, ws, med, False) * SPREAD_MULT, SPREAD_FLOOR)
            spread_hi[h] = max(side_std(values, ws, med, True) * SPREAD_MULT, SPREAD_FLOOR)
        else:
            spread_lo[h] = spread_hi[h] = max(
                dispersion_of(values, ws, med) * SPREAD_MULT, SPREAD_FLOOR)
    centre = smooth(centre, hours)
    spread_lo = smooth(spread_lo, hours)
    spread_hi = smooth(spread_hi, hours)
    if RAMP_TILT:
        ramp = case_ramp(case)
        for h in hours:
            centre[h] += RAMP_TILT * ramp * math.exp(-(h - 2) / 6.0)
    return {h: (math.exp(centre[h]), max(spread_lo[h], SPREAD_FLOOR),
                max(spread_hi[h], SPREAD_FLOOR)) for h in hours}


def forecast_case(case):
    levels = case.get("levels") or LEVELS
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is not None:
        p1 = max(1, first["viewers"])
        case_ch1 = first.get("channels")
    else:
        history = case.get("history_hourly") or []
        p1 = max(1, history[-1]["viewers"]) if history else 1
        case_ch1 = history[-1].get("channels") if history else None
    fitted = fit(case, p1, case_ch1)
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        base = 0.15 * math.sqrt(max(h - 1, 0))
        ratio, spread_lo, spread_hi = fitted.get(h, (1.0, base, base))
        quantiles = []
        for level in levels:
            z = z_for(level)
            spread = spread_hi if z > 0 else spread_lo
            q = p1 * ratio * math.exp(z * spread)
            q = min(max(q, 0.01), 1e9)
            quantiles.append(round2(q))
        # Rounding must never break the non-decreasing order.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def baseline_case(case):
    """The persistence baseline's output: always valid, whatever went wrong."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1, first["viewers"]) if first is not None else 1
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        spread = 0.15 * math.sqrt(max(h - 1, 0))
        rows.append({"hour": h, "q": [round2(p1 * math.exp(z_for(lv) * spread)) for lv in LEVELS]})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = baseline_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
