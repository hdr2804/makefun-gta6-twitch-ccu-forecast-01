# GTA VI Twitch CCU forecast — make.fun entry

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hdr2804/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hdr2804/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/c/gta6-twitch-ccu-forecast-01)

Entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/c/gta6-twitch-ccu-forecast-01):
a quantile forecast of Grand Theft Auto VI's release-day Twitch viewership.
Given the first UTC hour of the release day (00:00–01:00 UTC, 19 November
2026), the program prints five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95) of
the category's viewers for each of the remaining 23 hours.

## How it works

`main.py` (Python 3, standard library only) fits its curve at run time from
the comparables shipped inside each case — past game launches with their
full release-day hourly viewers:

- Each comparable's release day becomes a curve of log ratios,
  ln(viewers at hour h) − ln(viewers at hour 1), aligned on the UTC day.
- Comparables are weighted by resemblance to the case: a Gaussian kernel on
  the log of their hour-1 viewers (a launch whose first UTC hour is a
  pre-unlock trickle follows a different curve family from one whose first
  hour already holds a mass audience), the distance between their
  pre-release day's ramp shape and the case's own, a midnight-local rolling
  unlock, console platforms, and a recent era.
- The per-hour centre is a weighted trimmed mean of the donors' log ratios
  and the per-hour spread their weighted dispersion, both smoothed across
  neighbouring hours. The forecast for hour h is the case's own hour-1 print
  × ratio(h), with quantiles at the normal z-values × spread(h).
- Anything unusable in a case falls back to a flat persistence forecast, so
  the program always writes a valid answer file.

Calibration was done by leave-one-out cross-validation over the 37 launch
comparables (each launch in turn treated as a case, fitted on the rest):
mean log pinball loss 0.385 over all 37, and 0.245 over the 27 launches
whose hour-1 print is at least 20,000 viewers — the regime GTA VI's first
hour sits in — with 90.7% of actuals inside the 5%–95% band there.

## Running it

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

The program reads every `*.in` case file in the input directory in name
order and writes one `<case id>.out` per case. Without the environment
variables it uses `/input` and `/output`. It is deterministic and runs in
well under a second per case.

## License

MIT — see [LICENSE](LICENSE).
