![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mochilkity-cmyk/gta6-twitch-ccu-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mochilkity-cmyk/gta6-twitch-ccu-forecast-01/reward.svg)

# GTA VI Twitch CCU Forecast

An entry for [make.fun](https://make.fun) competition
**gta6-twitch-ccu-forecast-01**
([bounty page](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)).

`main.ts` forecasts how many people will be watching Grand Theft Auto VI live
on Twitch in each hour of its release day (19 November 2026, UTC), as five
quantiles an hour (5%, 25%, 50%, 75%, 95%), from the first hour's viewer
count and the past launches shipped in the case file.

## How it forecasts

A launch day's Twitch audience is not a flat line. Past big launches share a
common shape once each hour is divided by that launch's own hour-1 print: a
dip in the small hours of the UTC day, then a climb as the US unlocks and the
US day runs, peaking in the late UTC afternoon and evening. The program
learns that shape from the case's `comparables` at run time, per case, and
blends three views of the same pool, all in ln(viewers_h / viewers_1):

1. **Empirical quantiles** — comparables whose hour-1 print is at least a
   tenth of this case's hour-1 print and whose research note tiers them
   "comparable" (clean midnight-local console launches, no early-access
   contamination); if fewer than 10 qualify, it widens to every comparable
   with an hour-1 print at least a tenth of this case's, then to every
   comparable with a usable hour 1. A tiny hour-1 print makes a launch's
   ratios explode, so conditioning on the print's size is what keeps the
   tails honest. For each hour 2..24, the five quantiles of the log-ratios,
   smoothed across neighbouring hours with [0.25, 0.5, 0.25] and stretched
   1.15x away from the pool's log-median (a small-sample correction: with
   ~10-30 comparables the empirical tails sit too close in).
2. **Regression on print size** — a per-hour regression of the log-ratio on
   ln(hour-1 print), over rolling launches only (comparables whose own
   release day peaks at hour 15 or later — the midnight-local rolling
   launches, the regime this contest's launches are in), conditioned the
   same way on print size. The slope is negative: a launch that already has
   a huge hour-1 print has mostly launched, so it ramps less. Quantiles are
   the fitted value plus the normal quantile times the residual spread
   (floored at 0.08 in log units).
3. **Kernel-weighted mean and spread** — over the same rolling pool, weights
   Gaussian in ln(hour-1 print) with bandwidth 1.0, so launches of a similar
   size count most; mean and spread are smoothed across hours. Quantiles are
   mean plus the normal quantile times spread.
4. **Blend and scale** — the three legs' viewer quantiles are combined by
   geometric mean per hour and level, then q = p1 * exp(log-quantile),
   rounded to 2 decimals, clamped to (0, 1e9] and to at most 50x the hour-1
   print (a launch whose first hour already has live channels in major
   markets cannot plausibly grow fifty-fold by the evening), each row sorted
   ascending.

Everything is deterministic: sorting and arithmetic only, no randomness, no
clock, no network, standard library only. If anything about a case is
unusable, that case falls back to the house persistence baseline, so the
output is always a valid forecast.

## Validation

Leave-one-out on the kit's comparables (each comparable scored as a
pseudo-case against the others, pinball loss on the log scale, the contest's
x2 factor): house baseline 0.930 over all comparables and 0.243 over the
large clean launches (hour-1 print >= 50k, tier comparable); the v1
empirical method scored 0.640 / 0.205; this blend scores 0.524 / 0.189, and
0.263 over the rolling launches with hour-1 print >= 20k (the practice
cases' population), against 0.269 for v1. The 50x cap costs nothing on the
large launches (their ratios never reach it). The baseline scores ~0.387 on
the four practice launches; this method's practice score is only known once
the server grades it.

## Run it

```bash
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Layout

- `main.ts` — the forecaster
- `LICENSE` — MIT
