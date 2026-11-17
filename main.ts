// gta6-twitch-ccu-forecast-01: an empirical launch-shape forecaster (Node 22, standard library only).
//
// The idea, in one paragraph: a launch day's Twitch audience is not a flat line. Past big launches
// (the case's `comparables`, 38 in practice, 42 in the final) share a common shape once each hour is
// divided by that launch's own hour-1 print: a dip in the small hours of the UTC day, then a climb as
// the US unlocks and the US day runs, peaking in the late UTC afternoon and evening. This program
// learns that shape from the comparables at run time, per case, instead of hard-coding it. It blends
// three views of the same pool, all in ln(viewers_h / viewers_1):
//   A. Empirical quantiles over tier-preferred pools: comparables whose hour-1 print is at least a
//      tenth of this case's and whose research note tiers them "comparable" (clean midnight-local
//      console launches, no early-access contamination); if fewer than 10 qualify, widen to every
//      comparable with an hour-1 print at least a tenth of this case's, then to every comparable
//      with a usable hour 1. Per-hour quantiles, smoothed across neighbouring hours with
//      [0.25, 0.5, 0.25] and stretched 1.15x away from the pool's log-median (a small-sample
//      correction: with ~10-30 comparables the empirical tails sit too close in).
//   B. A per-hour regression of the log-ratio on ln(hour-1 print) over rolling launches only
//      (comparables whose own release day peaks at hour 15 or later — the midnight-local rolling
//      launches, the regime this contest's launches are in), conditioned the same way on print
//      size. The slope is negative: a launch that already has a huge hour-1 print has mostly
//      launched, so it ramps less. Quantiles are the fitted value plus the normal quantile times
//      the residual spread (floored at 0.08 in log units).
//   C. A kernel-weighted mean and spread over the same rolling pool, weights Gaussian in
//      ln(hour-1 print) with bandwidth 1.0, so launches of a similar size count most; mean and
//      spread are smoothed across hours. Quantiles are mean plus the normal quantile times spread.
// The three legs' viewer quantiles are combined by geometric mean per hour and level, then clamped
// to (0, 1e9] and to at most 50x the hour-1 print (a launch whose first hour already has live
// channels in major markets cannot plausibly grow fifty-fold by the evening), each row sorted
// ascending so the five quantiles never decrease.
// Everything is deterministic: sorting and arithmetic only, no randomness, no clock, no network.
// If anything about a case is unusable, that case falls back to the house persistence baseline
// (every hour at the hour-1 print, spread 0.15*sqrt(h-1)), so the output is always a valid forecast.
//
// Leave-one-out validation on the kit's comparables (each comparable scored as a pseudo-case
// against the others, pinball loss on the log scale, the contest's x2 factor): the v1 empirical
// method scored 0.640 over all comparables and 0.205 over the large clean launches (hour-1 print
// >= 50k, tier comparable); this blend scores 0.524 and 0.189, and 0.263 over the rolling launches
// with hour-1 print >= 20k (the practice cases' population), against 0.269 for v1. The baseline
// scores 0.930 / 0.243 on the same splits, and ~0.387 on the four practice launches; this
// method's practice score is only known once the server grades it.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, used when a case does not name its own. */
const DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each default level. */
const DEFAULT_Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
/** A comparable joins the pool when its hour-1 print is at least p1 / COND_DIV. */
const COND_DIV = 10;
/** Below this many comparables, widen the pool to the next tier. */
const MIN_POOL = 10;
/** Below this many log-ratios for one hour, widen the pool for that hour. */
const MIN_LOGS = 5;
/** Leg A: stretch of the smoothed quantiles away from the pool's log-median. */
const WIDEN = 1.15;
/**
 * Sanity cap: no quantile may exceed RATIO_CAP times the hour-1 print. A launch whose
 * first hour already has live channels in major markets cannot plausibly grow fifty-fold
 * by the evening, and the comparables' extreme ratios all come from launches whose hour-1
 * print was a pre-unlock trickle. Leave-one-out shows the cap costs nothing on large
 * launches (their ratios never reach it) while it stops a contaminated pool from writing
 * a ten-million-viewer 95% quantile.
 */
const RATIO_CAP = 50;
/** Hour-smoothing kernel for each quantile/mean/spread curve. */
const SMOOTH = [0.25, 0.5, 0.25];
/** Fallback (house baseline) spread per square-root hour, in log units. */
const BASE_SIGMA = 0.15;
/** The standard normal's quantile at each default level, for the fallback only. */
const BASE_Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** No quantile may exceed this (the contest's cap). */
const MAX_VIEWERS = 1e9;
/** A comparable counts as a rolling-midnight launch when its release day peaks at this hour or later. */
const ROLLING_PEAK_HOUR = 15;
/** Leg C: Gaussian kernel bandwidth on ln(hour-1 print). */
const KERNEL_BW = 1.0;
/** Floor on any estimated log-scale spread, so a lucky pool never claims certainty. */
const SD_FLOOR = 0.08;
/** Below this many points, the regression leg falls back to the pool's mean and spread. */
const MIN_REG_POINTS = 8;

interface KnownRow { hour: number; viewers: number; }
interface ForecastRow { hour: number; }
interface HourlyPoint { from: string; viewers: number; }
interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  hourly: HourlyPoint[];
}
interface CaseFile {
  id?: string;
  levels?: number[];
  known?: KnownRow[];
  forecast?: ForecastRow[];
  history_hourly?: Array<{ from: string; viewers: number }>;
  comparables?: Comparable[];
}
interface PoolEntry {
  hours: Map<number, number>;
  tier: string;
  h1: number;
  peakHour: number;
  /** ln(viewers_h / viewers_1) for hours 2..24. */
  logRatios: Map<number, number>;
}

/** Median of numbers (average of the two middle values for an even count). */
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

/** Arithmetic mean. */
const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

/** Population standard deviation (divide by n). */
const pstdev = (xs: number[]): number => {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) * (x - m), 0) / xs.length);
};

/** Weighted mean (weights need not sum to 1). */
const weightedMean = (xs: number[], ws: number[]): number => {
  let sw = 0;
  let s = 0;
  for (let i = 0; i < xs.length; i++) { sw += ws[i]; s += ws[i] * xs[i]; }
  return sw > 0 ? s / sw : 0;
};

/** Linear interpolation on sorted values at probability p (the type-7 quantile). */
const empiricalQuantile = (sorted: number[], p: number): number => {
  const n = sorted.length;
  const i = (n - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
};

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The standard normal CDF, Abramowitz & Stegun 26.2.17 (absolute error < 7.5e-8). */
const normalCdf = (x: number): number => {
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.2316419 * ax);
  const poly = (((((1.330274429 * t - 1.821255978) * t + 1.781477937) * t - 0.356563782) * t + 0.31938153) * t);
  const phi = Math.exp((-ax * ax) / 2) / Math.sqrt(2 * Math.PI);
  const cdf = 1 - phi * poly;
  return x >= 0 ? cdf : 1 - cdf;
};

/** The standard normal's quantile at level p: exact table for the contest's five levels, bisection otherwise. */
const zFor = (p: number): number => {
  for (let j = 0; j < DEFAULT_LEVELS.length; j++) {
    if (Math.abs(p - DEFAULT_LEVELS[j]) < 1e-12) return DEFAULT_Z[j];
  }
  let lo = -10;
  let hi = 10;
  for (let i = 0; i < 80; i++) {
    const mid = (lo + hi) / 2;
    if (normalCdf(mid) < p) lo = mid; else hi = mid;
  }
  return (lo + hi) / 2;
};

/** Smooth a per-hour map with the kernel, renormalising at the edges. */
const smoothCurve = (vals: Map<number, number>): Map<number, number> => {
  const out = new Map<number, number>();
  for (const h of vals.keys()) {
    let s = 0;
    let w = 0;
    for (let d = -1; d <= 1; d++) {
      const v = vals.get(h + d);
      if (v !== undefined) { s += v * SMOOTH[d + 1]; w += SMOOTH[d + 1]; }
    }
    if (w > 0) out.set(h, s / w);
  }
  return out;
};

/** One comparable's release-day hourly viewers, hour -> viewers (hours 1..24 of its UTC day). */
const dayHours = (comp: Comparable): Map<number, number> => {
  const t0 = Date.parse(comp.day_start);
  const byHour = new Map<number, number>();
  if (!Number.isFinite(t0)) return byHour;
  for (const p of comp.hourly ?? []) {
    if (typeof p.viewers !== 'number' || !(p.viewers > 0)) continue;
    const h = Math.round((Date.parse(p.from) - t0) / 3600000) + 1;
    if (h >= 1 && h <= 24) byHour.set(h, p.viewers);
  }
  return byHour;
};

/** The research tier of a comparable, parsed from its unlock note ("Tier: comparable", ...). */
const tierOf = (comp: Comparable): string => {
  const m = /Tier:\s*([a-z-]+)/.exec(comp.unlock ?? '');
  return m ? m[1] : 'unknown';
};

/** The comparables as pool entries, with log-ratios and peak hour precomputed. */
const buildEntries = (c: CaseFile): PoolEntry[] => {
  const out: PoolEntry[] = [];
  for (const comp of c.comparables ?? []) {
    const hours = dayHours(comp);
    const h1 = hours.get(1);
    if (h1 === undefined) continue;
    let peakHour = 1;
    let peak = -Infinity;
    for (const [h, v] of hours) {
      if (v > peak) { peak = v; peakHour = h; }
    }
    const logRatios = new Map<number, number>();
    for (const [h, v] of hours) {
      if (h >= 2) logRatios.set(h, Math.log(v / h1));
    }
    out.push({ hours, tier: tierOf(comp), h1, peakHour, logRatios });
  }
  return out;
};

/**
 * Leg A's pools, tightest first: clean tier + conditioned on the hour-1
 * print's size, then every conditioned comparable, then every comparable with a usable hour 1.
 * Only pools with at least MIN_POOL members are kept (plus the widest as a last resort).
 */
const buildPools = (entries: PoolEntry[], p1: number): PoolEntry[][] => {
  const cond = entries.filter((e) => e.h1 >= p1 / COND_DIV);
  const tier = cond.filter((e) => e.tier === 'comparable');
  const ordered = [tier, cond, entries].filter((pool) => pool.length >= MIN_POOL);
  if (ordered.length === 0) ordered.push(entries);
  return ordered;
};

/**
 * Legs B/C pool: rolling-midnight launches conditioned on print size, with fallbacks
 * to every conditioned comparable, then to every comparable.
 */
const rollingPool = (entries: PoolEntry[], p1: number): PoolEntry[] => {
  const rolling = entries.filter((e) => e.peakHour >= ROLLING_PEAK_HOUR && e.h1 >= p1 / COND_DIV);
  if (rolling.length >= MIN_REG_POINTS) return rolling;
  const cond = entries.filter((e) => e.h1 >= p1 / COND_DIV);
  if (cond.length >= MIN_REG_POINTS) return cond;
  return entries;
};

/** A pool's sorted log-ratios ln(viewers_h / viewers_1) for one hour, or undefined when too few. */
const logsForHour = (pool: PoolEntry[], h: number): number[] | undefined => {
  const logs: number[] = [];
  for (const e of pool) {
    const v = e.logRatios.get(h);
    if (v !== undefined) logs.push(v);
  }
  if (logs.length < MIN_LOGS) return undefined;
  return logs.sort((a, b) => a - b);
};

/** Clean one row of viewer quantiles: finite, in range, non-decreasing. */
const cleanRow = (q: number[], p1: number): number[] =>
  q.map((v) => {
    if (!Number.isFinite(v)) return p1;
    return Math.min(MAX_VIEWERS, RATIO_CAP * p1, Math.max(0.01, round2(v)));
  }).sort((a, b) => a - b);

/** The house persistence baseline for one hour: every hour at the hour-1 print, widening spread. */
const baselineRow = (p1: number, h: number, levels: number[]): number[] =>
  levels.map((level, j) => p1 * Math.exp(zFor(level) * BASE_SIGMA * Math.sqrt(h - 1)));

/**
 * Leg A: per-hour empirical quantiles over the tier-preferred pools, smoothed across
 * hours and stretched WIDEN-fold away from the pool's log-median.
 */
const legEmpirical = (
  entries: PoolEntry[], p1: number, levels: number[], wanted: number[],
): Map<number, number[]> => {
  const pools = buildPools(entries, p1);
  const raw = new Map<number, Map<number, number>>();
  const medians = new Map<number, number>();
  for (const level of levels) raw.set(level, new Map<number, number>());
  for (const h of wanted) {
    let logs: number[] | undefined;
    for (const pool of pools) {
      const got = logsForHour(pool, h);
      if (got !== undefined) { logs = got; break; }
    }
    if (logs === undefined) continue;
    medians.set(h, median(logs));
    for (const level of levels) raw.get(level)!.set(h, empiricalQuantile(logs, level));
  }
  const smoothed = new Map<number, Map<number, number>>();
  for (const level of levels) smoothed.set(level, smoothCurve(raw.get(level)!));
  const out = new Map<number, number[]>();
  for (const h of wanted) {
    const med = medians.get(h);
    const q = levels.map((level, j) => {
      const s = smoothed.get(level)!.get(h);
      if (s === undefined || med === undefined) {
        return p1 * Math.exp(BASE_Z[j] * BASE_SIGMA * Math.sqrt(h - 1));
      }
      return p1 * Math.exp(med + (s - med) * WIDEN);
    });
    out.set(h, cleanRow(q, p1));
  }
  return out;
};

/**
 * Leg B: per-hour regression of the log-ratio on ln(hour-1 print) over the rolling pool.
 * Quantiles are the fitted value plus the normal quantile times the residual spread.
 */
const legRegression = (
  entries: PoolEntry[], p1: number, levels: number[], wanted: number[],
): Map<number, number[]> => {
  const pool = rollingPool(entries, p1);
  const x0 = Math.log(p1);
  const out = new Map<number, number[]>();
  for (const h of wanted) {
    const pts: Array<[number, number]> = [];
    for (const e of pool) {
      const lr = e.logRatios.get(h);
      if (lr !== undefined) pts.push([Math.log(e.h1), lr]);
    }
    let qs: number[];
    if (pts.length >= MIN_REG_POINTS) {
      const n = pts.length;
      let mx = 0;
      let my = 0;
      for (const [x, y] of pts) { mx += x; my += y; }
      mx /= n;
      my /= n;
      let sxx = 0;
      let sxy = 0;
      for (const [x, y] of pts) { sxx += (x - mx) * (x - mx); sxy += (x - mx) * (y - my); }
      const b = sxx > 0 ? sxy / sxx : 0;
      const a = my - b * mx;
      const pred = a + b * x0;
      let ssr = 0;
      for (const [x, y] of pts) { const r = y - (a + b * x); ssr += r * r; }
      const sd = Math.max(Math.sqrt(ssr / Math.max(n - 2, 1)), SD_FLOOR);
      qs = levels.map((level) => p1 * Math.exp(pred + zFor(level) * sd));
    } else {
      const xs = pts.map(([, y]) => y);
      if (xs.length < 3) {
        qs = baselineRow(p1, h, levels);
      } else {
        const mu = mean(xs);
        const sd = Math.max(pstdev(xs), SD_FLOOR);
        qs = levels.map((level) => p1 * Math.exp(mu + zFor(level) * sd));
      }
    }
    out.set(h, cleanRow(qs, p1));
  }
  return out;
};

/**
 * Leg C: kernel-weighted mean and spread of the log-ratio over the rolling pool,
 * weights Gaussian in ln(hour-1 print); mean and spread smoothed across hours.
 */
const legKernel = (
  entries: PoolEntry[], p1: number, levels: number[], wanted: number[],
): Map<number, number[]> => {
  const pool = rollingPool(entries, p1);
  const x0 = Math.log(p1);
  const weights = pool.map((e) => Math.exp(-0.5 * ((Math.log(e.h1) - x0) / KERNEL_BW) ** 2));
  const mu = new Map<number, number>();
  const sd = new Map<number, number>();
  for (const h of wanted) {
    const xs: number[] = [];
    const ws: number[] = [];
    for (let i = 0; i < pool.length; i++) {
      const lr = pool[i].logRatios.get(h);
      if (lr !== undefined) { xs.push(lr); ws.push(weights[i]); }
    }
    if (xs.length < 3) continue;
    const m = weightedMean(xs, ws);
    const sw = ws.reduce((a, b) => a + b, 0);
    const sw2 = ws.reduce((a, w) => a + w * w, 0);
    const nEff = sw2 > 0 ? (sw * sw) / sw2 : xs.length;
    const variance = xs.reduce((a, x, i) => a + ws[i] * (x - m) * (x - m), 0) / sw;
    let s: number;
    if (nEff >= 3) {
      s = Math.sqrt((variance * nEff) / (nEff - 1));
    } else {
      // The kernel collapsed onto too few launches: fall back to the plain spread.
      s = Math.max(Math.sqrt(variance), pstdev(xs));
    }
    mu.set(h, m);
    sd.set(h, Math.max(s, SD_FLOOR));
  }
  const muS = smoothCurve(mu);
  const sdS = smoothCurve(sd);
  const out = new Map<number, number[]>();
  for (const h of wanted) {
    const m = muS.get(h);
    const s = sdS.get(h);
    let qs: number[];
    if (m === undefined || s === undefined) {
      qs = baselineRow(p1, h, levels);
    } else {
      qs = levels.map((level) => p1 * Math.exp(m + zFor(level) * s));
    }
    out.set(h, cleanRow(qs, p1));
  }
  return out;
};

/** The forecast of one case: 23 rows, hours 2..24, five quantiles each. */
const forecastCase = (c: CaseFile): { hours: Array<{ hour: number; q: number[] }> } => {
  const levels = Array.isArray(c.levels) && c.levels.length > 0 ? c.levels : DEFAULT_LEVELS;
  const wantedHours = (c.forecast ?? []).map((f) => f.hour).filter((h) => h >= 2 && h <= 24);
  const wanted = wantedHours.length > 0
    ? [...new Set(wantedHours)].sort((a, b) => a - b)
    : Array.from({ length: 23 }, (_, i) => i + 2);
  const known1 = (c.known ?? []).find((k) => k.hour === 1);
  let p1 = 1;
  if (known1 !== undefined && typeof known1.viewers === 'number' && known1.viewers > 0) {
    p1 = known1.viewers;
  } else {
    const tail = (c.history_hourly ?? [])[(c.history_hourly ?? []).length - 1];
    if (tail !== undefined && typeof tail.viewers === 'number' && tail.viewers > 0) p1 = tail.viewers;
  }
  p1 = Math.max(1, p1);
  const entries = buildEntries(c);
  const legA = legEmpirical(entries, p1, levels, wanted);
  const legB = legRegression(entries, p1, levels, wanted);
  const legC = legKernel(entries, p1, levels, wanted);
  const fallback = baselineRow(p1, 2, levels).map(() => p1);
  const rows = wanted.map((h) => {
    const a = legA.get(h) ?? fallback;
    const b = legB.get(h) ?? fallback;
    const d = legC.get(h) ?? fallback;
    const q = levels.map((_, j) => Math.exp((Math.log(a[j]) + Math.log(b[j]) + Math.log(d[j])) / 3));
    return { hour: h, q: cleanRow(q, p1) };
  });
  return { hours: rows };
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let out: { hours: Array<{ hour: number; q: number[] }> };
  try {
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
    out = forecastCase(c);
  } catch {
    // No usable case data: still write a file so the case is not missing outright.
    out = { hours: Array.from({ length: 23 }, (_, i) => ({ hour: i + 2, q: [1, 1, 1, 1, 1] })) };
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours: out.hours })}\n`);
}
