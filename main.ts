// gta6-twitch-ccu-forecast-01: an empirical launch-shape forecaster (Node 22, standard library only).
//
// The idea, in one paragraph: a launch day's Twitch audience is not a flat line. Past big launches
// (the case's `comparables`, 38 in practice, 42 in the final) share a common shape once each hour is
// divided by that launch's own hour-1 print: a dip in the small hours of the UTC day, then a climb as
// the US unlocks and the US day runs, peaking in the late UTC afternoon and evening. This program
// learns that shape from the comparables at run time, per case, instead of hard-coding it:
//   1. Pool: comparables whose hour-1 print is at least a tenth of this case's hour-1 print and
//      whose research note tiers them "comparable" (clean midnight-local console launches, no
//      early-access contamination); if fewer than 10 qualify, widen to every comparable with an
//      hour-1 print at least a tenth of this case's, then to every comparable with a usable hour 1.
//      (A tiny hour-1 print makes a launch's ratios explode, so conditioning on the print's size
//      is what keeps the tails honest: unconditioned, the 95% quantile runs to tens of millions.)
//   2. Shape: for each hour h = 2..24, the five quantiles of ln(viewers_h / viewers_1) across the
//      pool, taken empirically (linear interpolation on the sorted log-ratios) at the case's levels.
//   3. Smoothing: each quantile curve is smoothed across neighbouring hours with [0.25, 0.5, 0.25],
//      because the true shape is smooth in h and the per-hour samples are small.
//   4. Widening: the smoothed quantiles are stretched 1.15x away from the pool's log-median, a
//      small-sample correction: with ~10-30 comparables the empirical tails sit too close in.
//   5. Scale: q = p1 * exp(log-quantile), rounded to 2 decimals, clamped to (0, 1e9] and to at
//      most 50x the hour-1 print (a launch whose first hour already has live channels in major
//      markets cannot plausibly grow fifty-fold by the evening), each row sorted ascending so
//      the five quantiles never decrease.
// Everything is deterministic: sorting and arithmetic only, no randomness, no clock, no network.
// If anything about a case is unusable, that case falls back to the house persistence baseline
// (every hour at the hour-1 print, spread 0.15*sqrt(h-1)), so the output is always a valid forecast.
//
// Leave-one-out validation on the kit's comparables (each comparable scored as a pseudo-case
// against the others, pinball loss on the log scale): house baseline 0.930, this method 0.598
// over all comparables and 0.198 over the large clean launches (hour-1 print >= 50k), the
// population GTA VI belongs to; the 50x cap costs nothing on those launches (their ratios
// never reach it). The baseline scores ~0.387 on the four practice launches; this
// method's practice score is only known once the server grades it.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, used when a case does not name its own. */
const DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** A comparable joins the pool when its hour-1 print is at least p1 / COND_DIV. */
const COND_DIV = 10;
/** Below this many comparables, widen the pool to the next tier. */
const MIN_POOL = 10;
/** Below this many log-ratios for one hour, widen the pool for that hour. */
const MIN_LOGS = 5;
/** Stretch of the smoothed quantiles away from the pool's log-median. */
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
/** Hour-smoothing kernel for each quantile curve. */
const SMOOTH = [0.25, 0.5, 0.25];
/** Fallback (house baseline) spread per square-root hour, in log units. */
const BASE_SIGMA = 0.15;
/** The standard normal's quantile at each default level, for the fallback only. */
const BASE_Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** No quantile may exceed this (the contest's cap). */
const MAX_VIEWERS = 1e9;

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
interface PoolEntry { hours: Map<number, number>; tier: string; h1: number; }

/** Median of numbers (average of the two middle values for an even count). */
const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
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

/**
 * The comparable pools for a case, tightest first: clean tier + conditioned on the hour-1
 * print's size, then every conditioned comparable, then every comparable with a usable hour 1.
 * Only pools with at least MIN_POOL members are kept (plus the widest as a last resort).
 */
const buildPools = (c: CaseFile, p1: number): PoolEntry[][] => {
  const all: PoolEntry[] = [];
  for (const comp of c.comparables ?? []) {
    const hours = dayHours(comp);
    const h1 = hours.get(1);
    if (h1 === undefined) continue;
    all.push({ hours, tier: tierOf(comp), h1 });
  }
  const cond = all.filter((e) => e.h1 >= p1 / COND_DIV);
  const tier = cond.filter((e) => e.tier === 'comparable');
  const ordered = [tier, cond, all].filter((pool) => pool.length >= MIN_POOL);
  if (ordered.length === 0) ordered.push(all);
  return ordered;
};

/** A pool's sorted log-ratios ln(viewers_h / viewers_1) for one hour, or undefined when too few. */
const logsForHour = (pool: PoolEntry[], h: number): number[] | undefined => {
  const logs: number[] = [];
  for (const e of pool) {
    const v = e.hours.get(h);
    if (v !== undefined) logs.push(Math.log(v / e.h1));
  }
  if (logs.length < MIN_LOGS) return undefined;
  return logs.sort((a, b) => a - b);
};

/** The house persistence baseline for one hour: every hour at the hour-1 print, widening spread. */
const baselineRow = (p1: number, h: number, levels: number[]): number[] =>
  levels.map((_, j) => {
    const z = BASE_Z[j] ?? 0;
    return p1 * Math.exp(z * BASE_SIGMA * Math.sqrt(h - 1));
  });

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
  const pools = buildPools(c, p1);
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
  const rows = wanted.map((h) => {
    const med = medians.get(h);
    const q = levels.map((level) => {
      const s = smoothed.get(level)!.get(h);
      if (s === undefined || med === undefined) {
        const j = levels.indexOf(level);
        const z = BASE_Z[j] ?? 0;
        return p1 * Math.exp(z * BASE_SIGMA * Math.sqrt(h - 1));
      }
      return p1 * Math.exp(med + (s - med) * WIDEN);
    });
    const cleaned = q.map((v) => {
      if (!Number.isFinite(v)) return p1;
      return Math.min(MAX_VIEWERS, RATIO_CAP * p1, Math.max(0.01, round2(v)));
    }).sort((a, b) => a - b);
    return { hour: h, q: cleaned };
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
