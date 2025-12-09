#!/usr/bin/env python3
"""Visualize RDMA traffic captured by IbTraceRecord dumps.

The script understands the binary format written by ib_trace_dump_from_env and
creates a time-series plot that shows how much data traversed RDMA per time
slice.  Typical usage:

    python tools/plot_ibtrace.py /tmp/nccl_ibtrace.rank0.bin --bin-us 200 \
        --direction both --split-direction --scatter --output trace.png
"""

from __future__ import annotations

import argparse
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

HEADER_FMT = "<8sIIQ"  # magic[8], version(u32), record_size(u32), count(u64)
RECORD_FMT = "<QQIHHBBBBI"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_SIZE = struct.calcsize(RECORD_FMT)
HEADER_MAGIC = b"IBTRACE"


@dataclass
class TraceFileHeader:
  magic: bytes
  version: int
  record_size: int
  count: int


@dataclass
class IbTraceRecord:
  t_ns: int
  wr_id: int
  size: int
  dev: int
  qp: int
  opcode: int
  is_send: int
  phase: int
  status: int
  extra: int


def _read_header(file_obj) -> TraceFileHeader:
  header_data = file_obj.read(HEADER_SIZE)
  if len(header_data) != HEADER_SIZE:
    raise ValueError("File too small to contain trace header.")
  return TraceFileHeader(*struct.unpack(HEADER_FMT, header_data))


def _read_records(file_obj, header: TraceFileHeader, limit: int | None) -> List[IbTraceRecord]:
  if header.record_size != RECORD_SIZE:
    raise ValueError(
        f"Record size mismatch: file={header.record_size}, expected={RECORD_SIZE}. "
        "Please update RECORD_FMT to match IbTraceRecord.")

  records: List[IbTraceRecord] = []
  for _ in range(header.count):
    data = file_obj.read(RECORD_SIZE)
    if len(data) < RECORD_SIZE:
      break
    records.append(IbTraceRecord(*struct.unpack(RECORD_FMT, data)))
    if limit and len(records) >= limit:
      break
  return records


def load_trace(path: Path, limit: int | None) -> Tuple[TraceFileHeader, List[IbTraceRecord]]:
  with path.open("rb") as fh:
    header = _read_header(fh)
    if not header.magic.startswith(HEADER_MAGIC):
      raise ValueError(f"Magic mismatch: expected prefix {HEADER_MAGIC!r}, found {header.magic!r}")
    if header.count == 0:
      return header, []
    records = _read_records(fh, header, limit)
  records.sort(key=lambda r: r.t_ns)
  return header, records


def filter_records(records: Sequence[IbTraceRecord],
                   phase: str,
                   direction: str,
                   min_size: int) -> List[IbTraceRecord]:
  phase_map = {"post": 0, "complete": 1}
  phase_target = None if phase == "all" else phase_map[phase]
  dir_target = None
  if direction == "send":
    dir_target = 1
  elif direction == "recv":
    dir_target = 0

  filtered: List[IbTraceRecord] = []
  for rec in records:
    if phase_target is not None and rec.phase != phase_target:
      continue
    if dir_target is not None and rec.is_send != dir_target:
      continue
    if rec.size < min_size:
      continue
    filtered.append(rec)
  return filtered


def _aggregate_bins(records: Sequence[IbTraceRecord],
                    bin_ns: float,
                    split_direction: bool) -> Dict[str, Tuple[List[float], List[float]]]:
  if not records:
    return {}

  start_ns = records[0].t_ns
  bin_ms = bin_ns / 1e6

  def finalize(bin_dict: Dict[int, int]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for idx in sorted(bin_dict):
      xs.append(idx * bin_ms)
      ys.append(bin_dict[idx] / (1024 * 1024))  # MB per bin
    return xs, ys

  if split_direction:
    send_bins: Dict[int, int] = defaultdict(int)
    recv_bins: Dict[int, int] = defaultdict(int)
    for rec in records:
      idx = int((rec.t_ns - start_ns) // bin_ns)
      target = send_bins if rec.is_send else recv_bins
      target[idx] += rec.size
    series: Dict[str, Tuple[List[float], List[float]]] = {}
    if send_bins:
      series["send"] = finalize(send_bins)
    if recv_bins:
      series["recv"] = finalize(recv_bins)
    return series

  bins: Dict[int, int] = defaultdict(int)
  for rec in records:
    idx = int((rec.t_ns - start_ns) // bin_ns)
    bins[idx] += rec.size
  return {"traffic": finalize(bins)}


def _scatter_points(records: Sequence[IbTraceRecord]) -> Tuple[List[float], List[float]]:
  if not records:
    return [], []
  start_ns = records[0].t_ns
  xs = [(rec.t_ns - start_ns) / 1e6 for rec in records]
  ys = [rec.size / (1024 * 1024) for rec in records]
  return xs, ys


def plot_time_series(path: Path,
                     series: Dict[str, Tuple[List[float], List[float]]],
                     scatter: Tuple[List[float], List[float]] | None,
                     bin_us: float,
                     phase: str,
                     direction: str,
                     total_bytes: int,
                     record_count: int,
                     output: Path | None) -> None:
  if not series:
    print("No records to plot with the given filters.")
    return

  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:  # pragma: no cover - dependency error surfaced at runtime
    raise SystemExit("matplotlib is required for plotting. Install it via `pip install matplotlib`.") from exc

  fig, ax = plt.subplots(figsize=(12, 5))
  for label, (xs, ys) in series.items():
    ax.plot(xs, ys, label=f"{label} (bin={bin_us:.0f} us)", drawstyle="steps-mid")

  if scatter and scatter[0]:
    ax.scatter(scatter[0],
               scatter[1],
               s=10,
               c="tab:gray",
               alpha=0.4,
               label="records",
               linewidths=0)

  ax.set_xlabel("Time since first record (ms)")
  ax.set_ylabel("Data per bin (MB)")
  ax.set_title(
      f"{path.name} | {record_count} records | {total_bytes / (1024 * 1024):.2f} MB "
      f"| phase={phase} | direction={direction}")
  ax.legend()
  ax.grid(alpha=0.2)
  fig.tight_layout()

  if output:
    fig.savefig(output, dpi=150)
    print(f"Saved plot to {output}")
  else:
    plt.show()


def human_readable_bytes(num_bytes: int) -> str:
  units = ["B", "KB", "MB", "GB", "TB"]
  value = float(num_bytes)
  for unit in units:
    if value < 1024.0 or unit == units[-1]:
      return f"{value:.2f} {unit}"
    value /= 1024.0
  return f"{value:.2f} TB"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Parse IbTraceRecord dumps and plot RDMA traffic over time.")
  parser.add_argument("trace", type=Path, help="Path to an ib_trace binary dump.")
  parser.add_argument("--phase",
                      choices=["post", "complete", "all"],
                      default="complete",
                      help="Choose which phase to visualize (default: complete).")
  parser.add_argument("--direction",
                      choices=["send", "recv", "both"],
                      default="both",
                      help="Filter by RDMA direction (default: both).")
  parser.add_argument("--bin-us",
                      type=float,
                      default=1_000_000.0,
                      help="Bin width in microseconds for the aggregated time-series "
                           "(default: 1 second).")
  parser.add_argument("--min-size",
                      type=int,
                      default=0,
                      help="Filter out records smaller than this many bytes.")
  parser.add_argument("--limit",
                      type=int,
                      default=None,
                      help="Stop after parsing this many records (useful for quick tests).")
  parser.add_argument("--split-direction",
                      action="store_true",
                      help="Plot send/recv traffic separately (requires --direction both).")
  parser.add_argument("--scatter",
                      action="store_true",
                      help="Overlay per-record scatter points on top of the aggregated plot.")
  parser.add_argument("--output",
                      type=Path,
                      default=None,
                      help="Optional path to save the figure instead of showing it interactively.")
  return parser.parse_args(argv)


def main(argv: Sequence[str]) -> None:
  args = parse_args(argv)
  if args.split_direction and args.direction != "both":
    raise SystemExit("--split-direction only makes sense together with --direction both.")

  header, records = load_trace(args.trace, args.limit)
  print(f"Loaded header: version={header.version}, records={header.count}, record_size={header.record_size}")
  if not records:
    print("Trace file does not contain any records.")
    return

  filtered = filter_records(records, args.phase, args.direction, args.min_size)
  if not filtered:
    print("No records left after filtering.")
    return

  total_bytes = sum(rec.size for rec in filtered)
  time_range_ns = filtered[-1].t_ns - filtered[0].t_ns
  print(
      f"Filtered records: {len(filtered)} ("
      f"{len(filtered)/len(records)*100:.1f}% of parsed) covering "
      f"{time_range_ns / 1e6:.3f} ms, total {human_readable_bytes(total_bytes)}")

  bin_ns = args.bin_us * 1000.0
  series = _aggregate_bins(filtered, bin_ns, args.split_direction)
  scatter = _scatter_points(filtered) if args.scatter else None
  plot_time_series(args.trace, series, scatter, args.bin_us, args.phase, args.direction,
                   total_bytes, len(filtered), args.output)


if __name__ == "__main__":
  main(sys.argv[1:])
