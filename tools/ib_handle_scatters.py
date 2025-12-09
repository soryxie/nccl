#!/usr/bin/env python3
"""Visualize RDMA requests (per-record) captured by IbTraceRecord dumps.

This script parses the same ib_trace binary dump format and produces a scatter
plot where EACH POINT corresponds to ONE request, positioned by time since the
first record and sized by the request's payload size.

Typical usage:

  python tools/plot_ibtrace_requests.py /tmp/nccl_ibtrace.rank0.bin \
      --phase complete --direction both --units MB --color-by direction \
      --alpha 0.5 --size 12 --output requests.png
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional

# --- Binary format (same as your existing tool) ---
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

# --- IO ---

def _read_header(file_obj) -> TraceFileHeader:
  header_data = file_obj.read(HEADER_SIZE)
  if len(header_data) != HEADER_SIZE:
    raise ValueError("File too small to contain trace header.")
  return TraceFileHeader(*struct.unpack(HEADER_FMT, header_data))

def _read_records(file_obj, header: TraceFileHeader, limit: Optional[int]) -> List[IbTraceRecord]:
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

def load_trace(path: Path, limit: Optional[int]) -> Tuple[TraceFileHeader, List[IbTraceRecord]]:
  with path.open("rb") as fh:
    header = _read_header(fh)
    if not header.magic.startswith(HEADER_MAGIC):
      raise ValueError(f"Magic mismatch: expected prefix {HEADER_MAGIC!r}, found {header.magic!r}")
    records = _read_records(fh, header, limit) if header.count else []
  records.sort(key=lambda r: r.t_ns)
  return header, records

# --- Filtering (same semantics as你的现有脚本) ---

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

  out: List[IbTraceRecord] = []
  for rec in records:
    if phase_target is not None and rec.phase != phase_target:
      continue
    if dir_target is not None and rec.is_send != dir_target:
      continue
    if rec.size < min_size:
      continue
    out.append(rec)
  return out

# --- Plot helpers ---

def size_to_units(values: List[int], units: str) -> List[float]:
  div = 1.0
  if units == "KB":
    div = 1024.0
  elif units == "MB":
    div = 1024.0 * 1024.0
  elif units == "GB":
    div = 1024.0 * 1024.0 * 1024.0
  return [v / div for v in values]

def human_readable_bytes(num_bytes: int) -> str:
  units = ["B", "KB", "MB", "GB", "TB"]
  value = float(num_bytes)
  for unit in units:
    if value < 1024.0 or unit == units[-1]:
      return f"{value:.2f} {unit}"
    value /= 1024.0
  return f"{value:.2f} TB"

def build_series(records: Sequence[IbTraceRecord],
                 color_by: str,
                 units: str) -> Dict[str, Tuple[List[float], List[float]]]:
  """Return mapping label -> (xs_ms, ys_units). Each label is a group per color_by."""
  if not records:
    return {}
  start_ns = records[0].t_ns

  def key(rec: IbTraceRecord) -> str:
    if color_by == "none":
      return "requests"
    if color_by == "direction":
      return "send" if rec.is_send else "recv"
    if color_by == "phase":
      return "complete" if rec.phase == 1 else "post"
    if color_by == "opcode":
      return f"opcode={rec.opcode}"
    if color_by == "qp":
      return f"qp={rec.qp}"
    if color_by == "dev":
      return f"dev={rec.dev}"
    return "requests"

  groups: Dict[str, List[IbTraceRecord]] = {}
  for r in records:
    groups.setdefault(key(r), []).append(r)

  series: Dict[str, Tuple[List[float], List[float]]] = {}
  for label, rs in groups.items():
    xs_ms = [(r.t_ns - start_ns) / 1e6 for r in rs]
    ys = size_to_units([r.size for r in rs], units)
    series[label] = (xs_ms, ys)
  return series

def plot_scatter(trace_path: Path,
                 series: Dict[str, Tuple[List[float], List[float]]],
                 units: str,
                 alpha: float,
                 marker_size: float,
                 phase: str,
                 direction: str,
                 record_count: int,
                 total_bytes: int,
                 output: Optional[Path],
                 logy: bool) -> None:
  if not series:
    print("No records to plot with the given filters.")
    return

  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise SystemExit("matplotlib is required for plotting. Install it via `pip install matplotlib`.") from exc

  fig, ax = plt.subplots(figsize=(12, 5))
  # 每个分组画一层散点
  for label, (xs, ys) in series.items():
    ax.scatter(xs, ys, s=marker_size, alpha=alpha, linewidths=0, label=label)

  ax.set_xlabel("Time since first record (ms)")
  ax.set_ylabel(f"Request size ({units})")
  ax.set_title(
      f"{trace_path.name} | {record_count} records | total {human_readable_bytes(total_bytes)} "
      f"| phase={phase} | direction={direction}"
  )
  if logy:
    ax.set_yscale("log")
  ax.legend()
  ax.grid(alpha=0.25)
  fig.tight_layout()

  if output:
    fig.savefig(output, dpi=150)
    print(f"Saved scatter plot to {output}")
  else:
    plt.show()

# --- CLI ---

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
  p = argparse.ArgumentParser(
      description="Plot per-request size over time from IbTraceRecord dumps."
  )
  p.add_argument("trace", type=Path, help="Path to an ib_trace binary dump.")
  p.add_argument("--phase",
                 choices=["post", "complete", "all"],
                 default="complete",
                 help="Filter by phase (default: complete).")
  p.add_argument("--direction",
                 choices=["send", "recv", "both"],
                 default="both",
                 help="Filter by RDMA direction (default: both).")
  p.add_argument("--min-size",
                 type=int,
                 default=0,
                 help="Filter out records smaller than this many bytes.")
  p.add_argument("--limit",
                 type=int,
                 default=None,
                 help="Stop after parsing this many records.")
  p.add_argument("--units",
                 choices=["B", "KB", "MB", "GB"],
                 default="MB",
                 help="Y-axis units for request size (default: MB).")
  p.add_argument("--color-by",
                 choices=["none", "direction", "phase", "opcode", "qp", "dev"],
                 default="direction",
                 help="Color/group points by this field (default: direction).")
  p.add_argument("--alpha",
                 type=float,
                 default=0.5,
                 help="Point transparency (default: 0.5).")
  p.add_argument("--size",
                 type=float,
                 default=1.0,
                 help="Marker size for scatter points (default: 10).")
  p.add_argument("--logy",
                 action="store_true",
                 help="Use log scale for Y axis.")
  p.add_argument("--output",
                 type=Path,
                 default=None,
                 help="If set, save the figure to this path instead of showing it.")
  return p.parse_args(argv)

def main(argv: Sequence[str]) -> None:
  args = parse_args(argv)

  header, records = load_trace(args.trace, args.limit)
  print(f"Loaded header: version={header.version}, records={header.count}, record_size={header.record_size}")
  if not records:
    print("Trace file does not contain any records.")
    return

  filtered = filter_records(records, args.phase, args.direction, args.min_size)
  if not filtered:
    print("No records left after filtering.")
    return

  total_bytes = sum(r.size for r in filtered)
  time_range_ns = filtered[-1].t_ns - filtered[0].t_ns
  print(
      f"Filtered records: {len(filtered)} ({len(filtered)/len(records)*100:.1f}% of parsed) "
      f"covering {time_range_ns/1e6:.3f} ms, total {human_readable_bytes(total_bytes)}"
  )

  series = build_series(filtered, args.color_by, args.units)
  plot_scatter(args.trace, series, args.units, args.alpha, args.size,
               args.phase, args.direction, len(filtered), total_bytes,
               args.output, args.logy)

if __name__ == "__main__":
  main(sys.argv[1:])
