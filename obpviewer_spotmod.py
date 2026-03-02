# obpviewer_mod.py
# SPDX-FileCopyrightText: 2022 Freemelt AB
# SPDX-License-Identifier: Apache-2.0
"""
OBP data viewer modified by William Sjöström.

Adds:
  1) TimedPoints dwell-time color grading (separate colorbar from speed)
  2) Optional spot-size visualization via linewidth scaling (toggle with 'v')
  3) Optional "multi-layer" mode: pass a YAML/JSON build spec that references many .obp/.obp.gz files, NOT QUITE READY YET

Keyboard shortcuts (original + new):
  right / left, shift/ctrl/alt modifiers, p/n, a/e, s/b/r, 0-9 
  v            toggle spot size visualization (linewidth scaling)
"""

from __future__ import annotations

# Built-in
import argparse
import dataclasses
import gzip
import json
import pathlib
import sys
import tkinter
from collections import OrderedDict
from tkinter import ttk
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Freemelt
from obplib import OBP_pb2 as obp

# PyPI
try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "Error: matplotlib is not installed.\n"
        "Try:\n"
        " $ sudo apt install python3-matplotlib\n"
        "or\n"
        " $ python3 -m pip install matplotlib"
    )

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from google.protobuf.internal.decoder import _DecodeVarint32
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.ticker import EngFormatter
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

plt.style.use("dark_background")


# ----------------------------
# OBP parsing + normalization
# ----------------------------

def load_obp_objects(filepath: pathlib.Path) -> Iterator[Any]:
    """Deserialize OBP data and yield protobuf messages."""
    with open(filepath, "rb") as fh:
        data = fh.read()
    if filepath.suffix == ".gz":
        data = gzip.decompress(data)

    consumed = new_pos = 0
    while consumed < len(data):
        msg_len, new_pos = _DecodeVarint32(data, consumed)
        msg_buf = data[new_pos : new_pos + msg_len]
        consumed = new_pos + msg_len

        packet = obp.Packet()
        packet.ParseFromString(msg_buf)
        attr = packet.WhichOneof("payload")
        yield getattr(packet, attr)


class TimedPoint:
    """Lightweight shim so TimedPoints can fit the existing rendering pipeline."""
    __slots__ = ("x", "y", "t", "params")


def _unpack_timedpoints(obp_objects: Iterable[Any]) -> Iterator[Any]:
    """
    Treat each point in TimedPoints as a separate obp object.

    NOTE (same as upstream): This is a compatibility hack.
    """
    for obj in obp_objects:
        if isinstance(obj, obp.TimedPoints):
            t_prev = 0
            for point in obj.points:
                tp = TimedPoint()
                tp.x = point.x
                tp.y = point.y
                # Upstream hack: if point.t is 0, treat as "same as previous"
                if point.t == 0:
                    point.t = t_prev
                tp.t = t_prev = point.t
                tp.params = obj.params
                yield tp
        else:
            yield obj


# ----------------------------
# Build spec helpers (YAML/JSON)
# ----------------------------

def _iter_strings(obj: Any) -> Iterator[str]:
    """Yield all string leaf values from nested dict/list/tuples."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_strings(v)


def _extract_obp_paths_from_spec(spec_obj: Any, base_dir: pathlib.Path) -> List[pathlib.Path]:
    """
    Heuristic extractor:
      - find any string leaf that ends with .obp or .obp.gz
      - resolve relative paths against spec file's directory
      - preserve encounter order
      - de-duplicate preserving order
    """
    found: List[pathlib.Path] = []
    seen: set = set()
    for s in _iter_strings(spec_obj):
        s2 = s.strip().strip('"').strip("'")
        lower = s2.lower()
        if lower.endswith(".obp") or lower.endswith(".obp.gz"):
            p = pathlib.Path(s2)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            if p not in seen:
                seen.add(p)
                found.append(p)
    return found


def load_build_spec_paths(spec_path: pathlib.Path) -> List[pathlib.Path]:
    """
    Load a YAML/JSON build spec and return a list of referenced OBP files.
    This function is intentionally schema-agnostic.
    """
    suffix = spec_path.suffix.lower()
    base_dir = spec_path.parent.resolve()
    raw = spec_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".json":
        spec_obj = json.loads(raw)
    elif suffix in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "YAML build spec requested, but PyYAML is not installed. "
                "Install with: python3 -m pip install pyyaml"
            )
        spec_obj = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported build spec extension: {spec_path.suffix}")

    paths = _extract_obp_paths_from_spec(spec_obj, base_dir=base_dir)
    if not paths:
        raise ValueError(
            "No .obp/.obp.gz references found in build spec. "
            "This loader is schema-agnostic and searches for string leaf values ending in .obp/.obp.gz."
        )
    return paths


# ----------------------------
# Render data model
# ----------------------------

@dataclasses.dataclass
class Data:
    paths: List[Path]
    speeds_mps: np.ndarray
    dwells_s: np.ndarray
    is_timed: np.ndarray
    spotsizes: np.ndarray
    beampowers: np.ndarray
    syncpoints: Dict[int, np.ndarray]
    restores: np.ndarray


def _compute_timedpoint_dwells_s(
    is_timed: np.ndarray,
    tp_times: List[Optional[int]],
    dwell_time_scale: float,
) -> np.ndarray:
    """
    Compute per-object dwell (seconds) for TimedPoints.

    In OBP spot-melting, TimedPoints.point.t is typically the dwell time for that point.
    We therefore map dwell[i] = t[i] * dwell_time_scale for TimedPoints, else 0.

    Set dwell_time_scale=1e-9 if t is in nanoseconds.
    """
    dw = np.zeros(len(tp_times), dtype=float)
    for i, t in enumerate(tp_times):
        if is_timed[i] and t is not None:
            dw[i] = max(0, int(t)) * dwell_time_scale
    return dw

def load_artist_data(obp_objects: Iterable[Any], dwell_time_scale: float = 1e-9) -> Data:
    """Return data used when drawing matplotlib artists."""
    paths: List[Path] = []
    speeds: List[float] = []
    spotsizes: List[int] = []
    beampowers: List[int] = []
    is_timed_list: List[bool] = []
    tp_times: List[Optional[int]] = []

    syncpoints: Dict[int, List[int]] = {}
    _lastseen: Dict[int, int] = {}  # last seen sync points

    restores: List[int] = []
    _restore = 0

    for obj in _unpack_timedpoints(obp_objects):
        # Geometry
        if isinstance(obj, (obp.Line, obp.AcceleratingLine)):
            paths.append(
                Path(
                    np.array([[obj.x0, obj.y0], [obj.x1, obj.y1]]) / 1e6,
                    (Path.MOVETO, Path.LINETO),
                )
            )
            is_timed_list.append(False)
            tp_times.append(None)

        elif isinstance(obj, TimedPoint):
            # draw a tiny diamond (visible when zoomed)
            paths.append(
                Path(
                    np.array(
                        [
                            [obj.x - 100, obj.y],
                            [obj.x, obj.y + 100],
                            [obj.x + 100, obj.y],
                            [obj.x, obj.y - 100],
                            [obj.x - 100, obj.y],
                            [obj.x, obj.y],
                        ]
                    )
                    / 1e6,
                    (
                        Path.MOVETO,
                        Path.LINETO,
                        Path.LINETO,
                        Path.LINETO,
                        Path.LINETO,
                        Path.MOVETO,
                    ),
                )
            )
            is_timed_list.append(True)
            tp_times.append(int(obj.t))

        elif isinstance(obj, (obp.Curve, obp.AcceleratingCurve)):
            paths.append(
                Path(
                    np.array(
                        [
                            [obj.p0.x, obj.p0.y],
                            [obj.p1.x, obj.p1.y],
                            [obj.p2.x, obj.p2.y],
                            [obj.p3.x, obj.p3.y],
                        ]
                    )
                    / 1e6,
                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                )
            )
            is_timed_list.append(False)
            tp_times.append(None)

        elif isinstance(obj, obp.SyncPoint):
            if obj.endpoint not in syncpoints:
                # catch up with zeros when seen for the first time
                syncpoints[obj.endpoint] = [0] * len(paths)
            _lastseen[obj.endpoint] = int(obj.value)
            continue

        elif isinstance(obj, obp.Restore):
            _restore = 1
            continue

        else:
            # Metadata/vendor_setup etc -> ignored
            continue

        # Scalars
        if isinstance(obj, (obp.Line, obp.Curve)):
            speeds.append(obj.speed / 1e6)
        elif isinstance(obj, (obp.AcceleratingLine, obp.AcceleratingCurve)):
            speeds.append(float(obj.sf))
        elif isinstance(obj, TimedPoint):
            speeds.append(0.0)
        else:
            speeds.append(0.0)

        # Params are present on all supported payloads (including TimedPoint shim)
        spotsizes.append(int(obj.params.spot_size))
        beampowers.append(int(obj.params.beam_power))

        for k, v in _lastseen.items():
            syncpoints[k].append(v)

        restores.append(_restore)
        _restore = 0

    if len(paths) == 0:
        raise Exception("no lines/curves/points in obp data")

    # finalize arrays
    for key in list(syncpoints.keys()):
        syncpoints[key] = np.array(syncpoints[key], dtype=int)

    is_timed = np.array(is_timed_list, dtype=bool)
    dwells_s = _compute_timedpoint_dwells_s(is_timed=is_timed, tp_times=tp_times, dwell_time_scale=dwell_time_scale)

    return Data(
        paths=paths,
        speeds_mps=np.array(speeds, dtype=float),
        dwells_s=dwells_s,
        is_timed=is_timed,
        spotsizes=np.array(spotsizes, dtype=int),
        beampowers=np.array(beampowers, dtype=int),
        syncpoints={k: v for k, v in syncpoints.items()},
        restores=np.array(restores, dtype=int),
    )


# ----------------------------
# Viewer UI
# ----------------------------

def _map_linewidths_from_spotsize(spot_sizes: np.ndarray, lw_min: float = 0.2, lw_max: float = 3.0) -> np.ndarray:
    """
    Map spot_size values to Matplotlib linewidths (points).
    Uses min/max within the provided array.
    """
    if len(spot_sizes) == 0:
        return np.array([], dtype=float)
    s = spot_sizes.astype(float)
    smin = float(np.min(s))
    smax = float(np.max(s))
    if smax <= smin:
        return np.full_like(s, (lw_min + lw_max) / 2.0, dtype=float)
    x = (s - smin) / (smax - smin)
    return lw_min + x * (lw_max - lw_min)


class ObpFrame(ttk.Frame):
    def __init__(
        self,
        master: tkinter.Tk,
        data: Data,
        slice_size: int,
        index: Optional[int] = None,
        *,
        show_spot_size: bool = False,
        speed_cmap=plt.cm.rainbow,
        dwell_cmap=plt.cm.plasma,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.data = data
        self._show_spot_size = show_spot_size
        self._speed_cmap = speed_cmap
        self._dwell_cmap = dwell_cmap

        if index is None:
            index = slice_size

        def cap(i: int) -> int:
            return max(0, min(len(self.data.paths) - 1, int(i)))

        self.cap = cap
        index = cap(index)

        # Matplotlib artists and canvas
        fig = Figure(figsize=(9, 8), constrained_layout=True)
        self.fig = fig
        ax = fig.add_subplot(111)
        self.ax = ax

        ax.axhline(0, linewidth=1, zorder=0)
        ax.axvline(0, linewidth=1, zorder=0)

        radius = 0.05  # meters
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        si_meter = EngFormatter(unit="m")
        ax.xaxis.set_major_formatter(si_meter)
        ax.yaxis.set_major_formatter(si_meter)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        # UI state
        self._slice_size = tkinter.IntVar(value=int(slice_size))
        self._index = tkinter.IntVar(value=int(index))

        # Two collections: lines/curves (speed) and timedpoints (dwell)
        self.line_collection = mcoll.PathCollection(
            [],
            facecolors="none",
            transform=ax.transData,
            cmap=self._speed_cmap,
            norm=plt.Normalize(vmin=0, vmax=max(1e-12, float(np.max(self.data.speeds_mps)))),
        )
        # Dwell (spot melting) normalization:
        # Use log scaling to enhance contrast in the common 0.05–5 ms band,
        # while still supporting occasional shorter/longer dwells.
        DWELL_MIN_S = 5e-6   # 0.005 ms (lower clamp; must be >0 for LogNorm)
        DWELL_MAX_S = 5e-3   # 5 ms (upper clamp; values above saturate)

        self.tp_collection = mcoll.PathCollection(
            [],
            facecolors="none",
            transform=ax.transData,
            cmap=self._dwell_cmap,
            norm=LogNorm(vmin=DWELL_MIN_S, vmax=DWELL_MAX_S),
        )
        ax.add_collection(self.line_collection)
        ax.add_collection(self.tp_collection)

        # Colorbars
        self.cbar_speed = fig.colorbar(
            self.line_collection,
            ax=ax,
            pad=0.01,
            aspect=60,
            format=EngFormatter(unit="m/s"),
        )
        self.cbar_speed.ax.tick_params(axis="y", labelsize=8)
        self.cbar_speed.set_label("Speed", fontsize=8)

        self.cbar_dwell = fig.colorbar(
            self.tp_collection,
            ax=ax,
            pad=0.08,
            aspect=60,
            # Show dwell explicitly in milliseconds for readability
            format=mticker.FuncFormatter(lambda x, pos: f"{x*1e3:g}"),
        )
        self.cbar_dwell.ax.tick_params(axis="y", labelsize=8)
        self.cbar_dwell.set_label("Dwell (ms)", fontsize=8)

        # Marker for latest position
        seg = self.data.paths[index]
        self.marker = ax.scatter(*seg.vertices[-1], c="white", marker="*", zorder=3)

        self.canvas = canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.mpl_connect("key_press_event", self.keypress)

        # Slice size widget
        self._slice_size_spinbox = ttk.Spinbox(
            self,
            from_=1,
            to=len(self.data.paths),
            textvariable=self._slice_size,
            command=self.update_index,
            width=8,
        )
        self._slice_size_spinbox.bind("<Return>", self.update_index)

        # Index widgets
        self._index_scale = tkinter.Scale(
            self,
            from_=0,
            to=len(self.data.paths) - 1,
            orient=tkinter.HORIZONTAL,
            variable=self._index,
            command=self.update_index,
        )
        self._index_spinbox = ttk.Spinbox(
            self,
            from_=0,
            to=len(self.data.paths) - 1,
            textvariable=self._index,
            command=self.update_index,
            width=8,
        )
        self._index_spinbox.bind("<Return>", self.update_index)

        self.toolbar_frame = ttk.Frame(master=self)
        toolbar = NavigationToolbar2Tk(canvas, self.toolbar_frame)
        toolbar.update()

        self.info_value = tkinter.StringVar(value=", ".join(self.get_info(index)))
        self.info_label = ttk.Label(self, textvariable=self.info_value)

        self.button_quit = tkinter.Button(self, text="Quit", command=self.master.quit)

        # Initial draw
        self.update_index()

    def setup_grid(self):
        self.canvas.get_tk_widget().grid(row=0, columnspan=4, sticky="NSWE")
        self._index_scale.grid(row=1, columnspan=4, sticky="NSWE")
        self.info_label.grid(row=2, column=0, sticky="SW")
        self._slice_size_spinbox.grid(row=2, column=1, sticky="SE")
        self._index_spinbox.grid(row=2, column=2, sticky="SE")
        self.button_quit.grid(row=2, column=3, sticky="SE")
        self.toolbar_frame.grid(row=3, columnspan=4, sticky="NSWE")

    def keypress(self, event):
        stepsize = 1
        parts = event.key.split("+")
        if len(parts) == 2:
            prefix, key = parts
        else:
            prefix, key = "", parts[0]

        if prefix == "shift" or key in {"P", "N"}:
            stepsize = 10
        elif prefix == "ctrl":
            stepsize = 100
        elif prefix == "alt":
            stepsize = 1000

        if key in {"right", "p", "P"}:
            self._index.set(self.cap(self._index.get() + stepsize))
        elif key in {"left", "n", "N"}:
            self._index.set(self.cap(self._index.get() - stepsize))
        elif event.key == "a":
            self._index.set(0)
        elif event.key == "e":
            self._index.set(len(self.data.paths) - 1)
        elif event.key == "v":
            self._show_spot_size = not self._show_spot_size
        elif event.key.isdigit():
            n = int(event.key)
            for i, sp_key in enumerate(self.data.syncpoints):
                if i + 1 == n:
                    self.nextdifferent(self.data.syncpoints[sp_key])
        elif event.key == "r":
            self.nextdifferent(self.data.restores)
        elif event.key == "b":
            self.nextdifferent(self.data.beampowers)
        elif event.key == "s":
            self.nextdifferent(self.data.spotsizes)
        else:
            # keep upstream behavior: print unknown key
            print(event.key)
            return

        self.update_index()

    def nextdifferent(self, array: np.ndarray):
        start = self.cap(self._index.get())
        if start >= len(array) - 1:
            return
        bools = array[start:] != array[start]
        # if nothing differs, argmax returns 0; keep that consistent with upstream
        self._index.set(start + int(np.argmax(bools)))

    def update_index(self, new_index=None):
        if isinstance(new_index, tkinter.Event):
            # only act on Return; keep focus behavior stable
            if getattr(new_index, "keysym", None) != "Return":
                return
            self.canvas.get_tk_widget().focus_force()
            new_index = self._index.get()

        if new_index is None:
            new_index = self._index.get()

        index = self.cap(int(new_index))
        ss = int(self._slice_size.get() or 1)
        ss = max(1, ss)

        slice_ = slice(self.cap(index + 1 - ss), self.cap(index) + 1)

        # Partition slice into timedpoints vs lines/curves
        idxs = np.arange(len(self.data.paths))[slice_]
        mask_tp = self.data.is_timed[idxs]
        idx_line = idxs[~mask_tp]
        idx_tp = idxs[mask_tp]

        # Update collections
        if len(idx_line):
            paths_line = [self.data.paths[i] for i in idx_line.tolist()]
            self.line_collection.set_paths(paths_line)
            self.line_collection.set_array(self.data.speeds_mps[idx_line])
            _vals = self.data.speeds_mps[idx_line]
            _cols = self.line_collection.cmap(self.line_collection.norm(_vals))
            self.line_collection.set_edgecolors(_cols)
            if self._show_spot_size:
                self.line_collection.set_linewidths(_map_linewidths_from_spotsize(self.data.spotsizes[idx_line]))
            else:
                self.line_collection.set_linewidths(0.6)
        else:
            self.line_collection.set_paths([])
            self.line_collection.set_array(np.array([], dtype=float))
            self.line_collection.set_edgecolors([])

        if len(idx_tp):
            paths_tp = [self.data.paths[i] for i in idx_tp.tolist()]
            self.tp_collection.set_paths(paths_tp)
            self.tp_collection.set_array(self.data.dwells_s[idx_tp])
            # Ensure dwell colormap is visible even with facecolors="none" (hollow markers):
            _vals = self.data.dwells_s[idx_tp]
            _cols = self.tp_collection.cmap(self.tp_collection.norm(_vals))
            self.tp_collection.set_edgecolors(_cols)
            if self._show_spot_size:
                self.tp_collection.set_linewidths(_map_linewidths_from_spotsize(self.data.spotsizes[idx_tp]))
            else:
                self.tp_collection.set_linewidths(0.6)
        else:
            self.tp_collection.set_paths([])
            self.tp_collection.set_array(np.array([], dtype=float))
            self.tp_collection.set_edgecolors([])

        # Marker at "latest" segment end
        seg = self.data.paths[index]
        self.marker.set_offsets(seg.vertices[-1])

        self.canvas.draw()
        self.info_value.set(", ".join(self.get_info(index)))

    def get_info(self, index: int) -> List[str]:
        info = [f"{k}={int(v[index])}" for k, v in self.data.syncpoints.items()]
        info.append(f"Restore={int(self.data.restores[index])}")
        info.append(f"BeamPower={int(self.data.beampowers[index])}")
        info.append(f"SpotSize={int(self.data.spotsizes[index])}")
        if bool(self.data.is_timed[index]):
            info.append(f"Dwell={self.data.dwells_s[index]*1e3:.6g} ms")
        else:
            info.append(f"Speed={self.data.speeds_mps[index]:.6g}m/s")
        info.append(f"SpotViz={'ON' if self._show_spot_size else 'OFF'}")
        return info


class LayeredViewer(ttk.Frame):
    """
    Wraps an ObpFrame and adds a layer slider that lazy-loads OBP files from a build spec.
    """

    def __init__(
        self,
        master: tkinter.Tk,
        obp_paths: Sequence[pathlib.Path],
        slice_size: int,
        index: int,
        *,
        dwell_time_scale: float,
        show_spot_size: bool,
        cache_size: int = 3,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self.obp_paths = list(obp_paths)
        self.slice_size = slice_size
        self.index = index
        self.dwell_time_scale = dwell_time_scale
        self.show_spot_size = show_spot_size

        self._cache: "OrderedDict[int, Data]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))

        # UI state
        self._layer = tkinter.IntVar(value=0)

        # Layer widgets
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="NSWE")

        self.layer_label = ttk.Label(header, text=self._layer_text(0))
        self.layer_label.grid(row=0, column=0, sticky="W")

        self._layer_scale = tkinter.Scale(
            header,
            from_=0,
            to=max(0, len(self.obp_paths) - 1),
            orient=tkinter.HORIZONTAL,
            variable=self._layer,
            command=self._on_layer_change,
        )
        self._layer_scale.grid(row=0, column=1, sticky="NSWE", padx=8)
        header.columnconfigure(1, weight=1)

        self._frame_host = ttk.Frame(self)
        self._frame_host.grid(row=1, column=0, sticky="NSWE")

        # Create first layer frame
        self._obp_frame: Optional[ObpFrame] = None
        self._load_and_swap(layer_idx=0)

    def _layer_text(self, layer_idx: int) -> str:
        p = self.obp_paths[layer_idx]
        return f"Layer {layer_idx + 1}/{len(self.obp_paths)}: {p.name}"

    def _get_data_cached(self, layer_idx: int) -> Data:
        if layer_idx in self._cache:
            self._cache.move_to_end(layer_idx)
            return self._cache[layer_idx]

        path = self.obp_paths[layer_idx]
        obp_objects = load_obp_objects(path)
        data = load_artist_data(obp_objects, dwell_time_scale=self.dwell_time_scale)

        self._cache[layer_idx] = data
        self._cache.move_to_end(layer_idx)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def _load_and_swap(self, layer_idx: int):
        data = self._get_data_cached(layer_idx)

        # destroy previous frame
        if self._obp_frame is not None:
            self._obp_frame.destroy()

        # create new ObpFrame
        self._obp_frame = ObpFrame(
            self._frame_host,
            data,
            self.slice_size,
            self.index,
            show_spot_size=self.show_spot_size,
        )
        self._obp_frame.grid(row=0, column=0, sticky="NSWE", padx=5, pady=5)
        self._obp_frame.setup_grid()

        # update label
        self.layer_label.configure(text=self._layer_text(layer_idx))

    def _on_layer_change(self, _value=None):
        layer_idx = int(self._layer.get())
        layer_idx = max(0, min(len(self.obp_paths) - 1, layer_idx))
        self._load_and_swap(layer_idx=layer_idx)


# ----------------------------
# CLI
# ----------------------------

parser = argparse.ArgumentParser(
    description=(
        "OBP data viewer (modified).\n\n"
        "Positional argument can be:\n"
        "  - .obp or .obp.gz (single file)\n"
        "  - .json / .yml / .yaml (build spec referencing many OBPs; scroll layers)\n\n"
        "Additions:\n"
        "  - TimedPoints colored by dwell-time (separate colorbar)\n"
        "  - Toggle spot size visualization with 'v' (linewidth scaling)\n"
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument("obp_file", type=pathlib.Path, help="Path to .obp/.obp.gz or build spec (.json/.yml/.yaml).")
parser.add_argument("--slice-size", type=int, default=10_000, help="Initial slice size (default: %(default)s).")
parser.add_argument("--index", type=int, default=10_000, help="Initial index (default: %(default)s).")
parser.add_argument(
    "--spot-viz",
    action="store_true",
    help="Start with spot size visualization enabled (linewidth scaling). Toggle during viewing with 'v'.",
)
parser.add_argument(
    "--dwell-time-scale",
    type=float,
    default=1e-9,
    help=(
        "Scale factor applied to TimedPoint time deltas to get seconds (default: %(default)g).\n"
        "If OBP TimedPoints use nanoseconds, keep default (1e-9). If they are already seconds, use 1.0."
    ),
)
parser.add_argument(
    "--layer-cache",
    type=int,
    default=3,
    help="Number of layers to keep cached in memory in build-spec mode (default: %(default)s).",
)


def _main(args: argparse.Namespace) -> None:
    path = args.obp_file
    suffix = path.suffix.lower()

    root = tkinter.Tk()
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.option_add("*tearOff", tkinter.FALSE)

    if suffix in (".json", ".yml", ".yaml"):
        obp_paths = load_build_spec_paths(path)
        root.title(f"obpviewer_mod - {path.name} ({len(obp_paths)} layers)")
        frame = LayeredViewer(
            root,
            obp_paths=obp_paths,
            slice_size=int(args.slice_size),
            index=int(args.index),
            dwell_time_scale=float(args.dwell_time_scale),
            show_spot_size=bool(args.spot_viz),
            cache_size=int(args.layer_cache),
        )
        frame.grid(row=0, column=0, sticky="NSWE")
    else:
        root.title(f"obpviewer_mod - {path.name}")
        obp_objects = load_obp_objects(path)
        data = load_artist_data(obp_objects, dwell_time_scale=float(args.dwell_time_scale))
        frame = ObpFrame(root, data, int(args.slice_size), int(args.index), show_spot_size=bool(args.spot_viz))
        frame.grid(row=0, column=0, sticky="NSWE", padx=5, pady=5)
        frame.setup_grid()

    tkinter.mainloop()


def main() -> None:
    args = parser.parse_args()
    _main(args)


if __name__ == "__main__":
    main()