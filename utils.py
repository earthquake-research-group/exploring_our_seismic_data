from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Dict, List, Tuple, Optional
import pandas as pd
from typing import Optional
import re
import matplotlib.pyplot as plt
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.core.inventory import Inventory
from obspy.core.event import Catalog, Event, Origin, OriginQuality, Pick, Comment
from obspy.core.event.base import WaveformStreamID
from obspy.core.event.magnitude import Magnitude




@dataclass(frozen=True)
class Provider:
    """A single FDSN endpoint."""
    name: str
    base_url: str


@dataclass
class ProviderSelection:
    """What we ended up using from a provider."""
    provider: Provider
    networks: List[str]
    stations: List[str]
    inventory: Inventory


def _as_list_csv(x: Iterable[str]) -> str:
    return ",".join(sorted(set(s.strip() for s in x if s and s.strip())))


def collect_stations_and_waveforms(
    *,
    providers: Sequence[Provider],
    requested_networks: Sequence[str],
    latitude: float,
    longitude: float,
    maxradius: float,
    starttime: UTCDateTime,
    endtime: UTCDateTime,
    minradius: float = 0.0,
    level: str = "channel",
    location: str = "*",
    channel: str = "*",
    attach_response: bool = False,
    timeout: int = 60,
    merge: bool = True,
) -> Tuple[Stream, Dict[str, ProviderSelection]]:
    """
    Hierarchically resolve requested networks across multiple FDSN servers,
    then fetch waveforms per provider.

    Returns
    -------
    st : obspy.Stream
        Merged stream of all downloaded waveforms.
    selections : dict
        Keyed by provider.name, values include networks/stations/inventory used.
    """
    requested = [n.strip() for n in requested_networks if n and n.strip()]
    unresolved = set(requested)

    # Provider -> accumulated Inventory (stations found)
    selections: Dict[str, ProviderSelection] = {}
    chosen_network_provider: Dict[str, str] = {}  # net -> provider.name

    # 1) Resolve which provider serves which requested network (in this region)
    for p in providers:
        if not unresolved:
            break

        client = Client(p.base_url, timeout=timeout)

        # Ask only for currently-unresolved nets to reduce load
        net_query = _as_list_csv(unresolved)
        try:
            inv = client.get_stations(
                network=net_query,
                latitude=latitude,
                longitude=longitude,
                minradius=minradius,
                maxradius=maxradius,
                level=level,
            )
        except (FDSNNoDataException, Exception):
            # Nothing there, or server issue; just move on
            continue

        # Which of our unresolved networks did this provider actually return?
        returned_nets = sorted({net.code for net in inv})
        claimed = [n for n in returned_nets if n in unresolved]
        if not claimed:
            continue

        # Mark those networks as resolved by this provider
        for n in claimed:
            chosen_network_provider[n] = p.name
            unresolved.discard(n)

        # Save selection (inventory now; we’ll build station list below)
        if p.name in selections:
            # Combine inventories if we ever query same provider multiple times
            selections[p.name].inventory += inv
            selections[p.name].networks = sorted(set(selections[p.name].networks) | set(claimed))
        else:
            selections[p.name] = ProviderSelection(
                provider=p,
                networks=claimed,
                stations=[],
                inventory=inv,
            )

    if unresolved:
        # Not fatal, but worth making visible.
        missing = ", ".join(sorted(unresolved))
        print(f"[warn] Networks not found on any provider in this region: {missing}")

    # 2) Build station lists per provider
    for sel in selections.values():
        stas = sorted({sta.code for net in sel.inventory for sta in net})
        sel.stations = stas

    # 3) Fetch waveforms per provider (must be done per server)
    out = Stream()
    for sel in selections.values():
        if not sel.networks or not sel.stations:
            continue

        client = Client(sel.provider.base_url, timeout=timeout)

        network_csv = _as_list_csv(sel.networks)
        station_csv = _as_list_csv(sel.stations)

        try:
            st = client.get_waveforms(
                network=network_csv,
                station=station_csv,
                location=location,
                channel=channel,
                starttime=starttime,
                endtime=endtime,
                attach_response=attach_response,
            )
            out += st
        except FDSNNoDataException:
            print(f"[warn] No waveforms from {sel.provider.name} for {network_csv}.{station_csv}")
        except Exception as e:
            print(f"[warn] Waveform fetch failed from {sel.provider.name}: {e!r}")

    if merge and len(out):
        out.merge(method=1, fill_value="interpolate")

    return out, selections


def convert_to_catalog(events: pd.DataFrame,
                             assignments: pd.DataFrame,
                             algorithm_name: str = "SeisBench-PyOcto") -> Catalog:
    """
    Convert SeisBench-style 'events' and 'assignments' DataFrames to an ObsPy Catalog.

    Parameters
    ----------
    events : pd.DataFrame
        Columns must include:
            ['idx', 'time', 'latitude', 'longitude', 'depth']
        where 'time' is in UNIX epoch seconds.

    assignments : pd.DataFrame
        Columns must include:
            ['event_idx', 'station', 'phase', 'time', 'residual']

    algorithm_name : str, optional
        Added as a comment to each origin, default = 'SeisBench-PyOcto'.

    Returns
    -------
    catalog : obspy.core.event.Catalog
        Fully populated ObsPy Catalog object.

    Notes
    -----
    - Magnitude is filled with a placeholder (99.0) for VELEST compatibility.
    - Residuals are attached to picks as comments ('residual=... s').
    - Each pick's time is absolute UTC (converted from UNIX epoch seconds).
    """

    cat = Catalog()

    for _, ev in events.iterrows():
        # --- Origin metadata ---
        origin_time = UTCDateTime(float(ev["time"]))
        origin = Origin(
            time=origin_time,
            latitude=float(ev["latitude"]),
            longitude=float(ev["longitude"]),
            depth=float(ev["depth"]) * 1000.0,  # ObsPy uses meters
            depth_type="from location",
            evaluation_mode="automatic",
            evaluation_status="preliminary",
            quality=OriginQuality(used_phase_count=0),
            comments=[Comment(text=f"Localized by: {algorithm_name}", force_resource_id=False)]
        )

        # Dummy magnitude
        mag = Magnitude(mag=99.0, magnitude_type="ML")

        # --- Event container ---
        event = Event(origins=[origin], magnitudes=[mag])

        # --- Associated picks ---
        these_picks = assignments[assignments["event_idx"] == ev["idx"]]
        origin.quality.used_phase_count = len(these_picks)

        for _, p in these_picks.iterrows():
            pick = Pick()
            pick.time = UTCDateTime(float(p["time"]))
            parts = p["station"].split(".")
            net, sta, loc = (parts + ["", "", ""])[:3]

            pick.waveform_id = WaveformStreamID(
                network_code=net,
                station_code=sta,
                location_code=loc,
                channel_code=p["phase"]
            )
            pick.phase_hint = p["phase"]
            pick.evaluation_mode = "automatic"
            pick.evaluation_status = "preliminary"
            pick.comments = [Comment(text=f"residual={p['residual']:.3f} s")]
            event.picks.append(pick)

        cat.append(event)

    return cat


def SRC_velocity_format(file: str, surface_Vp: Optional[float] = None, surface_Vs: Optional[float] = None) -> pd.DataFrame:
    """Parse SRC/Eqlocl-style 1-D velocity model → DataFrame columns: depth, vp, vs"""
    def is_comment_or_blank(line: str) -> bool:
        s = line.strip()
        return (not s) or s.startswith('#')
    def first_floats(line: str, n: int) -> list[float]:
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
        return [float(x) for x in nums[:n]]

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace("\r\n","\n").replace("\r","\n")
    lines = [ln for ln in text.split("\n") if not is_comment_or_blank(ln)]
    if len(lines) < 3:
        raise ValueError("File too short.")

    n_layers = int(round(first_floats(lines[1], 1)[0]))
    start_idx, end_idx = 2, 2 + n_layers
    if end_idx > len(lines):
        raise ValueError(f"Layer count {n_layers} exceeds available lines.")

    depths, vps, vss = [], [], []
    for i in range(start_idx, end_idx):
        d, vp, vs = first_floats(lines[i], 3)
        depths.append(d); vps.append(vp); vss.append(vs)

    df = pd.DataFrame({"depth": depths, "vp": vps, "vs": vss}).sort_values("depth").reset_index(drop=True)

    # surface row @ depth=0
    has_zero = (len(df) and abs(df.iloc[0]["depth"]) < 1e-9)
    if surface_Vp is not None and surface_Vs is not None:
        surf = pd.DataFrame([{"depth": 0.0, "vp": float(surface_Vp), "vs": float(surface_Vs)}])
    else:
        surf = pd.DataFrame([{"depth": 0.0, "vp": float(df.iloc[0]["vp"]), "vs": float(df.iloc[0]["vs"])}])
    if has_zero:
        df.iloc[0] = surf.iloc[0]
    else:
        df = pd.concat([surf, df], ignore_index=True)
    return df





def plot_station_picks_panel(
    st,
    stations,                 # list like ["JEER","CLIF","TOT"]
    pick_dict=None,
    assignments=None,
    event_idx=0,
    channel="*Z",
    window_pre_p=5.0,
    window_post_s=5.0,
    fallback_post_p=10.0,
    sharex=True,
):
    """
    Plot stacked waveforms (one station per subplot) with P/S pick lines.

    Picks can come from exactly one of:
      - pick_dict: {sta: {"P": PhasePick, "S": PhasePick}} where picks have .datetime
      - assignments: DataFrame with columns ["event_idx","station","time","phase"] (+optional "probability")

    Stations are sorted by P arrival time (earliest first), and ALL panels are plotted
    relative to the earliest P pick (global time origin), so timing aligns across stations.
    Uses one trace per station, selected via ObsPy select(station=sta, channel=channel).
    """

    if (pick_dict is None) == (assignments is None):
        raise ValueError("Provide exactly one of pick_dict or assignments")

    # ---- helper: extract P/S pick times for one station ----
    def get_picks_for_station(sta):
        P_time = None
        S_time = None

        if pick_dict is not None:
            P_pick = pick_dict.get(sta, {}).get("P")
            S_pick = pick_dict.get(sta, {}).get("S")
            if P_pick is not None:
                P_time = P_pick.datetime
            if S_pick is not None:
                S_time = S_pick.datetime
            return P_time, S_time

        # assignments path
        df = assignments
        if "event_idx" in df.columns:
            df = df[df["event_idx"] == event_idx]

        def _sta_from_id(s):
            parts = str(s).split(".")
            return parts[1] if len(parts) >= 2 else str(s)

        df = df.copy()
        df["sta_code"] = df["station"].map(_sta_from_id)
        df_sta = df[df["sta_code"] == sta]

        def _pick_time(phase):
            sub = df_sta[df_sta["phase"].astype(str).str.upper() == phase]
            if len(sub) == 0:
                return None
            # choose best pick if multiple
            if "probability" in sub.columns:
                sub = sub.sort_values("probability", ascending=False)
            return UTCDateTime(float(sub.iloc[0]["time"]))

        P_time = _pick_time("P")
        S_time = _pick_time("S")
        return P_time, S_time

    # ---- collect picks for requested stations ----
    picked = []
    for sta in stations:
        P_time, S_time = get_picks_for_station(sta)
        picked.append((sta, P_time, S_time))

    # keep only stations with a P pick (required for alignment)
    picked = [(sta, P, S) for sta, P, S in picked if P is not None]
    if len(picked) == 0:
        raise ValueError("No P picks available for the requested stations")

    # sort by P time (earliest first)
    picked.sort(key=lambda x: x[1])

    # global reference time: earliest P pick - prewindow
    P0 = picked[0][1]
    t0 = P0 - window_pre_p

    # global end time: latest (S+post) or (P+fallback)
    t_end_global = max(
        (S + window_post_s) if S is not None else (P + fallback_post_p)
        for _, P, S in picked
    )

    # ---- plotting ----
    n = len(picked)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=sharex)
    if n == 1:
        axes = [axes]

    for ax, (sta, P_time, S_time) in zip(axes, picked):
        trs = st.select(station=sta, channel=channel)
        if len(trs) == 0:
            ax.set_title(f"{sta} (no trace for channel='{channel}')")
            ax.axis("off")
            continue

        tr = trs[0]  # choose first match
        tr_win = tr.copy().trim(starttime=t0, endtime=t_end_global)

        t = tr_win.times(reftime=t0)
        ax.plot(t, tr_win.data, "k", lw=0.8)

        # vertical pick lines placed in global time coordinates
        ax.axvline(P_time - t0, color="r", lw=2)
        if S_time is not None:
            ax.axvline(S_time - t0, color="b", lw=2)

        ax.set_ylabel(sta)
        ax.set_title(tr.id)

    axes[-1].set_xlabel(f"Time since (first P − {window_pre_p:.0f} s)")
    plt.tight_layout()
    plt.show()