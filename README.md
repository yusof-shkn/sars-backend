# SAR Change Watch: NISAR change detection (backend)

NASA Space Apps Challenge 2026, "Dancing with the SARs". This repository holds the processing
pipeline and the API; the map is in [sars-frontend](https://github.com/yusof-shkn/sars-frontend).

Before/after change detection on NISAR L-band GCOV, on a single full-screen map: swipe compare,
change classes, numbered areas of concentrated change, click-to-probe pixel values, basemaps
(gray / satellite / streets / NASA HLS and MODIS photos of both dates), and one glass sidebar on
the left: a searchable event dropdown, then everything about the open event, with the numbered
areas in their own searchable list (by number, or by darker / growth / double bounce). An "Areas"
switch shows or hides the numbered outlines. Opening or closing the sidebar never moves the map.

## Run
Clone both repositories side by side. Four processed events (the Meghna monsoon flood, the
Thunder Bay 36 fire, the Langtang Lirung glacier collapse and the Rasuwa debris flow it set off,
and the Mozambique floods on the Limpopo) are included in `data/events/`, so no download or login
is needed to run the app.

    git clone https://github.com/yusof-shkn/sars-backend.git
    git clone https://github.com/yusof-shkn/sars-frontend.git
    cd sars-backend && uv sync && uv run uvicorn main:app --reload
    cd sars-frontend && bun install && bun run start        # VITE_API_URL defaults to http://localhost:8000

Optional, for the NASA photo basemaps without NASA's servers: `uv run python -m app.modules.events.pipeline tiles` (see step 4 below).

## Process a real event
1. Earthdata Login: create a token at https://urs.earthdata.nasa.gov (Generate Token) and put
   `EARTHDATA_TOKEN=...` in `.env`, or just run step 3 and type your username and
   password once; it is stored in `~/.netrc`.
2. Don't know where to look? NASA EONET lists open hazards with a ready-made bbox:

       uv run python -m app.modules.events.pipeline eonet --days 60 --category floods

   Then find a before/after pair over the area (no login needed for the search):

       uv run python -m app.modules.events.pipeline search --bbox W S E N --start 2026-06-17

   Granules are grouped by pass geometry + mode + polarization, with each group's footprint.
   Pick two dates from ONE group and a bbox well inside the footprint (frames are tilted).
3. Run it. URLs are read in place over HTTPS, so only the bbox's chunks are fetched (~100 MB per
   date instead of 1–9 GB); the cropped subset is cached in `data/subsets/`, so re-runs are instant:

       uv run python -m app.modules.events.pipeline run --id my-flood --pre <url> --post <url> \
           --bbox W S E N --title "..." --hazard flood --location "..."

   Add `--download` to pull whole files instead; `--threshold` overrides the automatic one.
   The release maturity (BETA / PROVISIONAL / VALIDATED, from the collection and the CRID in the
   file name), orbit-data quality and frame coverage are recorded per granule and shown in the
   app. All three example events use PROVISIONAL products (P05023); VALIDATED products exist from
   2026-10-21 and NASA plans to reprocess earlier dates in 2027.

   **Memory adapts to the machine.** Before processing, the pipeline reads how much memory it
   can use (free memory, and any cgroup cap such as `systemd-run -p MemoryMax=...`). Processing
   peaks at about 255 bytes per pixel with HH+HV, so a capable computer keeps the native 10/20 m
   grid, and a small one (a WSL VM with ~1.8 GB, say) averages 2x2 or 3x3 blocks of linear power
   first, the standard SAR multilook; the HDF5 read cache shrinks the same way. The choice is
   recorded in `event.json` (`looks`, `pixel_m`) and printed. `--looks N` forces it.
4. Keep the NASA photos (HLS, MODIS) for every event on disk, so they load without NASA's servers
   (GIBS sends them with no-store, so the browser never caches them):

       uv run python -m app.modules.events.pipeline tiles          # all events; or --id <id> ...

   Everything a 1920 x 1080 screen can show around the scene, from two zoom levels out of the
   opening view down to each layer's finest level, for both dates (about 3,000 tiles and 280 MB
   for the three events). The app reads this copy first and asks GIBS only for tiles outside it.
   The Esri and OpenStreetMap basemaps are not downloaded: their terms forbid offline copies, so
   they rely on the browser cache (about a day for both).
5. Events processed before the ground layer existed get it without fetching radar again:

       uv run python -m app.modules.events.pipeline ground         # all events; or --id <id> ...

## Method (per pixel, native 10/20 m grid)
1. 5×5 box mean of gamma0 on each date (speckle), then `d = 10·log10(after / before)`.
2. Pixels darker than the noise floor on both dates are masked; GCOV's sub-swath `mask` drops
   invalid samples.
3. Threshold = max(3 dB, 3 × scene noise σ) unless given; a 5×5 majority sieve removes specks.
4. With a cross-pol channel (HV), the brighter class is split by mechanism: still water-dark
   (≤ −9 dB) → wind-roughened open water; otherwise HH rising at least half a threshold more
   than HV → double bounce (flooded vegetation, structures), else → growth / rougher surface.
   A "Colour (HH/HV)" view shows the standard false-colour composite (red HH, green HV, blue ratio).
   "Open water" is the strict subset of darker pixels that went from land level to below the
   scene's own water line: halfway between its calm-water histogram peak and the water/land
   (Otsu) split, as NASA's ARSET flood method does by hand (−16 dB when a scene has no water
   peak). Checked against NASA's optical surface-water product (OPERA DSWx-HLS) on the Meghna:
   the fixed −16 dB line caught 72% of its open water, the scene line 93% at 83% precision.
5. Statistics on the native grid. **Areas of concentrated change** (the numbered, outlined areas):
   on 50 m blocks, the share of the land (ground on at least one date) that changed, smoothed
   over 1.5 km. Every block denser than the scene's own average is in an area; that land is split
   into compact areas around its density peaks (≥ 1.75 km apart, watershed), enclosed ponds are
   filled, and a piece holding less than the listing minimum (1% of all changed land, at least
   0.25 km²) is folded into the neighbour it shares most border with. Every area at or above the
   minimum is numbered, largest first, with no cap, so no concentration of that size is left
   without a number. Each area reports its changed km², its share of all change, its land area
   and the split by class; outlines are drawn as map lines from the same blocks that are counted.
   Wind on open water is not counted. A large single blob that shifted alike in HH and HV is
   flagged separately as a possible acquisition artefact. Display layers are resampled to
   lon/lat (≤ 2048 px).
6. "Landscape 500 m": the block-mean change over land, relative to the scene's median, beyond a
   3σ threshold scaled from the pixel noise (floor 0.5 dB: unchanged forest on Thunder Bay drifts
   by 0.75 dB across the scene, so the layer is a pattern, not an edge). It shows change too small
   for any pixel but coherent over hundreds of metres: a boreal burn moves L-band HH by only ~+1 dB per pixel (trunks and
   soil are left standing), so fires appear here rather than in the pixel mask.
7. For each radar date the nearest NASA HLS (Landsat/Sentinel-2, 30 m) scene within ±8 days is
   recorded; the "HLS 30 m, both dates" basemap then shows before-date imagery left of the
   divider and after-date imagery right of it, like the radar.
8. **Ground on each date** (`ground.py`): both dates are painted in the same four classes so
   the before side of the swipe is coloured too, and the app lists each class's km² before ->
   after: water (below the scene's water line), smooth ground (below -12 dB: bare soil, mud),
   fields and shrubs (below -6 dB), bright (forest, buildings, rock, slopes facing the satellite).
   Areas count only ground the radar saw on both dates. The after date differs from the before
   date only where the change layer measured a real change (darker, double bounce, rougher);
   elsewhere it keeps the before class. Otherwise a shift below the noise threshold that happens
   to cross a class line reads as a change of ground: on Thunder Bay a +1 dB drift moved 200 km²
   from "fields" to "bright", and wind turned 32 km² of lakes into "smooth ground". The view
   therefore agrees with the change layer and its noise threshold.

Output lands in `data/events/<id>/`; the API serves it (`/api/v1/events`,
`/api/v1/events/{id}`, `/api/v1/events/{id}/probe?lat&lon`). `tests/test_change.py` holds the
ground-truth checks on the synthetic scene (recall > 97 %, false positives < 1 %, dual-pol split,
area definition). Two more checks run on the real events:

    uv run pytest                                                         # maths on synthetic scenes
    uv run python -m app.modules.events.verify                            # real events: markers, outlines, km², nothing left out

The browser checks live in sars-frontend (`e2e/`).

## Data and credits
Radar: NISAR (NASA and ISRO), L2 GCOV, distributed by the Alaska Satellite Facility. Photos:
NASA GIBS (HLS from Landsat 8/9 and Sentinel-2; MODIS Terra). Hazard list: NASA EONET.
