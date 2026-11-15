from pydantic import BaseModel


class Granule(BaseModel):
    """One NISAR acquisition, with the provenance judges (and scientists) need to trace it."""

    id: str
    datetime: str
    polarization: str
    track: int | None = None
    frame: int | None = None
    orbit_direction: str | None = None
    product_version: str | None = None
    maturity: str | None = None  # BETA / PROVISIONAL / VALIDATED (NISAR release maturity)
    crid: str | None = None  # composite release id, e.g. P05023
    orbit_quality: str | None = None  # precise / medium / near-real-time / forecast
    frame_coverage: str | None = None  # full / partial


class HistogramBin(BaseModel):
    db: float
    pixels: int


class Stats(BaseModel):
    aoi_km2: float
    masked_km2: float = 0.0
    decrease_km2: float
    open_water_km2: float = 0.0  # subset of decrease: land -> open-water darkness
    water_line_db: float | None = None  # this scene's open-water level (HH dB)
    increase_km2: float
    # Dual-pol split of increase_km2 (only when the mode has a cross-pol term)
    increase_double_bounce_km2: float | None = None
    increase_volume_km2: float | None = None
    increase_rough_water_km2: float | None = None
    changed_pct: float
    # Landscape-scale (~300 m block mean) coherent change beyond its own 3-sigma threshold
    subtle_threshold_db: float | None = None
    subtle_increase_km2: float | None = None
    subtle_decrease_km2: float | None = None
    mean_change_db: float
    noise_sigma_db: float = 0.0
    histogram: list[HistogramBin]
    # Ground on each date (ground.py): {"water": {"before_km2": .., "after_km2": ..}, ...}
    ground: dict[str, dict[str, float]] | None = None


class Region(BaseModel):
    """One of the largest connected blobs of change, for numbering on the map."""

    n: int
    cls: str  # decrease | increase | volume | rough_water
    km2: float
    lat: float
    lon: float
    d_hh: float | None = None  # median HH change in the region, dB
    d_hv: float | None = None  # median HV change, when the mode has HV
    suspect: bool = False  # large and shifted alike in HH and HV: possible acquisition artefact
    share_pct: float | None = None  # share of the scene's land change in this zone
    breakdown: dict[str, float] | None = None  # km² per class in the zone
    box: tuple[tuple[float, float], tuple[float, float]] | None = None  # zone [[s, w], [n, e]]
    extent_km2: float | None = None  # size of the area itself
    density_pct: float | None = None  # share of the area's ground that changed
    rings: list[list[tuple[float, float]]] | None = None  # outline rings, [lat, lon]


class EventSummary(BaseModel):
    id: str
    title: str
    hazard: str
    synthetic: bool
    pre_date: str
    post_date: str
    changed_pct: float


class StoryPhoto(BaseModel):
    file: str  # relative to the event folder in story.json; a /files URL once served
    caption: str
    credit: str
    licence: str
    source_url: str


class StorySource(BaseModel):
    label: str
    url: str


class StoryVideo(BaseModel):
    """A video about the event, shown as a link on the step (a YouTube watch URL)."""

    title: str
    url: str


class StoryPath(BaseModel):
    """A line on the map: the river the mud followed, a fault trace."""

    label: str
    points: list[tuple[float, float]]  # [lat, lon]


class PointRadar(BaseModel):
    """What the radar measured at a story point (story.point_radar); never hand-written."""

    covered: bool  # False: outside the box, or no radar sample on both dates
    pre_db: float | None = None
    post_db: float | None = None
    delta_db: float | None = None
    flag: str | None = None
    changed_pct_300m: float | None = None  # share of the ground within 300 m that changed


class StoryShow(BaseModel):
    """What the map should show for a step of the story; unset fields leave the map as it is."""

    view: str | None = None  # "ground" (before & after) | "change" (only changes)
    subtle: bool | None = None  # the Landscape 500 m layer
    basemap: str | None = None  # gray | satellite | streets | hls | modis
    radar_opacity: float | None = None  # 0 hides the radar so the photos underneath show
    split: float | None = None  # divider position, 0 (all after) .. 1 (all before)


class StoryPoint(BaseModel):
    id: str
    title: str
    headline: str | None = None  # one big line, e.g. a number: "214 km² of forest changed"
    area: int | None = None  # a numbered area to fly to and outline instead of the pin
    show: StoryShow | None = None
    lat: float
    lon: float
    zoom: float = 13
    what_happened: str
    photos: list[StoryPhoto] = []
    videos: list[StoryVideo] = []
    sources: list[StorySource] = []
    radar: PointRadar | None = None


class Story(BaseModel):
    summary: str
    path: StoryPath | None = None
    points: list[StoryPoint]


class Event(BaseModel):
    id: str
    title: str
    hazard: str
    location: str
    description: str
    synthetic: bool
    product: str
    threshold_db: float
    # NxN multilook applied to fit the processing machine's memory (1 = native grid), and the
    # resulting pixel size in metres. Older events have neither.
    looks: int = 1
    pixel_m: float | None = None
    # True when the threshold was set from the scene (max of 3 dB and 3 x noise sigma)
    threshold_auto: bool = False
    cross_pol: bool = False  # HV/VH available: brighter class split by scattering mechanism
    # Grayscale stretch [lo, hi] in dB shared by before.png and after.png
    display_db: tuple[float, float] | None = None
    # Leaflet order: [[south, west], [north, east]]
    bounds: tuple[tuple[float, float], tuple[float, float]]
    pre: Granule
    post: Granule
    stats: Stats
    regions: list[Region] = []
    suspects: list[Region] = []
    area_rule: dict | None = None  # how areas were defined, and the share of change they hold
    # Nearest NASA HLS optical scene per radar date: [{date, layer} | None, ...] or None
    hls: list[dict | None] | None = None
    layers: dict[str, str]
    # Hand-written story (story.json), with a radar reading computed at each point
    story: Story | None = None


class Probe(BaseModel):
    """One pixel of the display grid, read on click."""

    lat: float
    lon: float
    pre_db: float | None
    post_db: float | None
    delta_db: float | None
    flag: str | None  # "decrease" | "increase" | "volume" | "rough_water" | None
    hv_pre_db: float | None = None
    hv_post_db: float | None = None
