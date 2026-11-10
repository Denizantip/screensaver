#!/usr/bin/env python3
"""Rotating globe in the terminal, drawn with Braille characters.

Usage:  python globe_day_night_numpy.py [--fps 30] [--speed 40] [--tilt 20] [--color N]
                [--aspect 2.2] [--sun-azimuth 55] [--sun-elevation 10] [--no-night] [--real]
                [--relief relief.bin] [--no-relief] [--sea-relief]
                [--color-depth auto|truecolor|256|16] [--no-stars]
                [--preset NAME] [--relief none|land|full] [--grid|--no-grid]
                [--grid-spacing N] [--stars|--no-stars] [--star-density X]
                [--ocean-sparkle X] [--night|--no-night]
Quit with Ctrl+C.

Presets (--preset NAME): a bundle of the options below, see PRESETS in the settings.
  default  land relief, plain sea sparkle, stars, day/night   (the old behavior)
  full     land relief + sea-floor relief, stars, day/night
  simple   flat land + grid, day/night, no relief, no stars
  atlas    land + sea relief + grid, evenly lit (no day/night), no stars
  night    flat land, stars, day/night, no grid
  minimal  flat land only: no grid, no stars, no day/night
  mono     flat land + grid in one color, no stars, no day/night
Single options are applied after the preset and always win, e.g.
  --preset simple --stars --grid-spacing 15  Needs a terminal/font that supports Braille (U+2800..U+28FF).

Colors: by default each Braille cell is colored by the most important thing it
shows (land, outline, sunlit ocean or grid) and by how sunlit it is (night,
twilight or day) -- see REGION_COLORS. --color N paints everything with one
ANSI 256-color code instead; --color -1 turns color off.

Color depth (--color-depth): "auto" picks truecolor, 256 or 16 colors from the
environment. The Linux text console (TERM=linux, Ctrl+Alt+F3) cannot show truecolor
and mangles it into its 16 basic colors, so there the 16-color tables (COLORS_16
settings) are used instead; they are hand-picked for that palette.

This is the NumPy version (pip install numpy): the per-dot work and the frame
assembly are vectorized, so large terminals and high frame rates stay cheap.

The land map is a 0.5-degree bitmap (rasterized from Natural Earth, public
domain), embedded at the bottom of the settings as zlib-compressed, base64 text.

Relief: if relief.bin (720x360 little-endian int16 meters, row 0 = north, from
ETOPO 2022 ice-surface) sits next to the script, land is hillshaded and colored by
height. The hillshade is computed ONCE at startup over the whole height map (light
direction baked into the map, like a shaded-relief atlas) and then only sampled per
dot with the same map index as the land bitmap. Without the file the globe looks
as before (flat land).

Sea relief (--sea-relief): the same height map also holds the sea floor, so with this
option the oceans get their own hillshade (mid-ocean ridges, trenches, shelves) and
are colored by depth (SEA_RAMP). It replaces the plain sunlit-ocean sparkle. Without
the option (default) the globe looks exactly as before: relief on land only.

Stars: a fixed, random background field of stars in terminal cells, not moving with the
globe. Stars come in classes like real ones: most are single faint dots, a few are larger
and brighter (2 dots, a plus, a block). Globe cells always win over stars. --no-stars
turns them off. See STAR_CLASSES.

All tunable numbers live in the SETTINGS section right below. Nothing in the
code after that section contains hard-coded values.
"""
import argparse
import base64
import math
from datetime import datetime, timezone
import shutil
import sys
import time
import zlib
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Optional

try:
    import numpy as np
except ImportError:
    sys.exit("This version needs NumPy:  pip install numpy")

# =============================================================================
# SETTINGS
# =============================================================================

# --- Command-line defaults (can be overridden with --fps, --speed, ...) ------
DEFAULT_FPS = 30            # target frames per second
DEFAULT_SPEED = 40          # rotation speed, degrees per second
DEFAULT_TILT = 20           # visual axial tilt: how far the north pole leans toward you
DEFAULT_COLOR = None        # None = per-region colors (REGION_COLORS below);
                            #   an ANSI 256-color code = one color for everything;
                            #   -1 = no color
DEFAULT_COLOR_DEPTH = "auto"  # auto | truecolor | 256 | 16  (see --color-depth)
DEFAULT_ASPECT = 2.2        # fallback terminal cell height / width
DEFAULT_SEA_RELIEF = False  # True = relief for sea floor too (--sea-relief / --no-sea-relief)
DEFAULT_PRESET = "default"  # preset used when --preset is not given (see PRESETS)
DEFAULT_GRID = False        # grid on/off for the "default" preset (--grid / --no-grid)
REAL_SUN_UPDATE_SECONDS = 60  # refresh the real solar direction this often

# --- Globe appearance --------------------------------------------------------
GLOBE_MARGIN_DOTS = 0.5     # empty space between globe edge and screen edge
EDGE_THICKNESS = 0.8        # outline thickness (in terminal-cell widths)

# Grid (latitude / longitude lines)
GRID_SPACING = 30           # degrees between grid lines
GRID_LINE_HALF_WIDTH = 0.4   # line half-width in degrees (smaller = thinner; 0.01 is nearly invisible)
GRID_DENSITY = 0            # 0..1, share of dots drawn on a grid line; this is the value
                            #   while the grid is OFF. main() switches it to GRID_DENSITY_ON
                            #   when the grid is turned on (preset or --grid)
GRID_DENSITY_ON = 1         # 0..1, the same, while the grid is ON (lower = fainter grid)
GRID_MIN_Z = 0.2            # hide grid near the globe's edge (0 = show all)
GRID_MIN_COS_LAT = 0.25     # limits how wide lon-lines get near the poles

# Shading (3D look: surface is denser in the center, thinner at the edge)
SHADE_MIN = 0.15            # brightness at the very edge of the globe
SHADE_GAMMA = 0.7           # <1 keeps the center brighter for longer
LAND_DENSITY = 1            # lower = land more solid, higher = more dithered
                            #   (with relief on, flat land sits at RELIEF_FLAT / this)

# Relief (needs relief.bin). Hillshade is baked once; the light is fixed to the map.
RELIEF_FILE = "relief.bin"  # looked up next to this script unless --relief is given
RELIEF_AZIMUTH = 315        # light direction, degrees clockwise from north (315 = NW)
RELIEF_ALTITUDE = 40        # light height above the horizon, degrees
RELIEF_ZFACTOR = 60         # vertical exaggeration (0.5-degree cells hide most slope)
RELIEF_FLAT = 0.7           # dot density of flat land: headroom for lit AND shadowed slopes
RELIEF_MAX_REL = 1.5        # clamp of (slope shading / flat shading)
RELIEF_MIN_COS_LAT = 0.1    # limits east-west cell shrink near the poles
FLAT_LAND_LEVEL = "land_0"  # LAND_RAMP level used for flat land (--relief none, no height map)
METERS_PER_DEGREE = 111320.0
# Land height ramp: (name, lowest height in meters, daylight RGB), ascending.
# The last entry (snow) is not chosen by height but by the latitude-dependent snow
# line:  snow_line = SNOW_LINE_EQUATOR * cos(lat) ** SNOW_LINE_POWER
# Night and dusk colors are derived from the day color (see LAND_*_TINT/MIX).
# Add, remove or recolor levels freely; everything else adapts.
LAND_RAMP = (
    ("land_-4",    -1000, ( 6, 56, 70)),
    ("land_-3",    -600, (  8, 70,  60)),
    ("land_-2",    -300, ( 22, 100,  68)),
    ("land_-1",    -100, ( 30, 120,  74)),
    ("land_0",    0, ( 38, 140,  82)),   # coastal lowland
    ("land_1",  100, ( 60, 168,  90)),
    ("land_2",  300, ( 98, 176,  84)),
    ("land_3",  600, (140, 172,  84)),
    ("land_4", 1000, (174, 160,  90)),   # dry upland
    ("land_5", 1500, (178, 138,  92)),
    ("land_6", 2200, (156, 118,  88)),   # mountains
    ("land_7", 3000, (150, 134, 122)),   # bare rock
    ("snow",   None, (238, 243, 248)),
)
# Sea relief (only with --sea-relief). Same idea as land, using the negative heights.
SEA_ZFACTOR = 100           # vertical exaggeration of the sea floor
SEA_FLAT = 0.60             # dot density of flat sea floor (a bit below land's RELIEF_FLAT)
# Depth ramp: (name, lowest height in meters, daylight RGB), ascending. Anything below
# the first level uses the first level; lakes above sea level use the last one.
SEA_RAMP = (
    ("sea_0", -11000, ( 14,  36,  92)),   # trenches
    ("sea_1",  -6000, ( 18,  50, 118)),   # abyssal plain
    ("sea_2",  -4500, ( 24,  66, 145)),
    ("sea_3",  -3000, ( 30,  85, 170)),
    ("sea_4",  -2000, ( 40, 106, 192)),   # ridges, continental rise
    ("sea_5",  -1000, ( 56, 128, 208)),
    ("sea_6",   -400, ( 80, 155, 220)),   # continental slope
    ("sea_7",   -150, (112, 182, 230)),
    ("sea_8",    -30, (150, 208, 238)),   # shallow shelf
)
SEA_NIGHT_TINT, SEA_NIGHT_MIX = (8, 16, 45), 0.60
SEA_TWILIGHT_TINT, SEA_TWILIGHT_MIX = (120, 80, 110), 0.40
SNOW_LINE_EQUATOR = 6000
SNOW_LINE_POWER = 1.5
LAND_NIGHT_TINT, LAND_NIGHT_MIX = (12, 22, 48), 0.80      # dark blue silhouette
LAND_TWILIGHT_TINT, LAND_TWILIGHT_MIX = (205, 110, 50), 0.55   # orange dusk

# Day / night lighting. The sun is fixed relative to your screen, so as the
# globe spins, land moves from the dark side into the light and back again.
DAY_NIGHT = True            # False = no lighting (uniform globe)
SUN_AZIMUTH = 55            # sun direction, degrees left/right of the viewer
                            #   (+ = sun on the right, - = left, 0 = behind you)
SUN_ELEVATION = 10          # sun direction, degrees above (+) / below (-) equator
TERMINATOR_SOFTNESS = 0.15  # width of the dawn/dusk blend (0 = razor-sharp line)
NIGHT_LEVEL = 0.12          # how bright the night side is (0 = black, 1 = same as day)
DAY_AMBIENT = 0.6           # day-side brightness where the sun is at a low angle
                            #   (1 = flat daylight, lower = stronger noon highlight)
OCEAN_DAY_DENSITY = 0.1    # share of dots lighting up the sunlit ocean (0 = none)

# --- Background stars (--no-stars turns them off) ----------------------------
STARS = True                # star field on/off for the "default" preset (--stars / --no-stars)
STAR_DENSITY = 0.03         # stars per terminal cell (0.03 = about one in 33 cells)
STAR_SEED = 7               # fixed seed: the same sky on every run
STAR_GLOBE_MARGIN = 0.03    # no stars within this fraction of the globe radius past its rim
# Star classes, ascending brightness. Each: (share of stars, Braille dot offsets
# from the star's dot, RGB color). Most stars are faint single dots; the bigger,
# brighter ones are rare, as in a real sky.
STAR_CLASSES = (
    (0.80, ((0, 0),),                                       ( 70,  80, 105)),  # faint point
    (0.14, ((0, 0), (1, 0)),                                (120, 130, 160)),  # small pair
    (0.05, ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)),      (190, 200, 220)),  # bright plus
    (0.01, ((-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1)),                       (255, 248, 235)),  # brightest block
)
STAR_16 = (8, 7, 15, 15)    # 16-color codes for the same classes (Linux console)

# --- Presets (--preset NAME) -------------------------------------------------
# Every key is optional; what a preset leaves out comes from the DEFAULT_* / STARS /
# DAY_NIGHT / OCEAN_DAY_DENSITY settings above, so "default" (empty) = the old look.
#   relief  : "none" flat land | "land" hillshaded land | "full" land + sea floor
#   grid    : latitude / longitude lines on or off
#   stars   : background stars on or off
#   night   : day/night lighting on or off
#   sparkle : OCEAN_DAY_DENSITY, the dot density of the plain sunlit sea (ignored
#             with relief "full", which fills the sea itself)
#   color   : one ANSI 256-color code for everything (None = per-region colors)
MONO_COLOR = 114            # the one color of the "mono" preset
PRESETS = {
    "default": {},
    "full":    {"relief": "full", "grid": False, "stars": True,  "night": True},
    "simple":  {"relief": "none", "grid": True,  "stars": False, "night": True,
                "sparkle": 0},
    "atlas":   {"relief": "full", "grid": True,  "stars": False, "night": False},
    "night":   {"relief": "none", "grid": False, "stars": True,  "night": True},
    "minimal": {"relief": "none", "grid": False, "stars": False, "night": False,
                "sparkle": 0},
    "mono":    {"relief": "none", "grid": True,  "stars": False, "night": False,
                "sparkle": 0, "color": MONO_COLOR},
}
RELIEF_MODES = ("none", "land", "full")

# --- Region colors (ANSI 256-color codes) ------------------------------------
# A Braille cell can only have one color, so a cell takes the color of the most
# important thing drawn in it: land > outline > sunlit ocean > grid.
# Each entry is (night, twilight, day), chosen by how sunlit the dot is.
# Only used when --color / DEFAULT_COLOR is not set. Without day/night
# lighting everything uses the "day" color.
# REGION_COLORS = {
#     "land":  (24, 173, 78),     # dark blue silhouette -> orange dusk -> green
#     "edge":  (60, 103, 153),    # globe outline: dim violet -> pale blue
#     "ocean": (17, 33, 39),      # sparkle on the sunlit sea: navy -> bright blue
#     "grid":  (236, 240, 67),    # latitude / longitude lines: dark gray -> steel blue
# }

REGION_COLORS = {
    "edge": (
        (45, 55, 85),
        (105, 140, 190),
        (170, 205, 230),
    ),
    "ocean": (
        (5, 18, 35),
        (20, 75, 115),
        (35, 125, 190),
    ),
    "grid": (
        (55, 60, 75),
        (125, 135, 145),
        (195, 205, 215),
    ),
}

# 16-color tables, for the Linux text console. Codes 0-7 are the normal colors, 8-15
# the bright ones (0 black, 1 red, 2 green, 3 yellow/brown, 4 blue, 5 magenta, 6 cyan,
# 7 light gray; +8 = bright; 8 = dark gray, 15 = white). The land and sea ramps are
# mapped by height: (lowest height in meters, color code), ascending, so adding levels
# to LAND_RAMP / SEA_RAMP needs no change here.
LAND_16_RAMP = ((-11000, 2), (100, 10), (600, 11), (1500, 3), (3000, 7))
LAND_16_SNOW = 15
LAND_16_NIGHT, LAND_16_TWILIGHT = 8, 3
SEA_16_RAMP = ((-11000, 4), (-2000, 12), (-400, 6), (-150, 14))
SEA_16_NIGHT, SEA_16_TWILIGHT = 4, 5
REGION_16 = {               # (night, twilight, day)
    "edge":  (8, 12, 14),
    "ocean": (4, 4, 12),
    "grid":  (8, 8, 7),
}
# 256-color palette layout (xterm): 6x6x6 cube from code 16, then 24 grays from 232
XTERM_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)
XTERM_CUBE_FIRST = 16
XTERM_GRAY_FIRST, XTERM_GRAY_START, XTERM_GRAY_STEP, XTERM_GRAY_COUNT = 232, 8, 10, 24
# Auto-detection of the color depth from the environment
TERM_16_COLORS = ("linux",)             # TERM values that only have 16 colors
COLORTERM_TRUECOLOR = ("truecolor", "24bit")
TERM_256_MARK = "256"                   # TERM containing this = 256 colors

TWILIGHT_RANGE = (0.2, 0.8)  # dawn/dusk band, as share of daylight (0 = night, 1 = day)
                             #   (widen it by raising TERMINATOR_SOFTNESS too)

# --- Terminal / rendering geometry -------------------------------------------
RESERVED_ROWS = 1           # terminal lines left unused (prevents scrolling)
FALLBACK_TERMINAL_SIZE = (80, 24)   # used if terminal size can't be detected

BRAILLE_BASE = 0x2800       # Unicode code point of the empty Braille cell
DOTS_X = 2                  # Braille dots per cell, horizontally
DOTS_Y = 4                  # Braille dots per cell, vertically
# Bit value of each dot, indexed [dot_x][dot_y]
BRAILLE_BITS = ((0x01, 0x02, 0x04, 0x40),
                (0x08, 0x10, 0x20, 0x80))

# Ordered-dithering matrix (any square matrix of unique values 0..N*N-1)
BAYER = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)

# --- World map data ----------------------------------------------------------
MAP_W, MAP_H = 720, 360     # bitmap size (equirectangular projection)
LON_MIN = -180              # longitude of the left edge of the map
LAT_MAX = 90                # latitude of the top edge of the map
MAP_PX_PER_DEG = 2          # map pixels per degree (720 / 360)
HALF_TURN = 180             # degrees
FULL_TURN = 360             # degrees

# --- ANSI escape sequences ---------------------------------------------------
ANSI_ENTER_SCREEN = "\x1b[?1049h\x1b[?25l\x1b[40m\x1b[2J"  # alt screen, hide cursor, black bg, clear
ANSI_LEAVE_SCREEN = "\x1b[0m\x1b[?25h\x1b[?1049l"  # reset color, show cursor, back
ANSI_CLEAR = "\x1b[2J"
ANSI_HOME = "\x1b[H"
ANSI_COLOR = "\x1b[38;5;{}m"
ANSI_RGB_COLOR = "\x1b[38;2;{};{};{}m"  # 24-bit truecolor
ANSI_FG_16 = "\x1b[{}m"                 # 16 colors: 30-37 normal, 90-97 bright
ANSI_FG_NORMAL_BASE, ANSI_FG_BRIGHT_BASE = 30, 90
COLOR_DEPTHS = ("auto", "truecolor", "256", "16")

# --- Embedded land bitmap (zlib + base64, 1 bit per pixel, row 0 = north) ----
LAND_B64 = (
    "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"
)

# =============================================================================
# CODE
# =============================================================================

BAYER_SIZE = len(BAYER)
BAYER_LEVELS = BAYER_SIZE * BAYER_SIZE
DOT_WIDTH = 1 / DOTS_X      # width of one dot, in terminal-cell widths

# What a dot can show, in ascending priority (index = priority; 0 = nothing)
LAND_NAMES = tuple(name for name, _, _ in LAND_RAMP)
SEA_NAMES = tuple(name for name, _, _ in SEA_RAMP)
# Sea depth levels rank below the grid, so grid lines stay visible over shaded sea.
KIND_NAMES = ("none",) + SEA_NAMES + ("grid", "ocean", "edge") + LAND_NAMES
KIND_SEA_FIRST = 1                      # deepest sea level
KIND_GRID = KIND_SEA_FIRST + len(SEA_NAMES)
KIND_OCEAN, KIND_EDGE = KIND_GRID + 1, KIND_GRID + 2
KIND_LAND_FIRST = KIND_GRID + 3         # lowest land level; the last one is snow
KIND_SNOW = len(KIND_NAMES) - 1
BAND_NIGHT, BAND_TWILIGHT, BAND_DAY = 0, 1, 2
BANDS = 3
# Star colors are appended after the kind colors in the color table
STAR_CODE_BASE = len(KIND_NAMES) * BANDS
EMPTY_CELL = " "            # a plain space: stays blank in every font
BRAILLE_BIT_TABLE = np.array(BRAILLE_BITS, dtype=np.uint8)  # [dot_x, dot_y] -> bit
BAYER_ARRAY = np.array(BAYER)
# Output character for every possible cell value (0 = empty -> EMPTY_CELL)
CELL_CODEPOINTS = np.array([ord(EMPTY_CELL)] + [BRAILLE_BASE + b for b in range(1, 256)],
                           dtype="<u4")
NEWLINE_CODEPOINT = ord("\n")


def load_land():
    """Return a flat boolean array of MAP_W*MAP_H, True = land. Row 0 is the north edge."""
    packed = np.frombuffer(zlib.decompress(base64.b64decode(LAND_B64)), dtype=np.uint8)
    if packed.size * 8 != MAP_W * MAP_H:
        sys.exit(f"land bitmap has {packed.size * 8} bits, expected {MAP_W * MAP_H}")
    return np.unpackbits(packed).astype(bool)  # most significant bit first


class Relief(NamedTuple):
    """Per-map-pixel relief data, flat arrays of MAP_W*MAP_H (row 0 = north)."""
    light: np.ndarray       # float32 hillshade, ~RELIEF_FLAT on flat ground, 0..1
    kind: np.ndarray        # uint8 land kind (KIND_LAND_FIRST .. KIND_SNOW)
    sea_light: Optional[np.ndarray] = None   # same for the sea floor (None = no sea relief)
    sea_kind: Optional[np.ndarray] = None    # uint8 sea depth kind (KIND_SEA_FIRST ..)


def load_elevation(path):
    """Return int16 elevation [MAP_H, MAP_W] in meters, or None if unavailable."""
    try:
        data = np.fromfile(path, dtype="<i2")
    except OSError:
        return None
    if data.size != MAP_W * MAP_H:
        sys.exit(f"{path}: {data.size} values, expected {MAP_W * MAP_H} (720x360 int16)")
    return data.reshape(MAP_H, MAP_W)


def hillshade(e, lat, flat, zfactor):
    """Hillshade of a float32 height map [MAP_H, MAP_W]; ~`flat` on level ground, 0..1."""
    dy = METERS_PER_DEGREE / MAP_PX_PER_DEG
    dx = dy * np.maximum(np.cos(np.radians(lat)), RELIEF_MIN_COS_LAT)[:, None]
    padded = np.pad(e, ((1, 1), (0, 0)), mode="edge")           # poles: repeat edge
    dzdx = (np.roll(e, -1, axis=1) - np.roll(e, 1, axis=1)) / (2 * dx)  # wraps at 180
    dzdy = (padded[:-2] - padded[2:]) / (2 * dy)                # row 0 = north
    dzdx *= zfactor
    dzdy *= zfactor

    az, alt = math.radians(RELIEF_AZIMUTH), math.radians(RELIEF_ALTITUDE)
    lx, ly, lz = (math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt),
                  math.sin(alt))
    shading = (lz - dzdx * lx - dzdy * ly) / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1)
    rel = np.clip(shading / lz, 0.0, RELIEF_MAX_REL)            # 1.0 = flat ground
    return np.clip(flat * rel, 0.0, 1.0).astype(np.float32)


def build_relief(elev, sea_relief=False):
    """Hillshade + height tiers for the whole map, computed once.

    elev: int16 [MAP_H, MAP_W] or None (flat world). For the land shading the sea
    floor is clamped to sea level, otherwise every coast would become a huge cliff.
    With sea_relief the sea gets its own hillshade (land clamped to sea level) and
    depth tiers.
    """
    lat = LAT_MAX - (np.arange(MAP_H) + 0.5) / MAP_PX_PER_DEG
    if elev is None:
        light = np.full(MAP_W * MAP_H, RELIEF_FLAT, dtype=np.float32)
        kind = np.full(MAP_W * MAP_H, KIND_LAND_FIRST + LAND_NAMES.index(FLAT_LAND_LEVEL),
                       dtype=np.uint8)
        return Relief(light, kind)

    light = hillshade(np.maximum(elev, 0).astype(np.float32), lat,
                      RELIEF_FLAT, RELIEF_ZFACTOR)

    snow_line = SNOW_LINE_EQUATOR * np.cos(np.radians(lat))[:, None] ** SNOW_LINE_POWER
    # Levels use the real heights (land below sea level gets its own levels);
    # only the hillshade above works on the clamped copy.
    h = elev.astype(np.float32)
    floors = [metres for _, metres, _ in LAND_RAMP if metres is not None]
    level = np.searchsorted(floors, h, side="right") - 1        # -1 = below 1st floor
    kind = (KIND_LAND_FIRST + np.maximum(level, 0)).astype(np.uint8)
    kind[h >= snow_line] = KIND_SNOW

    sea_light = sea_kind = None
    if sea_relief:
        sea_light = hillshade(np.minimum(elev, 0).astype(np.float32), lat,
                              SEA_FLAT, SEA_ZFACTOR).ravel()
        sea_floors = [metres for _, metres, _ in SEA_RAMP]
        sea_level = np.searchsorted(sea_floors, h, side="right") - 1
        sea_kind = (KIND_SEA_FIRST + np.maximum(sea_level, 0)).astype(np.uint8).ravel()
    return Relief(light.ravel(), kind.ravel(), sea_light, sea_kind)


def detect_aspect(default=DEFAULT_ASPECT):
    """Cell height / cell width, from the terminal's pixel size if it reports it."""
    try:
        import fcntl
        import struct
        import termios
        rows, cols, xpix, ypix = struct.unpack(
            "HHHH", fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8))
        if rows and cols and xpix and ypix:
            return (ypix / rows) / (xpix / cols)
    except Exception:
        pass
    return default


def sun_vector(azimuth_deg, elevation_deg):
    """Unit vector pointing to the sun, in screen space (x right, y up, z toward you)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return (math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))


def solar_position_utc(utc):
    """Return (GMST, solar right ascension, solar declination), in degrees.

    The solar coordinates use the apparent ecliptic longitude and corrected
    obliquity. This compact approximation is accurate enough for a terminal
    globe while accounting for the equation of time and Earth's orbital tilt.
    """
    jd = 2440587.5 + utc.timestamp() / 86400.0
    t = (jd - 2451545.0) / 36525.0

    mean_longitude = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    mean_anomaly = math.radians(
        (357.52911 + t * (35999.05029 - 0.0001537 * t)) % 360.0)
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    equation_center = (
        math.sin(mean_anomaly) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * mean_anomaly) * (0.019993 - 0.000101 * t)
        + math.sin(3 * mean_anomaly) * 0.000289
    )
    true_longitude = mean_longitude + equation_center
    omega = math.radians(125.04 - 1934.136 * t)
    apparent_longitude = math.radians(
        true_longitude - 0.00569 - 0.00478 * math.sin(omega))

    mean_obliquity = 23.0 + (
        26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0
    ) / 60.0
    obliquity = math.radians(mean_obliquity + 0.00256 * math.cos(omega))

    right_ascension = math.degrees(math.atan2(
        math.cos(obliquity) * math.sin(apparent_longitude),
        math.cos(apparent_longitude))) % 360.0
    declination = math.degrees(math.asin(
        math.sin(obliquity) * math.sin(apparent_longitude)))

    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * t * t
        - t * t * t / 38710000.0
    ) % 360.0
    return gmst, right_ascension, declination


def real_sun_angles(right_ascension_deg, declination_deg, tilt_deg):
    """Convert the Sun's celestial coordinates to the globe's screen angles.

    right_ascension_deg is the Sun's RA *relative to the camera's forward
    direction* (the camera is anchored to the Sun so the day side faces you).
    The inverse globe tilt then converts the celestial direction to the
    un-tilted screen coordinates expected by precompute().
    """
    ra = math.radians(right_ascension_deg)
    dec = math.radians(declination_deg)
    tilt = math.radians(tilt_deg)
    sx = math.cos(dec) * math.sin(ra)
    sy_world = math.sin(dec)
    sz_world = math.cos(dec) * math.cos(ra)
    sy = sy_world * math.cos(tilt) - sz_world * math.sin(tilt)
    sz = sy_world * math.sin(tilt) + sz_world * math.cos(tilt)
    azimuth = math.degrees(math.atan2(sx, sz))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sy))))
    return azimuth, elevation


def smoothstep(edge0, edge1, x):
    """0 below edge0, 1 above edge1, smooth S-curve in between (x is an array)."""
    if edge1 <= edge0:
        return np.where(x >= edge0, 1.0, 0.0)
    u = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def ramp_colors(ramp, night_tint, night_mix, twilight_tint, twilight_mix):
    """(night, twilight, day) RGB for every level of a ramp, derived from its day color."""
    def mix(a, b, t):
        return tuple(round(x * (1 - t) + y * t) for x, y in zip(a, b))
    return {name: (mix(day, night_tint, night_mix),
                   mix(day, twilight_tint, twilight_mix), day)
            for name, _, day in ramp}


def land_colors():
    """(night, twilight, day) RGB for every LAND_RAMP level."""
    return ramp_colors(LAND_RAMP, LAND_NIGHT_TINT, LAND_NIGHT_MIX,
                       LAND_TWILIGHT_TINT, LAND_TWILIGHT_MIX)


def sea_colors():
    """(night, twilight, day) RGB for every SEA_RAMP level."""
    return ramp_colors(SEA_RAMP, SEA_NIGHT_TINT, SEA_NIGHT_MIX,
                       SEA_TWILIGHT_TINT, SEA_TWILIGHT_MIX)


def detect_color_depth(default="truecolor"):
    """Best guess of the terminal's color depth: "truecolor", "256" or "16"."""
    import os
    term = os.environ.get("TERM", "")
    if term in TERM_16_COLORS:
        return "16"
    if os.environ.get("COLORTERM", "").lower() in COLORTERM_TRUECOLOR:
        return "truecolor"
    if TERM_256_MARK in term:
        return "256"
    return default


def rgb_to_256(rgb):
    """Nearest xterm 256-color code (color cube or gray ramp) for an RGB color."""
    levels = XTERM_CUBE_LEVELS
    idx = [min(range(len(levels)), key=lambda i: abs(levels[i] - c)) for c in rgb]
    cube = tuple(levels[i] for i in idx)
    cube_code = XTERM_CUBE_FIRST + sum(
        i * len(levels) ** (2 - n) for n, i in enumerate(idx))
    gray_i = min(range(XTERM_GRAY_COUNT), key=lambda i: abs(
        XTERM_GRAY_START + XTERM_GRAY_STEP * i - sum(rgb) / 3))
    gray = (XTERM_GRAY_START + XTERM_GRAY_STEP * gray_i,) * 3
    dist = lambda a: sum((x - y) ** 2 for x, y in zip(a, rgb))
    return cube_code if dist(cube) <= dist(gray) else XTERM_GRAY_FIRST + gray_i


def code_16(code):
    """Escape for a 16-color code 0..15."""
    base = ANSI_FG_NORMAL_BASE if code < 8 else ANSI_FG_BRIGHT_BASE - 8
    return ANSI_FG_16.format(base + code)


def ramp_16(ramp, ramp16, night, twilight, snow=None):
    """(night, twilight, day) 16-color codes for every level of a ramp, by its height."""
    floors = [metres for metres, _ in ramp16]
    out = {}
    for name, metres, _ in ramp:
        if metres is None:      # the snow level
            day = snow
        else:
            day = ramp16[max(np.searchsorted(floors, metres, side="right") - 1, 0)][1]
        out[name] = (night, twilight, day)
    return out


def build_color_table(region_colors, depth="truecolor"):
    """Escape string for every (kind, band) pair, indexed by kind * BANDS + band.

    depth: "truecolor" (RGB from REGION_COLORS and the ramps), "256" (nearest xterm
    color) or "16" (the hand-picked 16-color tables).
    """
    if depth == "16":
        codes = {**REGION_16,
                 **ramp_16(LAND_RAMP, LAND_16_RAMP, LAND_16_NIGHT, LAND_16_TWILIGHT,
                           LAND_16_SNOW),
                 **ramp_16(SEA_RAMP, SEA_16_RAMP, SEA_16_NIGHT, SEA_16_TWILIGHT)}
        make = lambda name, band: code_16(codes[name][band])
    else:
        colors = {**region_colors, **land_colors(), **sea_colors()}
        if depth == "256":
            make = lambda name, band: ANSI_COLOR.format(rgb_to_256(colors[name][band]))
        else:
            make = lambda name, band: ANSI_RGB_COLOR.format(*colors[name][band])
    table = [""] * (len(KIND_NAMES) * BANDS)
    for kind, name in enumerate(KIND_NAMES):
        if kind == 0:
            continue
        for band in range(BANDS):
            table[kind * BANDS + band] = make(name, band)
    for i, (_, _, rgb) in enumerate(STAR_CLASSES):   # index = STAR_CODE_BASE + class
        if depth == "16":
            table.append(code_16(STAR_16[i]))
        elif depth == "256":
            table.append(ANSI_COLOR.format(rgb_to_256(rgb)))
        else:
            table.append(ANSI_RGB_COLOR.format(*rgb))
    return table


class Scene(NamedTuple):
    """Everything about the dots that stays the same from frame to frame.

    All arrays have one entry per dot inside the globe, sorted by terminal cell
    so each cell's dots are contiguous (that is what makes reduceat possible).
    """
    starts: np.ndarray      # index of the first dot of every non-empty cell
    cells: np.ndarray       # flat cell index (row * cols + col) for each of those
    base_lon: np.ndarray    # longitude of each dot before the globe rotates
    row_off: np.ndarray     # map row * MAP_W
    mask: np.ndarray        # Braille bit of the dot inside its cell
    band: np.ndarray        # night / twilight / day
    edge: np.ndarray        # dot belongs to the globe outline
    shade: np.ndarray       # float32 3D shading (globe curvature x day/night)
    land_thr: np.ndarray    # float32 dither threshold for land (already x LAND_DENSITY)
    thr: np.ndarray         # float32 plain dither threshold (used for the sea floor)
    water_kind: np.ndarray  # ... and if it is over water (lat lines already in)
    lon_idx: np.ndarray     # dots that need a per-frame longitude-line test
    lon_tol: np.ndarray     # their longitude-line tolerance


def precompute(cols, rows, tilt_deg, aspect, sun_az, sun_el, day_night,
               sea_relief=False):
    """Per-dot data that doesn't change from frame to frame (None = nothing to draw).

    Units: one terminal cell is 1 wide and `aspect` tall, so a Braille dot is
    DOT_WIDTH wide and aspect/DOTS_Y tall -- the globe stays round on screen.
    """
    W, H = cols * DOTS_X, rows * DOTS_Y
    dot_w, dot_h = DOT_WIDTH, aspect / DOTS_Y
    R = min(W * dot_w, H * dot_h) / 2 - GLOBE_MARGIN_DOTS
    if R <= 0:  # terminal too small for a globe
        return None
    cx, cy = W / 2, H / 2
    t = math.radians(tilt_deg)
    ct, st = math.cos(t), math.sin(t)
    edge_r2 = (1 - EDGE_THICKNESS / R) ** 2
    half_spacing = GRID_SPACING / 2
    sx, sy, sz = sun_vector(sun_az, sun_el)
    # (sea relief fills the sea itself, so the plain ocean sparkle is dropped)
    ocean_density = 0.0 if sea_relief else OCEAN_DAY_DENSITY #if day_night else 0.0

    py, px = np.mgrid[0:H, 0:W]  # every dot, row by row
    y = (cy - py - 0.5) * dot_h / R
    x = (px + 0.5 - cx) * dot_w / R
    r2 = x * x + y * y
    inside = r2 <= 1
    if not inside.any():
        return None
    px, py, x, y, r2 = px[inside], py[inside], x[inside], y[inside], r2[inside]

    z = np.sqrt(1 - r2)
    # tilt the globe's axis toward the viewer
    y2 = y * ct + z * st
    z2 = -y * st + z * ct
    lat = np.degrees(np.arcsin(np.clip(y2, -1.0, 1.0)))
    base_lon = np.degrees(np.arctan2(x, z2))
    row = np.clip(((LAT_MAX - lat) * MAP_PX_PER_DEG).astype(np.intp), 0, MAP_H - 1)
    lat_line = np.abs(((lat + half_spacing) % GRID_SPACING) - half_spacing) \
        < GRID_LINE_HALF_WIDTH
    lon_tol = GRID_LINE_HALF_WIDTH / np.maximum(
        np.cos(np.radians(lat)), GRID_MIN_COS_LAT)
    thr = (BAYER_ARRAY[py % BAYER_SIZE, px % BAYER_SIZE] + 0.5) / BAYER_LEVELS
    shade = SHADE_MIN + (1 - SHADE_MIN) * z ** SHADE_GAMMA
    band = np.full(x.shape, BAND_DAY, dtype=np.uint8)
    if day_night:
        # (x, y, z) is also the surface normal; dot with sun = sunlight
        light = x * sx + y * sy + z * sz
        day = smoothstep(-TERMINATOR_SOFTNESS, TERMINATOR_SOFTNESS, light)
        diffuse = DAY_AMBIENT + (1 - DAY_AMBIENT) * np.maximum(light, 0.0)
        shade = shade * (NIGHT_LEVEL + (1 - NIGHT_LEVEL) * day * diffuse)
        band = np.where(day < TWILIGHT_RANGE[0], BAND_NIGHT,
                        np.where(day < TWILIGHT_RANGE[1], BAND_TWILIGHT, BAND_DAY)
                        ).astype(np.uint8)

    # Decide as much as possible now. Per dot, the first rule that applies wins:
    # outline > land > sunlit ocean > grid line. (Land itself is decided per frame
    # in render(): it depends on the hillshade of the map pixel under the dot.)
    edge = r2 > edge_r2
    ocean_hit = thr < shade * ocean_density
    grid_ok = (z > GRID_MIN_Z) & (thr < GRID_DENSITY) & ~ocean_hit
    water_kind = np.where(edge, KIND_EDGE, np.where(
        ocean_hit, KIND_OCEAN, np.where(grid_ok & lat_line, KIND_GRID, 0)))
    needs_lon_test = grid_ok & ~lat_line & ~edge  # longitude lines move with rot

    cell = (py // DOTS_Y) * cols + px // DOTS_X
    order = np.argsort(cell, kind="stable")
    cell = cell[order]
    starts = np.flatnonzero(np.concatenate(([True], cell[1:] != cell[:-1])))
    lon_idx = np.flatnonzero(needs_lon_test[order])
    return Scene(
        starts=starts, cells=cell[starts],
        base_lon=base_lon[order], row_off=(row * MAP_W)[order],
        mask=BRAILLE_BIT_TABLE[px % DOTS_X, py % DOTS_Y][order],
        band=band[order],
        edge=edge[order],
        shade=shade.astype(np.float32)[order],
        land_thr=(thr * LAND_DENSITY).astype(np.float32)[order],
        thr=thr.astype(np.float32)[order],
        water_kind=water_kind.astype(np.uint8)[order],
        lon_idx=lon_idx, lon_tol=lon_tol[order][lon_idx])


def cells_to_text(bits, codes, cols, rows, color_table):
    """Turn per-cell Braille bits (and color codes) into the frame's text."""
    grid = np.empty((rows, cols + 1), dtype="<u4")
    grid[:, :cols] = CELL_CODEPOINTS[bits].reshape(rows, cols)
    grid[:, cols] = NEWLINE_CODEPOINT
    text = grid.tobytes().decode("utf-32-le")[:-1]  # drop the last newline
    if color_table is None:
        return text
    filled = np.flatnonzero(bits)
    if filled.size == 0:
        return text
    c = codes[filled]
    change = np.concatenate(([True], c[1:] != c[:-1]))  # color differs from previous
    at = filled[change]
    positions = (at + at // cols).tolist()  # cell index -> text index (newlines)
    parts, prev = [], 0
    for pos, code in zip(positions, c[change].tolist()):  # only a few hundred
        parts.append(text[prev:pos])
        parts.append(color_table[code])
        prev = pos
    parts.append(text[prev:])
    return "".join(parts)


def make_stars(cols, rows, aspect):
    """Static star field as (bits, codes) per cell, the same layout render_cells returns.

    Each star sits on a random Braille dot. Its class sets which dots it lights up and
    its color; where stars overlap, the brighter class wins. Stars are only placed
    outside the globe's disc (same geometry as precompute), so the ocean never gets
    stars. Built once per terminal size.
    """
    rng = np.random.default_rng(STAR_SEED)
    W, H = cols * DOTS_X, rows * DOTS_Y
    n = int(cols * rows * STAR_DENSITY)
    px = rng.integers(0, W, n)
    py = rng.integers(0, H, n)
    weights = np.array([share for share, _, _ in STAR_CLASSES], dtype=float)
    klass = rng.choice(len(STAR_CLASSES), size=n, p=weights / weights.sum())
    dots = np.zeros((H, W), dtype=np.uint8)          # 0 = no star, else class + 1
    for k, (_, offsets, _) in enumerate(STAR_CLASSES):
        sel = klass == k
        for dx, dy in offsets:
            x, y = px[sel] + dx, py[sel] + dy
            ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            np.maximum.at(dots, (y[ok], x[ok]), k + 1)
    # drop every star dot on or near the globe (same disc as precompute)
    dot_w, dot_h = DOT_WIDTH, aspect / DOTS_Y
    R = min(W * dot_w, H * dot_h) / 2 - GLOBE_MARGIN_DOTS
    if R > 0:
        gy, gx = np.mgrid[0:H, 0:W]
        gx = (gx + 0.5 - W / 2) * dot_w / R
        gy = (H / 2 - gy - 0.5) * dot_h / R
        outside = gx * gx + gy * gy > (1 + STAR_GLOBE_MARGIN) ** 2
        dots[~outside] = 0
    # group dots by cell: [rows, cols, DOTS_Y, DOTS_X]
    per_dot = dots.reshape(rows, DOTS_Y, cols, DOTS_X).transpose(0, 2, 1, 3)
    lit = (per_dot > 0).astype(np.uint8) * BRAILLE_BIT_TABLE.T   # [dot_y, dot_x] bits
    bits = np.bitwise_or.reduce(lit, axis=(2, 3))
    top = per_dot.max(axis=(2, 3))                   # brightest class in the cell
    codes = np.where(top > 0, STAR_CODE_BASE + top - 1, 0)
    return bits.ravel().astype(np.uint8), codes.ravel().astype(np.uint8)


def render_cells(scene, land, relief, cols, rows, rot):
    """Per-cell Braille bits and color codes for one frame (flat arrays)."""
    half_spacing = GRID_SPACING / 2
    lon = scene.base_lon - rot
    column = np.floor((lon - LON_MIN) * MAP_PX_PER_DEG).astype(np.intp) % MAP_W
    pixel = scene.row_off + column          # map pixel under every dot
    over_land = land[pixel]

    water = scene.water_kind.copy()
    idx = scene.lon_idx
    if idx.size:  # longitude lines
        hit = np.abs(((lon[idx] + half_spacing) % GRID_SPACING) - half_spacing) \
            < scene.lon_tol
        water[idx[hit]] = KIND_GRID
    if relief.sea_kind is not None:  # sea floor: hillshade sets density, depth sets color
        sea_on = scene.shade * relief.sea_light[pixel] > scene.thr
        sea = np.where(sea_on, relief.sea_kind[pixel], 0)
        water = np.where(water == 0, sea, water)  # outline and grid keep priority
    # Land: the hillshade of this map pixel sets how many dots light up (the same
    # dither as the globe's edge shading); the height tier sets the color.
    land_on = scene.edge | (scene.shade * relief.light[pixel] > scene.land_thr)
    land_kind = np.where(scene.edge, KIND_EDGE, relief.kind[pixel])
    kind = np.where(over_land, np.where(land_on, land_kind, 0), water)

    # Per cell: OR of the drawn dots' bits, and the highest-priority color code.
    # (A drawn dot's code is >= BANDS, an undrawn dot's is < BANDS, so drawn wins.)
    code = kind * BANDS + scene.band
    bits = np.zeros(rows * cols, dtype=np.uint8)
    codes = np.zeros(rows * cols, dtype=np.uint8)
    bits[scene.cells] = np.bitwise_or.reduceat(scene.mask * (kind > 0), scene.starts)
    codes[scene.cells] = np.maximum.reduceat(code, scene.starts)
    return bits, codes

def render(scene, land, relief, cols, rows, rot, color_table=None, stars=None):
    """Draw one frame and return it as text.

    color_table: None = plain Braille (the caller sets one global color), or the
    list from build_color_table() to color every cell individually.
    stars: (bits, codes) from make_stars(), or None for no stars. Stars only fill
    cells the globe leaves empty.
    """
    if scene is None:
        bits = np.zeros(rows * cols, dtype=np.uint8)
        codes = np.zeros(rows * cols, dtype=np.uint8)
    else:
        bits, codes = render_cells(scene, land, relief, cols, rows, rot)
    if stars is not None:
        star_bits, star_codes = stars
        empty = bits == 0
        bits = np.where(empty, star_bits, bits)
        codes = np.where(empty, star_codes, codes)
    return cells_to_text(bits, codes, cols, rows, color_table)


def resolve_options(args):
    """Final options: the settings' defaults, then the preset, then single command-line options."""
    opts = {"relief": "full" if DEFAULT_SEA_RELIEF else "land", "grid": DEFAULT_GRID,
            "stars": STARS, "night": DAY_NIGHT, "sparkle": OCEAN_DAY_DENSITY,
            "color": DEFAULT_COLOR}
    opts.update(PRESETS[args.preset])
    relief = args.relief
    if relief is None:      # the older flags still work
        if args.no_relief:
            relief = "none"
        elif args.sea_relief is True:
            relief = "full"
        elif args.sea_relief is False and opts["relief"] == "full":
            relief = "land"
    for key, value in (("relief", relief), ("grid", args.grid), ("stars", args.stars),
                       ("night", args.night), ("sparkle", args.ocean_sparkle),
                       ("color", args.color)):
        if value is not None:
            opts[key] = value
    return opts


def main():
    global GRID_DENSITY, GRID_SPACING, STAR_DENSITY, OCEAN_DAY_DENSITY
    preset_help = "; ".join(f"{name}: " + ", ".join(f"{k}={v}" for k, v in cfg.items())
                            for name, cfg in PRESETS.items() if cfg)
    ap = argparse.ArgumentParser(description="Rotating Braille globe",
                                 epilog="presets -- " + preset_help)
    ap.add_argument("--preset", choices=tuple(PRESETS), default=DEFAULT_PRESET,
                    help="a bundle of options (default: %(default)s); single options "
                         "below override it")
    ap.add_argument("--grid", action=argparse.BooleanOptionalAction, default=None,
                    help="latitude / longitude lines on or off")
    ap.add_argument("--grid-spacing", type=float, default=GRID_SPACING,
                    help="degrees between grid lines")
    ap.add_argument("--stars", action=argparse.BooleanOptionalAction, default=None,
                    help="background stars on or off")
    ap.add_argument("--star-density", type=float, default=STAR_DENSITY,
                    help="stars per terminal cell")
    ap.add_argument("--ocean-sparkle", type=float, default=None,
                    help="dot density 0..1 of the plain sunlit sea (ignored with full relief)")
    ap.add_argument("--night", action=argparse.BooleanOptionalAction, default=None,
                    help="day/night lighting on or off")
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS,
                    help="frames per second")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED,
                    help="rotation speed in degrees per second (ignored with --real)")
    ap.add_argument("--tilt", type=float, default=DEFAULT_TILT,
                    help="visual axial tilt, degrees")
    ap.add_argument("--color", type=int, default=DEFAULT_COLOR,
                    help="one ANSI 256 color for everything (-1 = no color); "
                         "default: per-region colors")
    ap.add_argument("--color-depth", choices=COLOR_DEPTHS, default=DEFAULT_COLOR_DEPTH,
                    help="how many colors the terminal can show; auto = guess from "
                         "TERM/COLORTERM (the Linux text console gets 16 colors)")
    ap.add_argument("--aspect", type=float, default=None,
                    help=f"cell height/width (default: auto-detect, fallback "
                         f"{DEFAULT_ASPECT}). Increase if the globe looks too "
                         f"narrow, decrease if too wide.")
    ap.add_argument("--sun-azimuth", type=float, default=SUN_AZIMUTH,
                    help="sun direction: degrees to the right (+) or left (-); with --real "
                         "only the sideways offset is used")
    ap.add_argument("--sun-elevation", type=float, default=SUN_ELEVATION,
                    help="sun direction: degrees above (+) or below (-) the equator (ignored with --real)")
    ap.add_argument("--relief", default=None, metavar="FILE|MODE",
                    help=f"height map file, 720x360 int16 (default: {RELIEF_FILE} next to "
                         f"the script); or a mode: none | land | full")
    ap.add_argument("--no-relief", action="store_true",
                    help="same as --relief none: flat land, no hillshade, no height colors")
    ap.add_argument("--sea-relief", action=argparse.BooleanOptionalAction, default=None,
                    help="same as --relief full: sea floor relief too (hillshade + depth "
                         "colors); needs the height map")
    ap.add_argument("--real", action="store_true",
                    help="real time: UTC-based Earth rotation and real sun declination, "
                         "viewed from the day side")
    args = ap.parse_args()
    # --relief is a mode word (none | land | full) or, as before, a height-map file
    relief_file = None
    if args.relief in RELIEF_MODES:
        mode = args.relief
    else:
        mode, relief_file = None, args.relief
    args.relief = mode
    opts = resolve_options(args)
    GRID_DENSITY = GRID_DENSITY_ON if opts["grid"] else 0
    GRID_SPACING = args.grid_spacing
    STAR_DENSITY = args.star_density
    OCEAN_DAY_DENSITY = opts["sparkle"]
    use_stars = opts["stars"]
    day_night = opts["night"]
    color = opts["color"]
    depth = detect_color_depth() if args.color_depth == "auto" else args.color_depth
    color_table = build_color_table(REGION_COLORS, depth) if color is None else None

    land = load_land()
    elev = None
    if opts["relief"] != "none":
        relief_path = Path(relief_file) if relief_file else Path(__file__).with_name(RELIEF_FILE)
        elev = load_elevation(relief_path)
        if elev is None:
            print(f"note: {relief_path} not found, drawing flat land", file=sys.stderr)
    sea_relief = opts["relief"] == "full" and elev is not None
    if opts["relief"] == "full" and elev is None:
        print("note: sea relief needs the height map, drawing flat sea", file=sys.stderr)
    relief = build_relief(elev, sea_relief)  # once: hillshade for the whole map
    aspect = args.aspect if args.aspect else detect_aspect()
    size = None
    scene = None
    stars = None
    cols = rows = 0

    sys.stdout.write(ANSI_ENTER_SCREEN)
    if color is not None and color >= 0:
        sys.stdout.write(ANSI_COLOR.format(color))
    rot = 0.0
    sun_azimuth, sun_elevation = args.sun_azimuth, args.sun_elevation
    next_sun_update = 0.0
    last = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            dt, last = now - last, now
            refresh_sun = False
            if args.real:
                utc_now = datetime.now(timezone.utc)
                gmst, sun_ra, sun_dec = solar_position_utc(utc_now)
                # Camera anchored to the Sun's right ascension (offset by
                # --sun-azimuth), so the sunlit hemisphere always faces the viewer.
                # Earth still turns once per solar day beneath the fixed sun.
                view_ra = sun_ra - args.sun_azimuth
                rot = (gmst - view_ra) % FULL_TURN
                if day_night and now >= next_sun_update:
                    sun_azimuth, sun_elevation = real_sun_angles(
                        args.sun_azimuth, sun_dec, args.tilt)
                    next_sun_update = now + REAL_SUN_UPDATE_SECONDS
                    refresh_sun = True
            else:
                rot = (rot + args.speed * dt) % FULL_TURN

            new_size = shutil.get_terminal_size(FALLBACK_TERMINAL_SIZE)
            resized = new_size != size
            if resized or refresh_sun:  # (re)build on resize / solar update
                size = new_size
                cols, rows = size.columns, max(1, size.lines - RESERVED_ROWS)
                if resized and not args.aspect:
                    aspect = detect_aspect()
                scene = precompute(cols, rows, args.tilt, aspect,
                                   sun_azimuth, sun_elevation, day_night, sea_relief)
                if resized:
                    stars = make_stars(cols, rows, aspect) if use_stars else None
                    sys.stdout.write(ANSI_CLEAR)

            frame = render(scene, land, relief, cols, rows, rot, color_table, stars)
            sys.stdout.write(ANSI_HOME + frame)
            sys.stdout.flush()

            sleep = 1.0 / args.fps - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(ANSI_LEAVE_SCREEN)
        sys.stdout.flush()


if __name__ == "__main__":
    main()