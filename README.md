# Braille TTY Screensaver

Configuration guide for running a Python screensaver that renders animations using Unicode Braille characters directly in a Linux virtual console (TTY).

## Overview

The Linux TTY may not display Unicode Braille characters correctly with its default console font. This configuration enables Braille rendering while retaining regular Latin characters.

The setup uses two font files:

- `Lat15-Fixed16.psf.gz` — regular Latin characters.
- `brl-16x8.psf` — Braille characters.

## Requirements

Install the console font utilities:

```bash
sudo apt update
sudo apt install console-braille
```

Verify that the required fonts exist:

```bash
ls /usr/share/consolefonts/Lat15-Fixed16.psf.gz
ls /usr/share/consolefonts/brl-16x8.psf
```

## Configuration

Edit `/etc/default/console-setup`:

```bash
sudo nano /etc/default/console-setup
```

Set the following options:

```ini
ACTIVE_CONSOLES="/dev/tty[1-6]"
CHARMAP="UTF-8"
CODESET="guess"
FONTFACE="Fixed"
FONTSIZE="16x8"
FONT="Lat15-Fixed16.psf.gz brl-16x8.psf"
```

**Important:** Keep both font filenames in the `FONT` setting. This configuration was verified to work for displaying both regular text and Braille characters.

Apply the configuration:

```bash
sudo setupcon
```

The configuration is persistent across reboots.



## Topo info 
Data was taken from https://www.ncei.noaa.gov/products/etopo-global-relief-model 

60 Arc-Second Resolution: Ice surface_elevation geotiff
and converted to lower resolution by script make_relief.py

`make_relief.py ETOPO_2022_v1_60s_N90W180_surface.tif relief.bin`

## Run the Python Screensaver

Run your animation script:

```bash
python3 globe_v3_real.py
```

python3 globe_v3_real.py --help
usage: globe_v3_real.py [-h] [--fps FPS] [--speed SPEED] [--tilt TILT] [--color COLOR] [--aspect ASPECT]
                        [--sun-azimuth SUN_AZIMUTH] [--sun-elevation SUN_ELEVATION] [--no-night] [--real]

Rotating Braille globe

options:
  -h, --help            show this help message and exit
  --fps FPS             frames per second
  --speed SPEED         rotation speed in degrees per second (ignored with --real)
  --tilt TILT           visual axial tilt, degrees
  --color COLOR         one ANSI 256 color for everything (-1 = no color); default: per-region colors
  --aspect ASPECT       cell height/width (default: auto-detect, fallback 2.2). Increase if the globe looks too narrow,
                        decrease if too wide.
  --sun-azimuth SUN_AZIMUTH
                        sun direction: degrees to the right (+) or left (-; ignored with --real)
  --sun-elevation SUN_ELEVATION
                        sun direction: degrees above (+) or below (-) the equator (ignored with --real)
  --no-night            turn off day/night lighting
  --real                use UTC for sidereal Earth rotation and astronomical sunlight

