# Zevbuild

Zevbuild is an independent software studio based in Goa, India. This repository contains the studio website, privacy-first web tools, an offline RetroWave audio player, a Python machine learning and predictive analytics engine, and Cloudflare Pages API functions.

- **Website:** [zevbuild.pages.dev](https://zevbuild.pages.dev)
- **GitHub organization:** [github.com/zevbuild](https://github.com/zevbuild)
- **Contact:** [zevbuildstudio@gmail.com](mailto:zevbuildstudio@gmail.com)

## What's in this repository?

| Path | Description |
| --- | --- |
| [`index.html`](index.html) | Main Zevbuild website — dark-themed, privacy-first company portal and product showcase |
| [`tools/index.html`](tools/index.html) | Embedded tools catalog and navigation hub |
| [`tools/matka/`](tools/matka/) | Kalyan Matka ML & statistical predictive analytics engine (Markov chains, recency weighting, seasonal decomposition, walk-forward backtesting, and local web dashboard) |
| [`tools/spotify/`](tools/spotify/) | Spotify playlist batch-downloader guide and offline RetroWave HTML5 audio player (drag-and-drop local MP3/WAV/OGG/M4A, Web Audio API synthwave Canvas visualizer) |
| [`tools/download/`](tools/download/) | Curated guide to legitimate, privacy-first, and open-source video downloaders and media transcoding utilities (yt-dlp, VLC, HandBrake, Cobalt, etc.) |
| [`tools/yt/`](tools/yt/) | Client-side YouTube video and audio fetcher (MP4 720p/1080p, MP3) powered by yt-dlp via Cloudflare Pages edge functions |
| [`tools/upi-link-generator/`](tools/upi-link-generator/) | Browser-based UPI payment link generator & QR code creator for Indian payments |
| [`functions/api/`](functions/api/) | Cloudflare Pages serverless edge functions (`yt-download.js`, `live-kalyan.js`, `fetch-and-predict.js`) |
| [`verify_links_and_assets.py`](verify_links_and_assets.py) | Automated link, DOM anchor, and static asset verification crawler harness (zero external dependencies) |
| [`404.html`](404.html), [`robots.txt`](robots.txt), [`sitemap.xml`](sitemap.xml) | Smart-routing 404 error handler, crawler rules, and SEO sitemap |

The ZevSafe, ZevSync, and CollegeBus links on the main website point to their separate open-source repositories under the [Zevbuild organization](https://github.com/zevbuild); their source code is managed in dedicated repositories.

## Run the website locally

From the repository root, start a static HTTP server:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## Run the verification crawler locally

Verify all internal links, anchors, entry points, and static assets across the repository:

```bash
python verify_links_and_assets.py --local-only
```

For full outbound external link checking:

```bash
python verify_links_and_assets.py --check-external
```

## Run the data dashboard locally

The Kalyan predictive analytics dashboard and CLI live in `tools/matka/`. Install its Python dependencies and start the local server:

```bash
cd tools/matka
python -m pip install -r requirements.txt
python main.py --web
```

The dashboard is served at [http://127.0.0.1:8080](http://127.0.0.1:8080). For CLI options, statistical models, and dataset schema, see [`tools/matka/README.md`](tools/matka/README.md).
