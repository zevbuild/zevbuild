# Zevbuild

Zevbuild is an independent software studio based in Goa, India. This repository contains the studio website, several web tools, a Python data dashboard, and Cloudflare Pages API functions.

- **Website:** [zevbuild.github.io](https://zevbuild.github.io)
- **GitHub organization:** [github.com/zevbuild](https://github.com/zevbuild)
- **Contact:** [zevbuildstudio@gmail.com](mailto:zevbuildstudio@gmail.com)

## What's in this repository?

| Path | Description |
| --- | --- |
| [`index.html`](index.html) | Main Zevbuild website |
| [`tools/index.html`](tools/index.html) | Index of the tools included in this repository |
| [`tools/satta-matka-tools/`](tools/satta-matka-tools/) | Python CLI, local web dashboard, and historical-data analysis |
| [`tools/sp-ms-downloader/index.html`](tools/sp-ms-downloader/index.html) | Local audio player |
| [`tools/top-10-free-video-downloaders-in-india/`](tools/top-10-free-video-downloaders-in-india/) | Static article page |
| [`functions/api/`](functions/api/) | Cloudflare Pages API functions |
| [`404.html`](404.html), [`robots.txt`](robots.txt), [`sitemap.xml`](sitemap.xml) | Site and search-engine support files |

The ZevSafe, ZevSync, and CollegeBus links on the main website point to their separate projects; their source code is not included in this repository.

## Run the website locally

From the repository root, start a static HTTP server:

```bash
python -m http.server 8000
```

Then open [http://localhost:8000](http://localhost:8000).

## Run the data dashboard locally

The dashboard and CLI live in `tools/satta-matka-tools/`. Install its Python dependencies and start the local server:

```bash
cd tools/satta-matka-tools
python -m pip install -r requirements.txt
python main.py --web
```

The dashboard is served at [http://127.0.0.1:8080](http://127.0.0.1:8080). For other CLI options and project details, see [`tools/satta-matka-tools/README.md`](tools/satta-matka-tools/README.md).
