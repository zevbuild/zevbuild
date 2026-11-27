# Zevbuild

Zevbuild is an independent software studio based in Goa, India. This repository contains the studio portal, privacy-first client-side web tools, an offline RetroWave audio workstation, Cloudflare Pages edge functions, and SEO/PWA infrastructure.

- **Website:** [zevbuild.pages.dev](https://zevbuild.pages.dev)
- **GitHub organization:** [github.com/zevbuild](https://github.com/zevbuild)
- **Contact:** [zevbuildstudio@gmail.com](mailto:zevbuildstudio@gmail.com)

## What's in this repository?

| Path | Description |
| --- | --- |
| [`index.html`](index.html) | Main Zevbuild website — dark-themed, privacy-first company portal and product showcase |
| [`tools/index.html`](tools/index.html) | Embedded tools catalog and searchable navigation hub |
| [`tools/upi-link-generator/`](tools/upi-link-generator/) | Browser-based UPI payment link generator & QR code creator for Indian payments |
| [`tools/spotify/`](tools/spotify/) | Spotify playlist batch-downloader guide and offline RetroWave HTML5 audio player (drag-and-drop local MP3/WAV/OGG/M4A, Web Audio API synthwave Canvas visualizer) |
| [`tools/matka/`](tools/matka/) | Lightweight 301/meta-refresh redirect stubs forwarding legacy Kalyan Matka URLs to standalone platform [kalyanhub.pages.dev](https://kalyanhub.pages.dev/) |
| [`donation/`](donation/) | Studio voluntary UPI contribution page |
| [`t/privacy/`](t/privacy/) & [`t/terms/`](t/terms/) | Official Privacy Policy (Zero-Telemetry / Zero-Knowledge) and Terms of Service |
| [`functions/api/`](functions/api/) | Cloudflare Pages serverless edge functions (`live-kalyan.js`, `fetch-and-predict.js`) |
| [`verify_links_and_assets.py`](verify_links_and_assets.py) | Automated 4-tier link, DOM anchor, and static asset verification crawler harness |
| [`404.html`](404.html), [`robots.txt`](robots.txt), [`sitemap.xml`](sitemap.xml), [`_redirects`](_redirects) | Smart-routing 404 handler, crawler rules, SEO sitemap, and Cloudflare Pages 301 edge redirect map |

## Ecosystem & External Platforms

- **ZevSafe** ([zevsafe.pages.dev](https://zevsafe.pages.dev)) — Zero-knowledge cryptographic folder vault (`AES-256-GCM` & `PBKDF2-SHA512`).
- **ZevBlock** ([github.com/zevbuild/zevblock](https://github.com/zevbuild/zevblock)) — Hardened Android accountability OS & local DNS sinkhole.
- **CollegeBus** ([collegebus.pages.dev](https://collegebus.pages.dev)) — Real-time civic transit scheduler & offline PWA for South Goa routes.
- **Kalyan Hub** ([kalyanhub.pages.dev](https://kalyanhub.pages.dev)) — Dedicated standalone quantitative analytics & multi-session predictor platform.
- **Sant Mistri Carpenter** ([santmistricarpenter.pages.dev](https://santmistricarpenter.pages.dev)) — Production woodworking & SQFT cost estimation web platform built by Zevbuild.

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

For full outbound external link auditing:

```bash
python verify_links_and_assets.py --check-external
```
