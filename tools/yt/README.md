# 🎬 v_yt — YouTube Video Downloader

> **Part of the [Zevbuild](https://zevbuild.pages.dev) tools portfolio.**

A browser-based YouTube video downloader powered by **yt-dlp** via the Cobalt API. Supports MP4 (720p / 1080p) and MP3 audio-only downloads with zero installs and zero trackers.

---

## ✨ Features

| Feature | Details |
|---|---|
| **MP4 720p** | Standard HD video download |
| **MP4 1080p** | Full HD video download |
| **MP3 Audio** | Audio-only extraction |
| **yt-dlp Engine** | Extraction powered by Cobalt (open-source yt-dlp wrapper) |
| **Zero Trackers** | No analytics, no cookies, no telemetry |
| **No Install** | Works entirely in your browser |
| **YouTube Shorts** | Supports `/shorts/` URLs |
| **Error Handling** | Clear messages for private/age-restricted/unavailable videos |

---

## 🚀 Usage

1. Open [`tools/yt/index.html`](index.html) in your browser
2. Paste a YouTube video URL (e.g. `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
3. Select your desired format (MP4 720p / MP4 1080p / MP3 Audio)
4. Click **Fetch Download Link**
5. Click the **Download** button to save the file

---

## 🏗️ Architecture

```
Browser (index.html)
        │
        │  GET /api/yt-download?url=...&format=...
        ▼
Cloudflare Worker (functions/api/yt-download.js)
        │
        │  POST https://api.cobalt.tools/
        ▼
   Cobalt API (yt-dlp wrapper)
        │
        │  { status: "redirect", url: "..." }
        ▼
  Direct CDN download link  ──► returned to browser  ──► user downloads file
```

**Key components:**

| File | Role |
|---|---|
| `tools/yt/index.html` | Static frontend UI (vanilla HTML + Tailwind + vanilla JS) |
| `functions/api/yt-download.js` | Cloudflare Pages Function — proxies to Cobalt API |

**Why a backend worker?** `yt-dlp` is a Python CLI tool — it cannot run in a browser. The Cloudflare Worker acts as a server-side proxy that calls Cobalt (an open-source yt-dlp web service) and returns direct download URLs to the browser.

---

## 📡 API Reference

### `GET /api/yt-download`

| Parameter | Required | Values | Default |
|---|---|---|---|
| `url` | ✅ | Any valid YouTube URL | — |
| `format` | ❌ | `mp4-720`, `mp4-1080`, `mp3` | `mp4-720` |

**Success Response:**
```json
{
  "status": "ok",
  "format": "mp4-720",
  "downloads": [
    {
      "label": "MP4 720p",
      "url": "https://rr1---sn-....googlevideo.com/...",
      "filename": "Video Title.mp4"
    }
  ]
}
```

**Error Response:**
```json
{
  "error": "This video is private.",
  "cobaltCode": "content.video.private"
}
```

**Supported URL formats:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/shorts/VIDEO_ID`
- `https://www.youtube.com/live/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

---

## ⚖️ Legal Notice

> **For personal and fair-use purposes only.**
>
> Downloading YouTube videos may violate [YouTube's Terms of Service](https://www.youtube.com/static?template=terms). Do not download copyrighted content without explicit authorization from the rights holder.
>
> Zevbuild is not responsible for misuse of this tool.

---

## 🤝 Credits

- **[Cobalt](https://github.com/imputnet/cobalt)** — Open-source media downloader service (MIT License)
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** — The underlying extraction engine (Unlicense)

---

## 📜 License

MIT — Copyright © 2026 Zevbuild Studio
