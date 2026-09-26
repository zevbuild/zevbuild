/**
 * functions/api/yt-download.js
 * Cloudflare Pages Function — YouTube download link extractor
 *
 * Proxies requests to the Cobalt API (https://cobalt.tools) which is powered
 * by yt-dlp under the hood. Returns direct download URLs for YouTube videos.
 *
 * Route: GET /api/yt-download?url=<encoded-youtube-url>&format=<format>
 *
 * Supported formats:
 *   mp4-720  → MP4 video, up to 720p
 *   mp4-1080 → MP4 video, up to 1080p
 *   mp3      → MP3 audio only
 *
 * Cobalt API docs: https://github.com/imputnet/cobalt/blob/main/docs/api.md
 */

// Cobalt API endpoint — open source yt-dlp web service
const COBALT_API = 'https://api.cobalt.tools/';

// Format code → Cobalt API parameter mapping
const FORMAT_MAP = {
  'mp4-720': {
    downloadMode: 'auto',
    videoQuality: '720',
    label: 'MP4 720p',
  },
  'mp4-1080': {
    downloadMode: 'auto',
    videoQuality: '1080',
    label: 'MP4 1080p',
  },
  'mp3': {
    downloadMode: 'audio',
    audioFormat: 'mp3',
    label: 'MP3 Audio',
  },
};

// Standard CORS headers — same pattern as live-kalyan.js
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

/**
 * Handle incoming requests
 * @param {EventContext} context - Cloudflare Pages Function context
 */
export async function onRequest(context) {
  const { request } = context;

  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  // Only allow GET requests
  if (request.method !== 'GET') {
    return jsonResponse({ error: 'Method not allowed. Use GET.' }, 405);
  }

  const { searchParams } = new URL(request.url);
  const videoUrl = searchParams.get('url');
  const formatCode = searchParams.get('format') || 'mp4-720';

  // ── Validate inputs ──────────────────────────────────────────────────────

  if (!videoUrl) {
    return jsonResponse({ error: 'Missing required parameter: url' }, 400);
  }

  // Validate that it is a plausible YouTube URL before hitting Cobalt
  if (!isValidYouTubeUrl(videoUrl)) {
    return jsonResponse({
      error: 'Invalid URL. Only YouTube video URLs are supported (youtube.com/watch?v=..., youtu.be/..., YouTube Shorts, YouTube Live).'
    }, 400);
  }

  const formatConfig = FORMAT_MAP[formatCode];
  if (!formatConfig) {
    return jsonResponse({
      error: `Unsupported format: "${formatCode}". Use one of: mp4-720, mp4-1080, mp3.`
    }, 400);
  }

  // ── Build Cobalt API request ─────────────────────────────────────────────

  const cobaltPayload = {
    url: videoUrl,
    downloadMode: formatConfig.downloadMode,
    filenameStyle: 'pretty',
    videoQuality: formatConfig.videoQuality,
    audioFormat: formatConfig.audioFormat,
    // Disable TikTok-specific options to keep responses lean
    tiktokFullAudio: false,
    disableMetadata: false,
  };

  // Remove undefined keys
  Object.keys(cobaltPayload).forEach(key => {
    if (cobaltPayload[key] === undefined) delete cobaltPayload[key];
  });

  // ── Call Cobalt API ──────────────────────────────────────────────────────

  let cobaltRes;
  try {
    cobaltRes = await fetch(COBALT_API, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        // Cobalt requires a proper User-Agent
        'User-Agent': 'Zevbuild/v_yt (+https://zevbuild.github.io/tools/v_yt/)',
      },
      body: JSON.stringify(cobaltPayload),
    });
  } catch (networkErr) {
    return jsonResponse({
      error: 'Could not reach the extraction service. Please try again in a few seconds.',
      detail: networkErr.message,
    }, 502);
  }

  // ── Parse Cobalt response ────────────────────────────────────────────────

  let cobaltData;
  try {
    cobaltData = await cobaltRes.json();
  } catch {
    return jsonResponse({
      error: 'Extraction service returned an unreadable response. Please try again.',
    }, 502);
  }

  // Cobalt response status values:
  // "tunnel"  → direct stream link (best)
  // "redirect" → direct file link
  // "picker"  → multiple streams (playlists/multi-stream)
  // "error"   → extraction failed

  if (cobaltData.status === 'error') {
    const cobaltErr = cobaltData.error?.code || 'unknown';
    return jsonResponse({
      error: mapCobaltError(cobaltErr),
      cobaltCode: cobaltErr,
    }, 422);
  }

  const downloads = [];

  if (cobaltData.status === 'redirect' || cobaltData.status === 'tunnel') {
    // Single download link
    if (cobaltData.url) {
      downloads.push({
        label: formatConfig.label,
        url: cobaltData.url,
        filename: cobaltData.filename || null,
      });
    }
  } else if (cobaltData.status === 'picker') {
    // Multiple streams (rare for YouTube — mainly for playlists)
    if (Array.isArray(cobaltData.picker)) {
      cobaltData.picker.slice(0, 4).forEach((item, idx) => {
        downloads.push({
          label: `${formatConfig.label} (Stream ${idx + 1})`,
          url: item.url,
          filename: item.filename || null,
        });
      });
    }
    // Also add audio if provided
    if (cobaltData.audio) {
      downloads.push({
        label: 'MP3 Audio',
        url: cobaltData.audio,
        filename: null,
      });
    }
  }

  if (downloads.length === 0) {
    return jsonResponse({
      error: 'No download links were returned. The video may be private, age-restricted, or geo-blocked.',
    }, 422);
  }

  // ── Return success response ──────────────────────────────────────────────

  return jsonResponse({
    status: 'ok',
    format: formatCode,
    downloads,
  }, 200);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Validate that a URL is a recognisable YouTube URL.
 * Cobalt also validates, but we check early to avoid unnecessary API calls.
 * @param {string} url
 * @returns {boolean}
 */
function isValidYouTubeUrl(url) {
  try {
    const u = new URL(url);
    const host = u.hostname.replace(/^www\./, '');

    if (host === 'youtube.com') {
      // Standard watch URL
      if (u.pathname === '/watch' && u.searchParams.has('v')) return true;
      // Shorts
      if (u.pathname.startsWith('/shorts/')) return true;
      // Live
      if (u.pathname.startsWith('/live/')) return true;
      // Embed
      if (u.pathname.startsWith('/embed/')) return true;
    }

    if (host === 'youtu.be' && u.pathname.length > 1) return true;

    return false;
  } catch {
    return false;
  }
}

/**
 * Map Cobalt API error codes to human-readable messages.
 * @param {string} code - Cobalt error code
 * @returns {string}
 */
function mapCobaltError(code) {
  const errorMessages = {
    'content.video.unavailable': 'This video is unavailable (may be private or deleted).',
    'content.video.age': 'This video is age-restricted and cannot be downloaded.',
    'content.video.private': 'This video is private.',
    'content.video.live': 'Live streams cannot be downloaded in real time.',
    'content.post.unavailable': 'This content is unavailable.',
    'content.too_long': 'This video is too long to process. Try a shorter video.',
    'fetch.fail': 'Could not fetch the video. Please check the URL and try again.',
    'fetch.rate': 'Rate limit reached. Please wait a moment and try again.',
    'api.invalid_url': 'Invalid or unsupported URL format.',
  };

  return errorMessages[code]
    || `Extraction failed (code: ${code}). The video may be restricted or unavailable.`;
}

/**
 * Build a JSON Response with CORS headers.
 * @param {object} body
 * @param {number} status
 * @returns {Response}
 */
function jsonResponse(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...CORS_HEADERS,
    },
  });
}
