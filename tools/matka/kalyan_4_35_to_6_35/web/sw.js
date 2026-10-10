// Kalyan Predictor Service Worker (PWA) - Zev Glass 2.0 Offline Resilience
const CACHE_NAME = 'kalyan-predictor-cache-v2';
const STATIC_ASSETS = [
  './',
  './index.html',
  './dashboard.html',
  './kalyan_penal_chart.html',
  './favicon.svg',
  './icon-192.png',
  './icon-512.png',
  './manifest.json',
  './prediction_data.json',
  './history.json',
  'https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
      .catch((err) => {
        console.warn('[SW] Precache failed during install:', err);
        return self.skipWaiting();
      })
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) return caches.delete(key);
        })
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);

  // Network-first for dynamic live API endpoints
  if (url.pathname.includes('/api/')) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response && response.status === 200) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(event.request) || caches.match('./prediction_data.json'))
    );
  } else {
    // Cache-first fallback to network for static assets and HTML shells
    event.respondWith(
      caches.match(event.request).then((cached) => {
        return cached || fetch(event.request).then((response) => {
          if (response && response.status === 200 && (url.origin === location.origin || url.origin === 'https://www.gstatic.com')) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        }).catch(() => {
          if (event.request.headers.get('accept')?.includes('text/html')) {
            return caches.match('./index.html');
          }
        });
      })
    );
  }
});
