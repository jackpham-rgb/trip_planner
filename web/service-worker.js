// Minimal app-shell cache so the static frontend loads offline; API calls
// always hit the network since suggestions/plans need live scoring.
const CACHE = "trip-planner-shell-v1";
const SHELL = ["/", "index.html", "styles.css", "app.js", "manifest.json", "icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (url.pathname.startsWith("/api/")) return; // never cache API responses
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
