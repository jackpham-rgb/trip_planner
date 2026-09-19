// Minimal app-shell cache so the static frontend loads offline; live
// suggestions/plans still need option_bank.json + (optionally) Firebase, so
// those network calls are never intercepted here -- only the app's own
// files are cached.
//
// Every path below is RELATIVE, resolved against this script's own URL by
// cache.addAll(). That matters because GitHub Pages project sites serve
// from a subpath (https://user.github.io/repo-name/), not the domain root
// -- a hardcoded "/" would silently point at the wrong site.
const CACHE = "trip-planner-shell-v2";
const SHELL = [
  "./", "index.html", "styles.css", "app.js", "recommender.js",
  "firebase-config.js", "manifest.json", "icon.svg",
];

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
  if (url.origin !== self.location.origin) return; // never intercept Firebase/weather/geocode calls
  if (url.pathname.endsWith("option_bank.json")) return; // content should always be fresh, not cached
  event.respondWith(
    caches.match(event.request).then((cached) => cached || fetch(event.request))
  );
});
