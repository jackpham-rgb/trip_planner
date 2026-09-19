/**
 * Paste your own Firebase project's config below (Firebase console ->
 * Project settings -> General -> "Your apps" -> Web app -> SDK setup and
 * configuration -> Config). Full walkthrough: README.md, "Your own Firebase
 * project" section.
 *
 * This file is committed with placeholder values on purpose. Firebase's web
 * config is NOT a secret the way a server API key is -- it just says which
 * project to talk to, not who's allowed to. Real access control lives in
 * the database security rules (firebase-rules.json), not in hiding this
 * file. Until you fill this in, the app runs fine using the browser's
 * localStorage instead (see app.js) -- no Firebase project required to try
 * it out or to demo it.
 */
window.FIREBASE_CONFIG = {
  apiKey: "YOUR_API_KEY",
  authDomain: "YOUR_PROJECT.firebaseapp.com",
  databaseURL: "https://YOUR_PROJECT-default-rtdb.firebaseio.com",
  projectId: "YOUR_PROJECT",
  storageBucket: "YOUR_PROJECT.appspot.com",
  messagingSenderId: "YOUR_SENDER_ID",
  appId: "YOUR_APP_ID",
};

window.FIREBASE_IS_CONFIGURED = window.FIREBASE_CONFIG.apiKey !== "YOUR_API_KEY";
