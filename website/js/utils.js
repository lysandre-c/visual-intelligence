/** Shared helpers for the project website. */
(function (global) {
  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn);
    } else {
      fn();
    }
  }

  async function fetchJSON(path) {
    const res = await fetch(path);
    if (!res.ok) {
      throw new Error(`Failed to load ${path}: ${res.status}`);
    }
    return res.json();
  }

  function showError(el, message) {
    if (!el) return;
    el.innerHTML = `<p class="load-error" role="alert">${message}</p>`;
  }

  global.SiteUtils = { onReady, fetchJSON, showError };
})(window);
