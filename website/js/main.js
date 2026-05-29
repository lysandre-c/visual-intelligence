(function () {
  function initSaliencyTabs() {
    const container = document.getElementById("saliency-gallery");
    if (!container) return;

    const tabs = container.querySelectorAll(".tab-bar button");
    const panels = container.querySelectorAll(".saliency-panel");

    tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.panel;
        tabs.forEach((t) => t.classList.toggle("active", t === btn));
        panels.forEach((p) => {
          p.hidden = p.id !== id;
        });
      });
    });
  }

  function openDetailsTarget(el) {
    if (!el || el.tagName !== "DETAILS") return;
    el.open = true;
    requestAnimationFrame(() => {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function initDetailsDeepLinks() {
    const handleHash = () => {
      const id = window.location.hash.replace(/^#/, "");
      if (!id) return;
      openDetailsTarget(document.getElementById(id));
    };

    document.querySelectorAll("a.open-details-link[href^='#']").forEach((link) => {
      link.addEventListener("click", (event) => {
        const id = link.getAttribute("href").slice(1);
        const details = document.getElementById(id);
        if (!details || details.tagName !== "DETAILS") return;
        event.preventDefault();
        openDetailsTarget(details);
        history.pushState(null, "", `#${id}`);
      });
    });

    window.addEventListener("hashchange", handleHash);
    handleHash();
  }

  function initPsychGalleryToggle() {
    const btn = document.getElementById("toggle-psych-gallery");
    const gallery = document.getElementById("psych-static-gallery");
    if (!btn || !gallery) return;

    gallery.classList.add("is-collapsed");
    gallery.removeAttribute("hidden");
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-controls", "psych-static-gallery");

    btn.addEventListener("click", () => {
      const collapsed = gallery.classList.toggle("is-collapsed");
      btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
      btn.textContent = collapsed
        ? "Show all static psychometric figures"
        : "Hide static psychometric figures";
    });
  }

  function boot() {
    initSaliencyTabs();
    initDetailsDeepLinks();
    initPsychGalleryToggle();
  }

  if (typeof SiteUtils !== "undefined") {
    SiteUtils.onReady(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
