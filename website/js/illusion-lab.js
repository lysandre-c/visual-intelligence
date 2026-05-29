(function () {
  const CATEGORY_LABELS = {
    geometric: "Geometric / length",
    color: "Color / brightness",
    angle: "Angle / orientation",
    motion: "Motion-from-static",
    impossible: "Impossible / scene-level",
  };

  let galleryData = null;
  let paramValues = {};

  const canvasIll = document.getElementById("illusion-canvas");
  const canvasCtrl = document.getElementById("illusion-control-canvas");
  const canvasView = document.getElementById("illusion-canvas-view");
  const staticView = document.getElementById("illusion-static-view");
  const paramRoot = document.getElementById("illusion-param-controls");
  const imgIll = document.getElementById("illusion-gallery-illusion");
  const imgCtrl = document.getElementById("illusion-gallery-control");

  function setVisible(el, show) {
    if (!el) return;
    el.classList.toggle("is-hidden", !show);
  }

  function defaultFromEntry(entry) {
    const out = { ...(entry.params || {}) };
    for (const c of entry.controls || []) {
      if (out[c.param] == null) {
        const v = c.values ? c.values[0] : c.min;
        out[c.param] = v;
      }
    }
    return window.applyIllusionParamRules
      ? window.applyIllusionParamRules(entry.id, out)
      : out;
  }

  function getSelectedEntry() {
    const id = document.getElementById("illusion-type-select")?.value;
    return galleryData?.illusions?.find((e) => e.id === id);
  }

  function renderCurrent() {
    const entry = getSelectedEntry();
    if (!entry?.programmatic) return;
    const renderer = window.IllusionRenderers?.[entry.id];
    if (!renderer || !canvasIll || !canvasCtrl) return;

    const params = window.applyIllusionParamRules
      ? window.applyIllusionParamRules(entry.id, { ...paramValues })
      : { ...paramValues };

    renderer.render(canvasIll, canvasCtrl, params);
  }

  function buildParamControls(entry) {
    if (!paramRoot) return;
    paramRoot.innerHTML = "";
    paramValues = defaultFromEntry(entry);

    if (!entry.controls?.length) {
      setVisible(paramRoot, false);
      return;
    }

    setVisible(paramRoot, true);
    for (const spec of entry.controls) {
      const label = document.createElement("label");
      const valSpan = document.createElement("span");
      valSpan.textContent = String(paramValues[spec.param]);
      label.appendChild(document.createTextNode(`${spec.label}: `));
      label.appendChild(valSpan);

      if (spec.type === "select") {
        const sel = document.createElement("select");
        sel.dataset.param = spec.param;
        for (const v of spec.values) {
          const opt = document.createElement("option");
          opt.value = String(v);
          opt.textContent = String(v);
          if (paramValues[spec.param] === v) opt.selected = true;
          sel.appendChild(opt);
        }
        sel.addEventListener("change", () => {
          paramValues[spec.param] = parseFloat(sel.value);
          valSpan.textContent = sel.value;
          renderCurrent();
        });
        label.appendChild(sel);
      } else {
        const input = document.createElement("input");
        input.type = "range";
        input.min = String(spec.min);
        input.max = String(spec.max);
        input.step = String(spec.step ?? 1);
        input.value = String(paramValues[spec.param]);
        input.dataset.param = spec.param;
        input.addEventListener("input", () => {
          const step = spec.step ?? 1;
          const v =
            step < 1 ? parseFloat(input.value) : parseInt(input.value, 10);
          paramValues[spec.param] = v;
          valSpan.textContent = String(v);
          renderCurrent();
        });
        label.appendChild(input);
      }
      paramRoot.appendChild(label);
    }
  }

  function resizeCanvases(entry) {
    const spec = window.IllusionRenderers?.[entry.id];
    if (!spec || !canvasIll || !canvasCtrl) return;
    canvasIll.width = spec.width;
    canvasIll.height = spec.height;
    canvasCtrl.width = spec.width;
    canvasCtrl.height = spec.height;
  }

  function showEntry(entry) {
    if (!entry) return;

    const desc = document.getElementById("illusion-type-desc");
    const human = document.getElementById("illusion-human-effect");

    if (desc) {
      const parts = [entry.description, entry.control_note].filter(Boolean);
      desc.textContent = parts.join(" ");
    }
    if (human) {
      human.textContent = entry.human_effect
        ? `Human bias: ${entry.human_effect}`
        : "";
    }

    const programmatic = entry.programmatic === true;
    setVisible(canvasView, programmatic);
    setVisible(staticView, !programmatic);
    setVisible(paramRoot, programmatic && (entry.controls?.length > 0));

    if (programmatic) {
      buildParamControls(entry);
      resizeCanvases(entry);
      renderCurrent();
    } else if (imgIll && imgCtrl) {
      imgIll.src = entry.illusion_image;
      imgIll.alt = `${entry.label} illusion stimulus`;
      imgCtrl.src = entry.control_image;
      imgCtrl.alt = `${entry.label} control stimulus`;
    }
  }

  function populateSelectors(data) {
    const catSel = document.getElementById("illusion-category-select");
    const typeSel = document.getElementById("illusion-type-select");
    if (!catSel || !typeSel) return;

    const byCategory = {};
    for (const entry of data.illusions) {
      if (!byCategory[entry.category]) byCategory[entry.category] = [];
      byCategory[entry.category].push(entry);
    }

    const categories = data.category_order.filter((c) => byCategory[c]?.length);
    catSel.innerHTML = categories
      .map((c) => `<option value="${c}">${CATEGORY_LABELS[c] || c}</option>`)
      .join("");

    function fillTypes(category) {
      const items = byCategory[category] || [];
      typeSel.innerHTML = items
        .map((e) => `<option value="${e.id}">${e.label}</option>`)
        .join("");
      if (items[0]) showEntry(items[0]);
    }

    catSel.addEventListener("change", () => fillTypes(catSel.value));
    typeSel.addEventListener("change", () => {
      showEntry(data.illusions.find((e) => e.id === typeSel.value));
    });

    fillTypes(categories[0] || data.illusions[0]?.category);
  }

  async function initIllusionLab() {
    if (!document.getElementById("illusion-type-select")) return;
    if (!window.IllusionRenderers || !window.applyIllusionParamRules) {
      console.error("Load illusion-renderers.js before illusion-lab.js");
      return;
    }

    try {
      galleryData = await SiteUtils.fetchJSON("./data/illusion_gallery.json");
    } catch (err) {
      const panel = document.querySelector("#illusion-lab .interactive-panel");
      if (panel) {
        SiteUtils.showError(
          panel,
          `Could not load stimulus gallery. ${err.message}`
        );
      }
      return;
    }

    populateSelectors(galleryData);
  }

  SiteUtils.onReady(() => initIllusionLab().catch(console.error));
})();
