(function () {
  const MODEL_LABELS = {
    resnet50: "ResNet-50",
    convnext_base: "ConvNeXt-B",
    vit_b_16: "ViT-B/16",
    vit_l_16: "ViT-L/16",
    clip_vit_b32: "CLIP",
    dinov2_vit_b14: "DINOv2",
    "llava_1.5": "LLaVA",
    "llava_1.5_dpo": "LLaVA+DPO",
  };

  const CATEGORY_LABELS = {
    geometric: "Geometric",
    color: "Color",
    angle: "Angle",
    motion: "Motion",
    impossible: "Impossible",
  };

  const ILLUSION_LABELS = {
    muller_lyer: "Müller-Lyer",
    ponzo: "Ponzo",
    ebbinghaus: "Ebbinghaus",
    simultaneous_contrast: "Simultaneous contrast",
    whites_illusion: "White's illusion",
    zollner: "Zöllner",
    poggendorff: "Poggendorff",
    scintillating_grid: "Scintillating grid",
    rotating_snakes: "Rotating snakes",
    illusion_vqa: "IllusionVQA",
    hallusion_bench: "HallusionBench",
  };

  const INTERPRETATION_TEXT = {
    near_human: "Model illusory rate is close to the category reference (within ~5%).",
    under_illusory:
      "Model rarely picks the human-illusory answer, often “correct” despite the illusion.",
    over_illusory: "Model over-reports illusion relative to the category reference.",
    moderate_mismatch: "Partial mismatch in illusory rate vs. the category reference.",
  };

  let heasData = null;
  let cellDetails = null;
  let selected = null;
  let compareChart = null;

  function heasColor(value) {
    if (value == null || Number.isNaN(value)) return "#e2e8f0";
    const t = Math.max(0, Math.min(1, value));
    const r = Math.round(215 + (34 - 215) * t);
    const g = Math.round(48 + (139 - 48) * t);
    const b = Math.round(39 + (34 - 39) * t);
    return `rgb(${r},${g},${b})`;
  }

  function textColor(value) {
    if (value == null) return "#718096";
    return value > 0.55 ? "#fff" : "#1a202c";
  }

  function cellKey(cat, model) {
    return `${cat}|${model}`;
  }

  function fmt(v, digits = 2) {
    return v != null && !Number.isNaN(v) ? v.toFixed(digits) : "N/A";
  }

  function renderDetailPanel(cat, model) {
    const panel = document.getElementById("heas-detail-panel");
    if (!panel) return;

    const key = cellKey(cat, model);
    const detail = cellDetails?.[key];
    const heas = heasData?.values[cat]?.[model];
    const pHuman = heasData?.human_baselines[cat];

    if (heas == null && !detail) {
      panel.innerHTML =
        "<p class=\"heas-detail-empty\">No data for this model on this category (not evaluated).</p>";
      return;
    }

    const pModel = detail?.p_model;
    const interp = detail?.interpretation
      ? INTERPRETATION_TEXT[detail.interpretation] || ""
      : "";

    let illusionRows = "";
    if (detail?.illusions?.length) {
      const sorted = [...detail.illusions].sort((a, b) =>
        (ILLUSION_LABELS[a.illusion_type] || a.illusion_type).localeCompare(
          ILLUSION_LABELS[b.illusion_type] || b.illusion_type
        )
      );
      illusionRows = sorted
        .map((row) => {
          const name = ILLUSION_LABELS[row.illusion_type] || row.illusion_type;
          return `<tr>
            <td>${name}</td>
            <td>${fmt(row.p_illusory)}</td>
            <td>${fmt(row.p_correct)}</td>
            <td>${fmt(row.p_other)}</td>
            <td>${row.control_pass_rate != null ? fmt(row.control_pass_rate) : "N/A"}</td>
          </tr>`;
        })
        .join("");
    }

    const pHumanPct = pHuman != null ? pHuman * 100 : 0;
    const pModelPct = pModel != null ? pModel * 100 : 0;

    panel.innerHTML = `
      <h4>${MODEL_LABELS[model] || model} · ${CATEGORY_LABELS[cat] || cat}</h4>
      <div class="heas-rate-bars">
        <div class="heas-rate-row">
          <span class="heas-rate-label">Reference rate</span>
          <div class="heas-rate-track"><div class="heas-rate-fill human" style="width:${pHumanPct}%"></div></div>
          <span class="heas-rate-val">${fmt(pHuman)}</span>
        </div>
        <div class="heas-rate-row">
          <span class="heas-rate-label">Model illusory</span>
          <div class="heas-rate-track"><div class="heas-rate-fill model" style="width:${pModelPct}%"></div></div>
          <span class="heas-rate-val">${fmt(pModel)}</span>
        </div>
      </div>
      <p class="heas-detail-stats">
        <strong>HEAS:</strong> ${fmt(heas, 3)}
        · <strong>|p<sub>model</sub> − p<sub>human</sub>|:</strong> ${fmt(detail?.alignment_gap, 3)}
      </p>
      ${interp ? `<p class="heas-detail-interp">${interp}</p>` : ""}
      ${
        illusionRows
          ? `<table class="data-table heas-illusion-table">
          <thead><tr><th>Illusion</th><th>P(illusory)</th><th>P(correct)</th><th>P(other)</th><th>Control pass</th></tr></thead>
          <tbody>${illusionRows}</tbody>
        </table>`
          : ""
      }
    `;
  }

  function selectCell(cat, model, cellEl, { scroll = false } = {}) {
    selected = { cat, model };
    document.querySelectorAll(".heas-cell.selected").forEach((el) => el.classList.remove("selected"));
    if (cellEl) cellEl.classList.add("selected");
    renderDetailPanel(cat, model);
    if (scroll) {
      const panel = document.getElementById("heas-detail-panel");
      if (panel) panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function renderHeatmap() {
    const root = document.getElementById("heas-heatmap");
    if (!root || !heasData) return;

    const models = heasData.models;
    const categories = heasData.categories;

    root.innerHTML = "";
    root.style.gridTemplateColumns = `100px repeat(${models.length}, minmax(72px, 1fr))`;

    root.appendChild(document.createElement("div")).className = "heas-header";

    models.forEach((m) => {
      const h = document.createElement("div");
      h.className = "heas-header";
      h.textContent = MODEL_LABELS[m] || m;
      root.appendChild(h);
    });

    categories.forEach((cat) => {
      const label = document.createElement("div");
      label.className = "heas-row-label";
      label.textContent = CATEGORY_LABELS[cat] || cat;
      root.appendChild(label);

      models.forEach((m) => {
        const v = heasData.values[cat]?.[m];
        const cell = document.createElement("button");
        cell.type = "button";
        cell.className = "heas-cell";
        cell.style.background = heasColor(v);
        cell.style.color = textColor(v);
        cell.textContent = v != null ? v.toFixed(2) : "N/A";
        cell.disabled = v == null;
        cell.title = "Click for breakdown";

        if (
          selected &&
          selected.cat === cat &&
          selected.model === m
        ) {
          cell.classList.add("selected");
        }

        cell.addEventListener("click", () => selectCell(cat, m, cell, { scroll: true }));
        root.appendChild(cell);
      });
    });
  }

  function renderCompareChart() {
    const canvas = document.getElementById("heas-compare-chart");
    const selA = document.getElementById("heas-compare-a");
    const selB = document.getElementById("heas-compare-b");
    if (!canvas || !selA || !selB || !heasData || typeof Chart === "undefined") return;

    const modelA = selA.value;
    const modelB = selB.value;
    const labels = heasData.categories.map((c) => CATEGORY_LABELS[c] || c);

    const dataA = heasData.categories.map((c) => heasData.values[c]?.[modelA] ?? null);
    const dataB = heasData.categories.map((c) => heasData.values[c]?.[modelB] ?? null);

    if (compareChart) compareChart.destroy();

    compareChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: MODEL_LABELS[modelA] || modelA,
            data: dataA,
            backgroundColor: "rgba(49, 130, 206, 0.75)",
          },
          {
            label: MODEL_LABELS[modelB] || modelB,
            data: dataB,
            backgroundColor: "rgba(128, 90, 213, 0.75)",
          },
        ],
      },
      options: SiteUtils.chartOptions({
        title: "Category-level HEAS comparison",
        scales: {
          y: { min: 0, max: 1, title: { text: "HEAS" } },
        },
      }),
    });
  }

  function setupCompareControls() {
    const selA = document.getElementById("heas-compare-a");
    const selB = document.getElementById("heas-compare-b");
    if (!selA || !selB || !heasData) return;

    const opts = heasData.models
      .map((m) => `<option value="${m}">${MODEL_LABELS[m] || m}</option>`)
      .join("");
    selA.innerHTML = opts;
    selB.innerHTML = opts;
    selA.value = "resnet50";
    selB.value = "llava_1.5_dpo";

    const onChange = () => renderCompareChart();
    selA.addEventListener("change", onChange);
    selB.addEventListener("change", onChange);
    renderCompareChart();
  }

  function setupModeTabs() {
    const tabs = document.querySelectorAll("#heas-explorer-tabs button");
    const explore = document.getElementById("heas-mode-explore");
    const compare = document.getElementById("heas-mode-compare");

    tabs.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.mode;
        tabs.forEach((t) => t.classList.toggle("active", t === btn));
        if (explore) explore.hidden = mode !== "explore";
        if (compare) compare.hidden = mode !== "compare";
        if (mode === "compare") renderCompareChart();
      });
    });
  }

  async function initHeasExplorer() {
    const root = document.getElementById("heas-heatmap");
    if (!root) return;

    try {
      const [heas, details] = await Promise.all([
        SiteUtils.fetchJSON("./data/heas_table.json"),
        SiteUtils.fetchJSON("./data/heas_cell_details.json"),
      ]);
      heasData = heas;
      cellDetails = details;
    } catch (err) {
      SiteUtils.showError(root, `Could not load HEAS data. ${err.message}`);
      return;
    }

    renderHeatmap();
    setupCompareControls();
    setupModeTabs();
  }

  SiteUtils.onReady(() => initHeasExplorer().catch(console.error));
})();
