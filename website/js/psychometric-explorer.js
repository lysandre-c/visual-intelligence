(function () {
  const MODEL_COLORS = {
    resnet50: "#3182ce",
    convnext_base: "#2c5282",
    vit_b_16: "#38a169",
    vit_l_16: "#276749",
    clip_vit_b32: "#d69e2e",
    dinov2_vit_b14: "#dd6b20",
    "llava_1.5": "#805ad5",
    "llava_1.5_dpo": "#553c9a",
  };

  const MODEL_LABELS = {
    resnet50: "ResNet-50",
    convnext_base: "ConvNeXt-B",
    vit_b_16: "ViT-B/16",
    vit_l_16: "ViT-L/16",
    clip_vit_b32: "CLIP ViT-B/32",
    dinov2_vit_b14: "DINOv2",
    "llava_1.5": "LLaVA-1.5",
    "llava_1.5_dpo": "LLaVA + DPO",
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
  };

  const PARAM_LABELS = {
    fin_length: "Fin length (px)",
    convergence_deg: "Convergence (°)",
    large_sat_radius: "Satellite radius",
    contrast_delta: "Contrast delta",
    stripe_height: "Stripe height",
    hatch_angle_deg: "Hatch angle (°)",
    occluder_width: "Occluder width",
    disc_radius: "Disc radius",
    wheel_radius: "Wheel radius",
  };

  let chart = null;
  let psychData = null;

  function getSelectedModels() {
    return [...document.querySelectorAll("#psych-model-checkboxes input:checked")].map(
      (el) => el.value
    );
  }

  function updateChart() {
    const illusion = document.getElementById("psych-illusion-select")?.value;
    const canvas = document.getElementById("psych-chart");
    const link = document.getElementById("psych-static-link");
    if (!psychData || !illusion || !canvas) return;

    if (typeof Chart === "undefined") {
      SiteUtils.showError(
        canvas.parentElement,
        "Chart.js failed to load. Check your network connection."
      );
      return;
    }

    const info = psychData.illusions[illusion];
    if (!info) return;

    const models = getSelectedModels();
    const paramLabel = PARAM_LABELS[info.sweep_param] || info.sweep_param;

    if (link) {
      link.href = `./assets/figures/psychometric_${illusion}.png`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = `View static figure: psychometric_${illusion}.png (opens in new tab)`;
    }

    const datasets = models.map((m) => {
      const points = info.models[m] || [];
      return {
        label: MODEL_LABELS[m] || m,
        data: points
          .filter((p) => p.illusory_rate != null)
          .map((p) => ({ x: p.param, y: p.illusory_rate })),
        borderColor: MODEL_COLORS[m] || "#718096",
        backgroundColor: "transparent",
        tension: 0.25,
        pointRadius: 3,
      };
    });

    const params = info.param_values;
    datasets.push({
      label: "Category reference",
      data: params.map((p) => ({ x: p, y: info.human_rate })),
      borderColor: "#e53e3e",
      borderDash: [6, 4],
      pointRadius: 0,
      tension: 0,
    });

    if (chart) chart.destroy();

    chart = new Chart(canvas, {
      type: "line",
      data: { datasets },
      options: SiteUtils.chartOptions({
        title: ILLUSION_LABELS[illusion] || illusion,
        scales: {
          x: {
            type: "linear",
            title: { text: paramLabel },
          },
          y: {
            min: 0,
            max: 1,
            title: { text: "P(illusory response)" },
          },
        },
      }),
    });
  }

  function buildModelCheckboxes(illusion) {
    const wrap = document.getElementById("psych-model-checkboxes");
    if (!wrap || !psychData) return;
    const info = psychData.illusions[illusion];
    if (!info) return;

    const defaultOn = new Set([
      "resnet50",
      "clip_vit_b32",
      "llava_1.5",
      "llava_1.5_dpo",
    ]);

    wrap.innerHTML = "";
    Object.keys(info.models).forEach((m) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = m;
      input.checked = defaultOn.has(m);
      input.addEventListener("change", updateChart);
      label.appendChild(input);
      label.appendChild(document.createTextNode(" " + (MODEL_LABELS[m] || m)));
      wrap.appendChild(label);
    });
  }

  async function initPsychometricExplorer() {
    const select = document.getElementById("psych-illusion-select");
    const panel = document.querySelector("#results-psych .interactive-panel");
    if (!select) return;

    try {
      psychData = await SiteUtils.fetchJSON("./data/psychometric_curves.json");
    } catch (err) {
      SiteUtils.showError(panel, `Could not load psychometric data. ${err.message}`);
      return;
    }

    const illusions = Object.keys(psychData.illusions).sort((a, b) => {
      const la = ILLUSION_LABELS[a] || a;
      const lb = ILLUSION_LABELS[b] || b;
      return la.localeCompare(lb);
    });

    select.innerHTML = illusions
      .map((k) => `<option value="${k}">${ILLUSION_LABELS[k] || k}</option>`)
      .join("");

    select.addEventListener("change", () => {
      buildModelCheckboxes(select.value);
      updateChart();
    });

    buildModelCheckboxes(select.value);
    updateChart();
  }

  SiteUtils.onReady(() => initPsychometricExplorer().catch(console.error));
})();
