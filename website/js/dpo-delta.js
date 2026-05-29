(function () {
  const DPO_CATEGORY_LABELS = {
    geometric: "Geometric",
    color: "Color",
    angle: "Angle",
    motion: "Motion",
    impossible: "Impossible",
  };

  let dpoChart = null;

  async function initDpoDeltaChart() {
    const canvas = document.getElementById("dpo-delta-chart");
    const panel = canvas?.closest(".interactive-panel");
    if (!canvas) return;

    if (typeof Chart === "undefined") {
      SiteUtils.showError(panel, "Chart.js failed to load.");
      return;
    }

    let rows;
    try {
      rows = await SiteUtils.fetchJSON("./data/dpo_delta.json");
    } catch (err) {
      SiteUtils.showError(panel, `Could not load DPO data. ${err.message}`);
      return;
    }

    if (!rows.length) return;

    const labels = rows.map((r) => DPO_CATEGORY_LABELS[r.category] || r.category);

    if (dpoChart) dpoChart.destroy();

    dpoChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "LLaVA-1.5",
            data: rows.map((r) => r.llava_base),
            backgroundColor: "rgba(128, 90, 213, 0.7)",
          },
          {
            label: "LLaVA + DPO",
            data: rows.map((r) => r.llava_dpo),
            backgroundColor: "rgba(85, 60, 154, 0.9)",
          },
        ],
      },
      options: SiteUtils.chartOptions({
        title: "DPO alignment gain (higher HEAS = closer match to category reference)",
        scales: {
          y: { min: 0, max: 1, title: { text: "HEAS" } },
        },
      }),
    });
  }

  SiteUtils.onReady(() => initDpoDeltaChart().catch(console.error));
})();
