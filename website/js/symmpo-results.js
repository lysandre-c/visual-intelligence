(function () {
  const CATEGORY_LABELS = {
    geometric: "Geometric",
    color: "Color",
    angle: "Angle",
    motion: "Motion",
    impossible: "Impossible",
  };

  let symmpoChart = null;

  function renderChart(chartWrap, canvas, data) {
    const labels = data.rows.map((r) => CATEGORY_LABELS[r.category] || r.category);
    const isGated = data.symmpo_metric === "control_gated";

    if (symmpoChart) symmpoChart.destroy();

    symmpoChart = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            label: "LLaVA-1.5",
            data: data.rows.map((r) => r.llava_base),
            backgroundColor: "rgba(128, 90, 213, 0.55)",
          },
          {
            label: "LLaVA + DPO",
            data: data.rows.map((r) => r.llava_dpo),
            backgroundColor: "rgba(85, 60, 154, 0.75)",
          },
          {
            label: isGated ? "LLaVA + SymMPO (gated)" : "LLaVA + SymMPO",
            data: data.rows.map((r) => r.llava_symmpo),
            backgroundColor: "rgba(49, 130, 206, 0.85)",
          },
        ],
      },
      options: SiteUtils.chartOptions({
        title: isGated
          ? "HEAS by alignment method (SymMPO: control-gated; base/DPO: standard)"
          : "HEAS by alignment method (higher = closer to category reference)",
        scales: {
          y: { min: 0, max: 1, title: { text: "HEAS" } },
        },
      }),
    });
  }

  async function initSymmpoChart() {
    const panel = document.getElementById("symmpo-results-panel");
    const chartWrap = panel?.querySelector(".symmpo-chart-wrap");
    const canvas = document.getElementById("symmpo-delta-chart");
    if (!panel || !chartWrap || !canvas) return;

    if (typeof Chart === "undefined") {
      SiteUtils.showError(chartWrap, "Chart.js failed to load.");
      return;
    }

    let data;
    try {
      data = await SiteUtils.fetchJSON("./data/symmpo_delta.json");
    } catch (err) {
      SiteUtils.showError(chartWrap, `Could not load SymMPO data. ${err.message}`);
      return;
    }

    if (data.status !== "ready" || !data.rows?.length) {
      SiteUtils.showError(
        chartWrap,
        "SymMPO comparison data is not available yet. Run export after adding results."
      );
      return;
    }

    try {
      renderChart(chartWrap, canvas, data);
    } catch (err) {
      console.error(err);
      SiteUtils.showError(chartWrap, `Could not render SymMPO chart. ${err.message}`);
    }
  }

  SiteUtils.onReady(() => initSymmpoChart().catch(console.error));
})();
