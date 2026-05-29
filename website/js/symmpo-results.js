(function () {
  const CATEGORY_LABELS = {
    geometric: "Geometric",
    color: "Color",
    angle: "Angle",
    motion: "Motion",
    impossible: "Impossible",
  };

  let symmpoChart = null;

  function setPending(panel, pendingMsg) {
    if (pendingMsg) pendingMsg.hidden = false;
    panel.classList.add("results-pending");
    const chartWrap = panel.querySelector(".symmpo-chart-wrap");
    if (chartWrap) chartWrap.hidden = true;
    panel.querySelectorAll(".symmpo-slot img").forEach((img) => {
      img.hidden = true;
    });
  }

  function setReady(panel, pendingMsg, data) {
    panel.classList.remove("results-pending");
    if (pendingMsg) pendingMsg.hidden = true;

    const chartWrap = panel.querySelector(".symmpo-chart-wrap");
    const canvas = document.getElementById("symmpo-delta-chart");
    if (chartWrap && canvas && data.rows?.length && typeof Chart !== "undefined") {
      chartWrap.hidden = false;
      const labels = data.rows.map(
        (r) => CATEGORY_LABELS[r.category] || r.category
      );
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
              label: "LLaVA + SymMPO",
              data: data.rows.map((r) => r.llava_symmpo),
              backgroundColor: "rgba(49, 130, 206, 0.85)",
            },
          ],
        },
        options: SiteUtils.chartOptions({
          title: "HEAS by alignment method (higher = closer to category reference)",
          scales: {
            y: { min: 0, max: 1, title: { text: "HEAS" } },
          },
        }),
      });
    }

    panel.querySelectorAll(".symmpo-slot img").forEach((img) => {
      img.hidden = false;
      img.addEventListener(
        "error",
        () => {
          img.hidden = true;
        },
        { once: true }
      );
    });
  }

  async function initSymmpoResults() {
    const panel = document.getElementById("symmpo-results-panel");
    if (!panel) return;

    const pendingMsg = panel.querySelector(".results-pending-msg");

    let data;
    try {
      data = await SiteUtils.fetchJSON("./data/symmpo_delta.json");
    } catch (err) {
      setPending(panel, pendingMsg);
      return;
    }

    if (data.status !== "ready" || !data.rows?.length) {
      setPending(panel, pendingMsg);
      return;
    }

    setReady(panel, pendingMsg, data);
  }

  SiteUtils.onReady(() => initSymmpoResults().catch(console.error));
})();
