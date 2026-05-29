/** Shared Chart.js styling: white canvas, minimal grid, matches site background. */
(function (global) {
  const TICK = "#64748b";
  const TITLE = "#0f172a";
  const GRID = "#eef1f4";

  function configureChartTheme() {
    if (!global.Chart) return;
    Chart.defaults.color = TICK;
    Chart.defaults.borderColor = GRID;
    Chart.defaults.backgroundColor = "#ffffff";
    Chart.defaults.font.family = "'Inter', system-ui, -apple-system, sans-serif";
    Chart.defaults.font.size = 12;
    if (Chart.defaults.plugins?.legend?.labels) {
      Chart.defaults.plugins.legend.labels.boxWidth = 10;
      Chart.defaults.plugins.legend.labels.padding = 14;
      Chart.defaults.plugins.legend.labels.usePointStyle = true;
    }
  }

  function scaleDefaults(overrides = {}) {
    const base = {
      grid: { color: GRID, drawBorder: false },
      border: { display: false },
      ticks: { color: TICK },
      title: { color: TICK, font: { size: 12, weight: "500" } },
    };
    const out = {};
    for (const key of ["x", "y"]) {
      out[key] = { ...base, ...(overrides[key] || {}) };
      if (overrides[key]?.title) {
        out[key].title = { display: true, ...base.title, ...overrides[key].title };
      }
      if (overrides[key]?.grid) {
        out[key].grid = { ...base.grid, ...overrides[key].grid };
      }
    }
    return out;
  }

  /** Build Chart.js options with white-background-friendly scales and plugins. */
  function chartOptions(custom = {}) {
    configureChartTheme();
    const plugins = {
      legend: {
        position: "bottom",
        labels: { color: TICK, padding: 16, usePointStyle: true },
      },
      ...custom.plugins,
    };
    if (custom.title) {
      plugins.title = {
        display: true,
        text: custom.title,
        color: TITLE,
        font: { size: 14, weight: "600" },
        padding: { bottom: 12 },
      };
    }
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins,
      scales: scaleDefaults(custom.scales || {}),
      ...custom.extra,
    };
  }

  configureChartTheme();

  const utils = global.SiteUtils || {};
  utils.configureChartTheme = configureChartTheme;
  utils.chartOptions = chartOptions;
  global.SiteUtils = utils;
})(window);
