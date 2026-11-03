(() => {
  let runtimePromise;

  function loadRuntime(container) {
    if (window.AttaChartPreview) return Promise.resolve(window.AttaChartPreview);
    if (runtimePromise) return runtimePromise;

    runtimePromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = container.dataset.runtimeUrl;
      script.integrity = container.dataset.runtimeIntegrity;
      script.crossOrigin = "anonymous";
      const timer = setTimeout(() => {
        script.remove();
        reject(new Error("Chart runtime load timed out"));
      }, 15000);
      script.onload = () => {
        clearTimeout(timer);
        if (window.AttaChartPreview) resolve(window.AttaChartPreview);
        else reject(new Error("Chart runtime did not initialize"));
      };
      script.onerror = () => {
        clearTimeout(timer);
        script.remove();
        reject(new Error("Chart runtime failed to load"));
      };
      document.head.append(script);
    }).catch((error) => {
      runtimePromise = null;
      throw error;
    });

    return runtimePromise;
  }

  function chartTheme() {
    return document.documentElement.dataset.theme === "dark" ? "AttaDark" : "AttaLight";
  }

  document.querySelectorAll(".benchmark-performance-row").forEach((row) => {
    const container = row.querySelector("[data-benchmark-chart]");
    const status = row.querySelector(".benchmark-performance-status");
    const data = JSON.parse(document.getElementById(container.dataset.chartData).textContent);
    let handle;
    let pending;

    function showChart() {
      if (handle || pending) return;
      status.hidden = false;
      status.textContent = "正在加载图表…";

      pending = (async () => {
        const runtime = await loadRuntime(container);
        if (row.hidden) return;
        handle = await runtime.render({
          el: container,
          data,
          builder: window[container.dataset.chartBuilder],
          theme: chartTheme(),
        });
        handle.setTheme(chartTheme());
        status.hidden = true;
      })().catch((error) => {
        console.error("Benchmark chart failed:", error);
        status.hidden = false;
        status.textContent = "图表加载失败，请收起后重试。模型 EM：" +
          data.map((item) => item.model + " " + item.em.toFixed(1) + "%").join("；");
      }).finally(() => {
        pending = null;
      });
    }

    row.addEventListener("benchmark:toggle", (event) => {
      if (event.detail.open) showChart();
    });

    new MutationObserver(() => {
      if (handle) handle.setTheme(chartTheme());
    }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  });
})();
