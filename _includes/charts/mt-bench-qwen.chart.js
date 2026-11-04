function buildOption(data) {
  function formatScore(value) {
    return Number(value).toFixed(2) + " 分";
  }

  return {
    title: [
      { text: "MT-Bench · Qwen 系列 7B", subtext: "Judge Score（满分 10 分）↑" },
      { text: "", show: false }
    ],
    xAxis: {
      type: "category",
      data: data.map(function (row) { return row.model; }),
      axisLabel: {
        formatter: function (name) { return name.replace(/^Qwen-?/, "Qwen\n"); }
      }
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 10,
      axisLabel: { formatter: formatScore }
    },
    tooltip: {
      trigger: "axis",
      formatter: function (params) {
        if (!params) return "";
        var item = Array.isArray(params) ? params[0] : params;
        if (!item || item.value == null) return "";
        return item.name + "<br>Judge Score: " + formatScore(item.value) + " / 10";
      }
    },
    series: [
      {
        name: "Judge Score",
        type: "bar",
        data: data.map(function (row) { return row.score; }),
        itemStyle: { color: "#61A9BD" },
        label: { show: true, position: "top", formatter: function (params) { return formatScore(params.value); } }
      }
    ]
  };
}
