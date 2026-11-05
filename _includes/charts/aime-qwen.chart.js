function buildOption(data) {
  function formatScore(value) {
    var score = Number(value);
    return (Number.isInteger(score) ? score.toFixed(1) : String(score)) + "%";
  }

  return {
    title: [
      { text: "AIME · Qwen3-8B", subtext: "得分（%）↑" },
      { text: "", show: false }
    ],
    xAxis: {
      type: "category",
      data: data.map(function (row) { return row.model; }),
      axisLabel: {
        formatter: function (name) { return name.replace(" · ", "\n"); }
      }
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 100,
      axisLabel: { formatter: formatScore }
    },
    tooltip: {
      trigger: "axis",
      formatter: function (params) {
        if (!params) return "";
        var item = Array.isArray(params) ? params[0] : params;
        if (!item || item.value == null) return "";
        return item.name + "<br>得分: " + formatScore(item.value);
      }
    },
    series: [
      {
        name: "得分",
        type: "bar",
        data: data.map(function (row) { return row.score; }),
        itemStyle: { color: "#61A9BD" },
        label: { show: true, position: "top", formatter: function (params) { return formatScore(params.value); } }
      }
    ]
  };
}
