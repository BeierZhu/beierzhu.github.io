function buildOption(data) {
  function formatScore(value) {
    var score = Number(value);
    return (Number.isInteger(score) ? score.toFixed(1) : String(score)) + "%";
  }

  return {
    title: [
      { text: "MBPP · Qwen 系列 7B–8B", subtext: "Pass@1 (%) ↑" },
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
      max: 100,
      axisLabel: { formatter: formatScore }
    },
    tooltip: {
      trigger: "axis",
      formatter: function (params) {
        if (!params) return "";
        var item = Array.isArray(params) ? params[0] : params;
        if (!item || item.value == null) return "";
        return item.name + "<br>Pass@1: " + formatScore(item.value);
      }
    },
    series: [
      {
        name: "Pass@1",
        type: "bar",
        data: data.map(function (row) { return row.pass_at_1; }),
        itemStyle: { color: "#61A9BD" },
        label: { show: true, position: "top", formatter: function (params) { return formatScore(params.value); } }
      }
    ]
  };
}
