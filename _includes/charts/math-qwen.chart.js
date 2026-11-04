function buildOption(data) {
  function formatAccuracy(value) {
    var score = Number(value);
    return (Number.isInteger(score) ? score.toFixed(1) : String(score)) + "%";
  }

  return {
    title: [
      { text: "MATH · Qwen 系列 7B–8B", subtext: "Accuracy (%) ↑" },
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
      axisLabel: { formatter: formatAccuracy }
    },
    tooltip: {
      trigger: "axis",
      formatter: function (params) {
        if (!params) return "";
        var item = Array.isArray(params) ? params[0] : params;
        if (!item || item.value == null) return "";
        return item.name + "<br>Accuracy: " + formatAccuracy(item.value);
      }
    },
    series: [
      {
        name: "Accuracy",
        type: "bar",
        data: data.map(function (row) { return row.accuracy; }),
        itemStyle: { color: "#61A9BD" },
        label: { show: true, position: "top", formatter: function (params) { return formatAccuracy(params.value); } }
      }
    ]
  };
}
