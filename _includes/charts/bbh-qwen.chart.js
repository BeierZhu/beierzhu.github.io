function buildOption(data) {
  function formatEM(value) {
    return Number(value).toFixed(1) + "%";
  }

  return {
    title: [
      { text: "BBH · Qwen 系列 7B / 8B", subtext: "EM (%) ↑" },
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
      axisLabel: { formatter: formatEM }
    },
    tooltip: {
      trigger: "axis",
      formatter: function (params) {
        if (!params) return "";
        var item = Array.isArray(params) ? params[0] : params;
        if (!item || item.value == null) return "";
        return item.name + "<br>EM: " + formatEM(item.value);
      }
    },
    series: [
      {
        name: "EM",
        type: "bar",
        data: data.map(function (row) { return row.em; }),
        itemStyle: { color: "#61A9BD" },
        label: { show: true, position: "top", formatter: function (params) { return formatEM(params.value); } }
      }
    ]
  };
}
