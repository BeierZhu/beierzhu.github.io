function buildOption(data) {
  function formatEM(value) {
    return Number(value).toFixed(1) + "%";
  }

  return {
    title: [
      {
        text: "Qwen3-8B 在所列模型中 EM 最高",
        subtext: "Natural Questions · Qwen 系列 7B / 8B · EM (%) ↑"
      },
      { text: "数据由用户提供；测评设置与日期未提供。" }
    ],
    xAxis: {
      type: "category",
      data: data.map(function (row) { return row.model; }),
      axisLabel: {
        formatter: function (name) {
          return name.replace("-Instruct", "\nInstruct");
        }
      }
    },
    yAxis: {
      type: "value",
      min: 0,
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
