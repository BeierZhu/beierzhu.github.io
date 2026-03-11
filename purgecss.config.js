module.exports = {
  content: ["_site/**/*.html", "_site/**/*.js"],
  css: ["_site/assets/css/*.css"],
  output: "_site/assets/css/",
  skippedContentGlobs: ["_site/assets/**/*.html"],
  safelist: {
    standard: [
      "hiring-banner",
      "stats-tables-container",
      "venue-stats-table",
      "class-stats-table",
      "total-row",
    ],
    deep: [
      /^hiring-banner/,
      /^stats-tables-container/,
      /^venue-stats-table/,
      /^class-stats-table/,
    ],
  },
};
