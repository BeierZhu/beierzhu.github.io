document.querySelectorAll(".benchmark-citation-toggle, .benchmark-datasets-toggle, .benchmark-performance-toggle, .benchmark-examples-toggle").forEach((button) => {
  const detailRow = document.getElementById(button.getAttribute("aria-controls"));
  if (!detailRow) return;

  button.addEventListener("click", () => {
    detailRow.hidden = !detailRow.hidden;
    button.setAttribute("aria-expanded", String(!detailRow.hidden));
    detailRow.dispatchEvent(new CustomEvent("benchmark:toggle", { detail: { open: !detailRow.hidden } }));
  });
});
