document.querySelectorAll(".benchmark-citation-toggle, .benchmark-datasets-toggle").forEach((button) => {
  const detailRow = document.getElementById(button.getAttribute("aria-controls"));
  if (!detailRow) return;

  button.addEventListener("click", () => {
    detailRow.hidden = !detailRow.hidden;
    button.setAttribute("aria-expanded", String(!detailRow.hidden));
  });
});
