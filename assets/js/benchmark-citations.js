document.querySelectorAll(".benchmark-citation-toggle").forEach((button) => {
  const citationRow = document.getElementById(button.getAttribute("aria-controls"));
  if (!citationRow) return;

  button.addEventListener("click", () => {
    citationRow.hidden = !citationRow.hidden;
    button.setAttribute("aria-expanded", String(!citationRow.hidden));
  });
});
