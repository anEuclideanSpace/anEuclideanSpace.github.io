document.addEventListener("DOMContentLoaded", () => {
  const callouts = document.querySelectorAll(".note-content blockquote");

  callouts.forEach((callout) => {
    const firstParagraph = callout.querySelector("p");
    if (!firstParagraph) return;

    const marker = firstParagraph.innerHTML.match(/^\[!(\w+)\]([+-])?\s*([^<]*)/i);
    if (!marker) return;

    const type = marker[1].toLowerCase();
    const title = marker[3].trim() || type;
    const replacement = `<span class="callout-title">${title}</span><br>`;

    firstParagraph.innerHTML = firstParagraph.innerHTML.replace(marker[0], replacement);
    callout.classList.add("callout", `callout-${type}`);
  });
});
