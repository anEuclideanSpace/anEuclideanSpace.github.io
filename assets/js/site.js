const transformCallouts = () => {
  const callouts = document.querySelectorAll(".note-content blockquote");

  callouts.forEach((callout) => {
    const firstParagraph = callout.querySelector("p");
    if (!firstParagraph) return;

    const firstNode = firstParagraph.firstChild;
    if (!firstNode || firstNode.nodeType !== Node.TEXT_NODE) return;

    const marker = firstNode.nodeValue.match(/^\[!(\w+)\]([+-])?[ \t]*([^\n]*)/i);
    if (!marker) return;

    const type = marker[1].toLowerCase();
    const title = marker[3].trim() || type;
    const titleElement = document.createElement("span");
    titleElement.className = "callout-title";
    titleElement.textContent = title;

    firstNode.nodeValue = firstNode.nodeValue
      .slice(marker[0].length)
      .replace(/^\n/, "");
    firstParagraph.prepend(document.createElement("br"));
    firstParagraph.prepend(titleElement);
    callout.classList.add("callout", `callout-${type}`);
  });
};

transformCallouts();
