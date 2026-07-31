(function () {
  var SECTIONS = [
    { id: "01", slug: "01-background", title: "Background" },
    { id: "02", slug: "02-data-preparation", title: "Data Preparation" },
    { id: "03", slug: "03-model-architecture", title: "Model Architecture" },
    { id: "04", slug: "04-training-protocols", title: "Training Protocols" },
    { id: "05", slug: "05-inference", title: "Inference" },
    { id: "06", slug: "06-postprocessing", title: "Post-processing" },
    { id: "07", slug: "07-evaluation-visualization", title: "Evaluation & Visualization" },
    { id: "08", slug: "08-conclusion", title: "Conclusion" },
    { id: "09", slug: "09-quiz", title: "Quiz" },
  ];

  window.COURSE_SECTIONS = SECTIONS;

  var script = document.currentScript;
  var active = script.getAttribute("data-active") || "";
  var base = script.getAttribute("data-base") || "";
  var root = document.getElementById("nav-root");
  if (!root) return;

  var links = SECTIONS.map(function (s) {
    var cls = s.id === active ? "active" : "";
    return '<li><a class="' + cls + '" href="' + base + s.slug + '/"><span class="num">' + s.id + "</span>" + s.title + "</a></li>";
  }).join("");

  root.innerHTML =
    '<nav class="drawer" id="drawer">' +
    '<div class="drawer-title">Course Contents</div>' +
    '<ul><li><a href="' + base + 'index.html">Home</a></li>' + links + "</ul>" +
    "</nav>" +
    '<header class="site-header" id="site-header">' +
    '<div class="header-links"><a href="https://github.com/yws0322/minicourse-multimodal" target="_blank" rel="noopener" aria-label="GitHub">' +
    '<svg viewBox="0 0 16 16" width="20" height="20" fill="currentColor" aria-hidden="true">' +
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>' +
    "</svg></a></div>" +
    "</header>";

  window.addEventListener("scroll", function () {
    var header = document.getElementById("site-header");
    if (window.scrollY > 4) header.classList.add("scrolled");
    else header.classList.remove("scrolled");
  });
})();
