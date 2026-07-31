(function () {
  var SECTIONS = [
    { id: "01", href: "01-background.html", title: "Background" },
    { id: "02", href: "02-data-preparation.html", title: "Data Preparation" },
    { id: "03", href: "03-model-architecture.html", title: "Model Architecture" },
    { id: "04", href: "04-training-protocols.html", title: "Training Protocols" },
    { id: "05", href: "05-inference.html", title: "Inference" },
    { id: "06", href: "06-postprocessing.html", title: "Post-processing" },
    { id: "07", href: "07-evaluation-visualization.html", title: "Evaluation & Visualization" },
    { id: "08", href: "08-conclusion.html", title: "Conclusion" },
    { id: "09", href: "09-quiz.html", title: "Quiz" },
  ];

  window.COURSE_SECTIONS = SECTIONS;

  var script = document.currentScript;
  var active = script.getAttribute("data-active") || "";
  var root = document.getElementById("nav-root");
  if (!root) return;

  var links = SECTIONS.map(function (s) {
    var cls = s.id === active ? "active" : "";
    return '<li><a class="' + cls + '" href="' + s.href + '"><span class="num">' + s.id + "</span>" + s.title + "</a></li>";
  }).join("");

  root.innerHTML =
    '<div class="drawer-backdrop" id="drawer-backdrop"></div>' +
    '<nav class="drawer" id="drawer">' +
    '<div class="drawer-title">Course Contents</div>' +
    '<ul><li><a href="index.html">Home</a></li>' + links + "</ul>" +
    "</nav>" +
    '<header class="site-header" id="site-header">' +
    '<button class="menu-btn" id="menu-btn" aria-label="Open menu"><span></span></button>' +
    '<a class="brand" href="index.html">Minicourse</a>' +
    '<div class="header-links"><a href="https://github.com/yws0322/minicourse-multimodal" target="_blank" rel="noopener" aria-label="GitHub">' +
    '<svg viewBox="0 0 16 16" width="20" height="20" fill="currentColor" aria-hidden="true">' +
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>' +
    "</svg></a></div>" +
    "</header>";

  var drawer = document.getElementById("drawer");
  var backdrop = document.getElementById("drawer-backdrop");
  var menuBtn = document.getElementById("menu-btn");

  function openDrawer() { drawer.classList.add("open"); backdrop.classList.add("open"); }
  function closeDrawer() { drawer.classList.remove("open"); backdrop.classList.remove("open"); }

  menuBtn.addEventListener("click", openDrawer);
  backdrop.addEventListener("click", closeDrawer);

  window.addEventListener("scroll", function () {
    var header = document.getElementById("site-header");
    if (window.scrollY > 4) header.classList.add("scrolled");
    else header.classList.remove("scrolled");
  });
})();
