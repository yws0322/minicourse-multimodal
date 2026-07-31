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
    '<a class="brand" href="index.html">Multimodal Survival Prediction — Minicourse</a>' +
    '<div class="header-links"><a href="https://github.com/yws0322/minicourse-multimodal" target="_blank" rel="noopener">GitHub</a></div>' +
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
