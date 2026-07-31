(function () {
  var script = document.currentScript;
  var active = script.getAttribute("data-active") || "";
  var root = document.getElementById("page-nav-root");
  if (!root || !window.COURSE_SECTIONS) return;

  var idx = window.COURSE_SECTIONS.findIndex(function (s) { return s.id === active; });
  if (idx === -1) return;

  var prev = idx > 0 ? window.COURSE_SECTIONS[idx - 1] : null;
  var next = idx < window.COURSE_SECTIONS.length - 1 ? window.COURSE_SECTIONS[idx + 1] : null;

  var html = "";
  html += prev
    ? '<a class="prev" href="' + prev.href + '"><span class="dir">&larr; Previous</span>' + prev.id + " · " + prev.title + "</a>"
    : "<span></span>";
  html += next
    ? '<a class="next" href="' + next.href + '"><span class="dir">Next &rarr;</span>' + next.id + " · " + next.title + "</a>"
    : "<span></span>";

  root.innerHTML = html;
})();
