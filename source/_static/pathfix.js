/*
 * Fix relative "_static/..." image paths on nested pages.
 * Sphinx exposes URL_ROOT (e.g., "../" for pages in subfolders).
 */
(function () {
  function getUrlRoot() {
    if (
      typeof window !== "undefined" &&
      window.DOCUMENTATION_OPTIONS &&
      typeof window.DOCUMENTATION_OPTIONS.URL_ROOT === "string"
    ) {
      return window.DOCUMENTATION_OPTIONS.URL_ROOT;
    }
    return "";
  }

  function rewriteImagePaths() {
    var urlRoot = getUrlRoot();
    if (!urlRoot) return;

    var images = document.querySelectorAll('img[src^="_static/"]');
    images.forEach(function (img) {
      img.setAttribute("src", urlRoot + img.getAttribute("src"));
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", rewriteImagePaths);
  } else {
    rewriteImagePaths();
  }
})();
