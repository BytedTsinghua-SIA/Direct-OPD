const syncMobileHeroScale = () => {
  const viewportWidth = document.documentElement.clientWidth || window.innerWidth;
  const scale = viewportWidth <= 560 ? Math.max(0.24, (viewportWidth - 16) / 1040) : 1;
  document.documentElement.style.setProperty("--hero-mobile-scale", scale.toFixed(4));
};

syncMobileHeroScale();
window.addEventListener("resize", syncMobileHeroScale);
window.visualViewport?.addEventListener("resize", syncMobileHeroScale);
