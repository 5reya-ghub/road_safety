function markerColor(level) {
  const normalized = (level || "").toUpperCase();
  if (normalized === "LOW") return "green";
  if (normalized === "MEDIUM") return "yellow";
  if (normalized === "HIGH") return "red";
  return "blue";
}

const data = window.RESULT_DATA;
const map = L.map("resultMap").setView([data.lat, data.lng], 16);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

const color = markerColor(data.riskLevel);
const marker = L.circleMarker([data.lat, data.lng], {
  radius: 10,
  color,
  fillColor: color,
  fillOpacity: 0.8,
}).addTo(map);

marker.bindPopup(`${data.locationName}<br>Risk: ${data.riskLevel}`).openPopup();

const video = document.getElementById("annotatedVideo");
const jumpButtons = document.querySelectorAll(".jump-btn");
if (video && jumpButtons.length > 0) {
  jumpButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = Number(btn.dataset.jumpS || "0");
      if (Number.isFinite(t)) {
        video.currentTime = Math.max(0, t);
        video.play().catch(() => {});
      }
    });
  });
}
