const map = L.map("map").setView([10.8505, 76.2711], 7);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let marker = null;

function updateCoords(lat, lng) {
  document.getElementById("latitude").value = lat.toFixed(6);
  document.getElementById("longitude").value = lng.toFixed(6);
  document.getElementById("coords").textContent = `Selected: ${lat.toFixed(6)}, ${lng.toFixed(6)}`;
}

map.on("click", (e) => {
  const { lat, lng } = e.latlng;

  if (marker) {
    marker.setLatLng(e.latlng);
  } else {
    marker = L.marker(e.latlng).addTo(map);
  }

  updateCoords(lat, lng);
});

const form = document.getElementById("uploadForm");
const submitBtn = document.getElementById("submitBtn");
const statusEl = document.getElementById("processingStatus");

form.addEventListener("submit", (e) => {
  const lat = document.getElementById("latitude").value;
  const lng = document.getElementById("longitude").value;
  const video = document.getElementById("video").files[0];

  if (!video) {
    e.preventDefault();
    alert("Please choose a video file.");
    return;
  }

  if (!lat || !lng) {
    e.preventDefault();
    alert("Please click a location on the map.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Processing...";
  statusEl.textContent = "Uploading and running AI detection. This can take 1-5 minutes for larger videos.";
});
