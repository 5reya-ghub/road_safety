const map = L.map("map").setView([10.8505, 76.2711], 7);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

let marker = null;
const latInput = document.getElementById("latitude");
const lngInput = document.getElementById("longitude");
const locationNameInput = document.getElementById("location_name");
const coordsEl = document.getElementById("coords");
const lookupStatusEl = document.getElementById("coordLookupStatus");
const applyCoordsBtn = document.getElementById("applyCoordsBtn");
const lookupLocationBtn = document.getElementById("lookupLocationBtn");

function isValidCoordinatePair(lat, lng) {
  return Number.isFinite(lat) && Number.isFinite(lng) && lat >= -90 && lat <= 90 && lng >= -180 && lng <= 180;
}

function updateCoords(lat, lng) {
  latInput.value = lat.toFixed(6);
  lngInput.value = lng.toFixed(6);
  coordsEl.textContent = `Selected: ${lat.toFixed(6)}, ${lng.toFixed(6)}`;
}

function setMarkerFromCoords(lat, lng, shouldZoom = true) {
  if (!isValidCoordinatePair(lat, lng)) return false;
  const latlng = { lat, lng };
  if (marker) {
    marker.setLatLng(latlng);
  } else {
    marker = L.marker(latlng).addTo(map);
  }
  if (shouldZoom) {
    map.setView([lat, lng], 15);
  }
  updateCoords(lat, lng);
  return true;
}

map.on("click", (e) => {
  const { lat, lng } = e.latlng;
  setMarkerFromCoords(lat, lng, false);
});

const form = document.getElementById("uploadForm");
const submitBtn = document.getElementById("submitBtn");
const statusEl = document.getElementById("processingStatus");

applyCoordsBtn.addEventListener("click", () => {
  const lat = Number(latInput.value);
  const lng = Number(lngInput.value);
  if (!setMarkerFromCoords(lat, lng, true)) {
    lookupStatusEl.textContent = "Invalid coordinates. Latitude must be -90..90 and longitude -180..180.";
    return;
  }
  lookupStatusEl.textContent = "Map pin moved to provided coordinates.";
});

lookupLocationBtn.addEventListener("click", async () => {
  const lat = Number(latInput.value);
  const lng = Number(lngInput.value);
  if (!isValidCoordinatePair(lat, lng)) {
    lookupStatusEl.textContent = "Enter valid coordinates before location lookup.";
    return;
  }

  lookupStatusEl.textContent = "Looking up location name from coordinates...";
  lookupLocationBtn.disabled = true;
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lng)}`;
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`Lookup failed with status ${response.status}`);
    }
    const data = await response.json();
    const displayName = data?.display_name || "";
    if (displayName) {
      locationNameInput.value = displayName;
      lookupStatusEl.textContent = "Location name auto-filled from coordinates.";
    } else {
      lookupStatusEl.textContent = "No readable location name found for these coordinates.";
    }
  } catch (_err) {
    lookupStatusEl.textContent = "Could not fetch location name right now. You can still submit with coordinates.";
  } finally {
    lookupLocationBtn.disabled = false;
  }
});

form.addEventListener("submit", (e) => {
  const lat = Number(latInput.value);
  const lng = Number(lngInput.value);
  const video = document.getElementById("video").files[0];

  if (!video) {
    e.preventDefault();
    alert("Please choose a video file.");
    return;
  }

  if (!isValidCoordinatePair(lat, lng)) {
    e.preventDefault();
    alert("Please provide valid latitude and longitude.");
    return;
  }

  setMarkerFromCoords(lat, lng, false);
  submitBtn.disabled = true;
  submitBtn.textContent = "Processing...";
  statusEl.textContent = "Uploading and running AI detection. This can take 1-5 minutes for larger videos.";
});
