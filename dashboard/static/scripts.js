const API_BASE = "http://127.0.0.1:5000/api";

// ---------------------------
// Fetch Recent AQI Readings
// ---------------------------
async function loadRecentReadings() {
    try {
        const res = await fetch(`${API_BASE}/recent-readings`);
        const data = await res.json();

        const tbody = document.getElementById("readings-tbody");
        tbody.innerHTML = "";

        data.slice(0, 20).forEach(row => {
            tbody.innerHTML += `
                <tr>
                    <td>${row.city}</td>
                    <td>${row.station}</td>
                    <td>${row.AQI_official}</td>
                    <td>${row.AQI_category}</td>
                    <td>${row["PM2.5"]}</td>
                    <td>${row.PM10}</td>
                    <td>${row.last_update}</td>
                    <td><button>View</button></td>
                </tr>
            `;
        });

    } catch (error) {
        console.error("Error loading readings:", error);
    }
}

// ---------------------------
// Fetch City Stats
// ---------------------------
async function loadCityStats() {
    try {
        const res = await fetch(`${API_BASE}/city-stats`);
        const data = await res.json();

        const container = document.getElementById("city-grid");
        container.innerHTML = "";

        data.forEach(city => {
            container.innerHTML += `
                <div class="city-card">
                    <div class="city-header">
                        <h3 class="city-name">${city.city}</h3>
                        <span class="city-aqi">${city.avg_aqi.toFixed(1)}</span>
                    </div>
                    <div class="city-pollutants">
                        <div class="pollutant-item"><span>PM2.5</span><span>${city.avg_pm25.toFixed(1)}</span></div>
                        <div class="pollutant-item"><span>PM10</span><span>${city.avg_pm10.toFixed(1)}</span></div>
                        <div class="pollutant-item"><span>NO₂</span><span>${city.avg_no2.toFixed(1)}</span></div>
                    </div>
                </div>
            `;
        });

    } catch (error) {
        console.error("Error loading city stats:", error);
    }
}

// ---------------------------
// Fetch Anomalies
// ---------------------------
async function loadAnomalies() {
    try {
        const res = await fetch(`${API_BASE}/anomalies`);
        const data = await res.json();

        const tbody = document.getElementById("anomaly-tbody");
        tbody.innerHTML = "";

        data.forEach(a => {
            tbody.innerHTML += `
                <tr>
                    <td>${a.last_update}</td>
                    <td>${a.city}</td>
                    <td>${a.station}</td>
                    <td>${a.AQI_official}</td>
                    <td>${a.iforest_anomaly ? "Isolation Forest" : ""}${a.z_anomaly ? " Z-score" : ""}</td>
                    <td>${a.zscore ?? "-"}</td>
                    <td><button>Details</button></td>
                </tr>
            `;
        });

    } catch (error) {
        console.error("Error loading anomalies:", error);
    }
}

// ---------------------------
// Initialize Dashboard
// ---------------------------
loadRecentReadings();
loadCityStats();
loadAnomalies();
