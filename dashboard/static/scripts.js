// dashboard/static/scripts.js
// SMART AQI FRONTEND JAVASCRIPT (complete)
// Uses relative API base so it works when served by Flask
const API_BASE = "/api";

// Global memory
let aqiData = [];
let cityStats = [];
let anomalyData = [];

// Helper category
function getAQICategory(aqi) {
    if (!aqi && aqi !== 0) return "Unknown";
    aqi = Number(aqi);
    if (aqi <= 50) return "Good";
    if (aqi <= 100) return "Satisfactory";
    if (aqi <= 200) return "Moderate";
    if (aqi <= 300) return "Poor";
    if (aqi <= 400) return "Very Poor";
    return "Severe";
}

// ---------- 1) Ingest (trigger server to fetch OpenAQ) ----------
async function triggerIngest(country="IN", limit=500) {
    try {
        const res = await fetch(`${API_BASE}/ingest`, {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({country, limit})
        });
        const j = await res.json();
        console.log("Ingest response:", j);
        return j;
    } catch (err) {
        console.warn("Ingest failed", err);
        return null;
    }
}

// ---------- 2) Fetch realtime and predict ----------
async function fetchRealtimeAndPredict() {
    try {
        const res = await fetch(`${API_BASE}/realtime`);
        if (!res.ok) {
            console.warn("No realtime data (status):", res.status);
            // still attempt to load static dataset & anomalies
            await loadCityStats();
            await loadAnomalies();
            return;
        }
        const rows = await res.json();
        if (!Array.isArray(rows) || rows.length === 0) {
            console.warn("Realtime empty");
            await loadCityStats();
            await loadAnomalies();
            return;
        }

        // limit to last N rows to avoid spamming predict endpoint
        const limited = rows.slice(-150);

        // call predict endpoint for each row (batch concurrency)
        const predictions = await Promise.all(limited.map(async (r) => {
            // build payload expected by backend
            const payload = {
                "PM2.5": r["PM2.5"] ?? r["pm25"] ?? null,
                "PM10": r["PM10"] ?? r["pm10"] ?? null,
                "NO2": r["NO2"] ?? r["no2"] ?? null,
                "SO2": r["SO2"] ?? r["so2"] ?? null,
                "OZONE": r["OZONE"] ?? r["o3"] ?? null,
                "CO": r["CO"] ?? r["co"] ?? null,
                "NH3": r["NH3"] ?? r["nh3"] ?? null,
                "hour_of_day": r.last_update ? new Date(r.last_update).getHours() : new Date().getHours(),
                "day_of_week": r.last_update ? new Date(r.last_update).getDay() : new Date().getDay()
            };

            const p = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify(payload)
            }).then(resp => resp.json())
              .catch(() => ({error: "prediction_failed"}));

            return {...r, prediction: p};
        }));

        // format for UI
        aqiData = predictions.map(x => ({
            city: x.city || x.city || "Unknown",
            station: x.station || x.location || "Station",
            AQI_official: x.AQI_official ?? x["PM2.5"] ?? null,
            PM25: x["PM2.5"],
            PM10: x["PM10"],
            NO2: x["NO2"],
            SO2: x["SO2"],
            OZONE: x["OZONE"],
            CO: x["CO"],
            last_update: x.last_update,
            iforest_anomaly: x.prediction?.anomaly ? 1 : 0,
            unsafe_prob: x.prediction?.unsafe_prob ?? null,
            unsafe_label: x.prediction?.unsafe_label ?? null,
            AQI_category: getAQICategory(x.AQI_official ?? 0)
        }));

        renderRecentTable();
        renderStatsOverview();

    } catch (err) {
        console.error("Realtime/predict error:", err);
    }
}

// ---------- 3) Load city stats (uses cleaned dataset) ----------
async function loadCityStats() {
    try {
        const res = await fetch(`${API_BASE}/city-stats`);
        const data = await res.json();
        cityStats = Array.isArray(data) ? data : [];
        renderCityGrid();
    } catch (err) {
        console.error("City stats error", err);
    }
}

// ---------- 4) Load anomalies (uses cleaned anomalies CSV) ----------
async function loadAnomalies() {
    try {
        const res = await fetch(`${API_BASE}/anomalies`);
        const data = await res.json();
        anomalyData = Array.isArray(data) ? data : [];
        renderAnomalyTable();
        renderAnomalyStats();
    } catch (err) {
        console.error("Anomalies load error", err);
    }
}

// ---------- UI renderers ----------
function renderStatsOverview() {
    const stats = {good:0, moderate:0, poor:0, severe:0};
    aqiData.forEach(r => {
        const cat = r.AQI_category;
        if (cat === "Good") stats.good++;
        else if (cat === "Satisfactory" || cat === "Moderate") stats.moderate++;
        else if (cat === "Poor" || cat === "Very Poor") stats.poor++;
        else if (cat === "Severe") stats.severe++;
    });
    document.getElementById("stat-good").textContent = stats.good;
    document.getElementById("stat-moderate").textContent = stats.moderate;
    document.getElementById("stat-poor").textContent = stats.poor;
    document.getElementById("stat-severe").textContent = stats.severe;
    document.getElementById("last-update-time").textContent = new Date().toLocaleString();
}

function renderRecentTable() {
    const tbody = document.getElementById("readings-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    aqiData.slice(0,40).forEach(r => {
        tbody.innerHTML += `
            <tr>
                <td>${r.city}</td>
                <td>${r.station}</td>
                <td>${r.AQI_official ?? '-'}</td>
                <td>${r.AQI_category}</td>
                <td>${r.PM25 ?? '-'}</td>
                <td>${r.PM10 ?? '-'}</td>
                <td>${r.last_update ?? '-'}</td>
                <td><button class="btn-icon" onclick="viewCityDetails('${encodeURIComponent(r.city)}')"><i class="fas fa-eye"></i></button></td>
            </tr>
        `;
    });
}

function renderCityGrid() {
    const grid = document.getElementById("city-grid");
    if (!grid) return;
    grid.innerHTML = "";
    cityStats.forEach(city => {
        grid.innerHTML += `
            <div class="city-card" onclick="viewCityDetails('${encodeURIComponent(city.city)}')">
                <div class="city-header">
                    <div class="city-name">${city.city}</div>
                    <span class="aqi-badge">${Number(city.avg_aqi).toFixed(0)}</span>
                </div>
                <div class="city-pollutants">
                    <div class="pollutant-item"><span>PM2.5</span><strong>${Number(city.avg_pm25).toFixed(1)}</strong></div>
                    <div class="pollutant-item"><span>PM10</span><strong>${Number(city.avg_pm10).toFixed(1)}</strong></div>
                    <div class="pollutant-item"><span>NO₂</span><strong>${Number(city.avg_no2).toFixed(1)}</strong></div>
                </div>
            </div>
        `;
    });
}

function renderAnomalyTable() {
    const tbody = document.getElementById("anomaly-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    anomalyData.slice(0,50).forEach(a => {
        const methods = `${a.iforest_anomaly ? "Isolation Forest":""} ${a.z_anomaly ? "Z-Score":""}`.trim();
        tbody.innerHTML += `
            <tr>
                <td>${a.last_update ?? '-'}</td>
                <td>${a.city ?? '-'}</td>
                <td>${a.station ?? '-'}</td>
                <td>${a.AQI_official ?? '-'}</td>
                <td>${methods || '-'}</td>
                <td>${a.zscore ?? '-'}</td>
                <td><button class="btn-icon" onclick='viewAnomalyDetails(${JSON.stringify(a).replace(/"/g,"&quot;")})'><i class="fas fa-info-circle"></i></button></td>
            </tr>
        `;
    });
}

function renderAnomalyStats() {
    document.getElementById("anomaly-count").textContent = anomalyData.length;
    const ratio = aqiData.length ? ((anomalyData.length / aqiData.length) * 100).toFixed(1) : 0;
    document.getElementById("anomaly-rate").textContent = `${ratio}%`;
}

// ---------- small utilities & event handlers ----------
function viewCityDetails(city) {
    // navigate to city detail page (works when Flask route exists)
    const decoded = decodeURIComponent(city);
    window.location.href = `/city/${encodeURIComponent(decoded)}`;
}
function viewAnomalyDetails(row) {
    alert(`Anomaly details:\nCity: ${row.city}\nStation: ${row.station}\nAQI: ${row.AQI_official}`);
}

// ---------- init & auto-refresh ----------
async function initDashboard() {
    // initial load: try ingest -> then fetch realtime/predict and others
    await triggerIngest();            // on-demand fetch from OpenAQ (optional)
    await fetchRealtimeAndPredict();  // fetch realtime file + predict
    await loadCityStats();            // aggregated stats from cleaned dataset
    await loadAnomalies();            // anomalies from cleaned dataset

    // auto refresh every 5 minutes (300000ms). adjust as needed.
    setInterval(async () => {
        console.log("Auto-refreshing realtime/predictions...");
        await triggerIngest();          // optional - you can remove to avoid repeated ingest
        await fetchRealtimeAndPredict();
        await loadAnomalies();
    }, 300000);
}

// Wire refresh button if present
document.addEventListener("DOMContentLoaded", () => {
    const refreshBtn = document.getElementById("refresh-btn");
    if (refreshBtn) refreshBtn.addEventListener("click", async () => {
        refreshBtn.disabled = true;
        await triggerIngest();
        await fetchRealtimeAndPredict();
        await loadAnomalies();
        refreshBtn.disabled = false;
    });
    initDashboard();
});
