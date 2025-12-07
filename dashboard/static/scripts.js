// dashboard/static/scripts.js

const API_BASE = "/api";

// Global state
let baseRows = [];
let predictedRows = [];
let cityStats = [];
let realtimeData = [];
let allCities = [];
let modelMetrics = {};

// ============================================
// HELPERS
// ============================================

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

function getCategoryColor(category) {
    const colors = {
        "Good": "#00C853",
        "Satisfactory": "#FFD600",
        "Moderate": "#FF6D00",
        "Poor": "#DD2C00",
        "Very Poor": "#7B1FA2",
        "Severe": "#4A148C"
    };
    return colors[category] || "#888";
}

function showLoading() {
    const o = document.getElementById("loadingOverlay");
    if (o) o.classList.add("active");
}
function hideLoading() {
    const o = document.getElementById("loadingOverlay");
    if (o) o.classList.remove("active");
}

function formatDate(dateStr) {
    if (!dateStr) return "-";
    try {
        return new Date(dateStr).toLocaleString("en-IN", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit"
        });
    } catch {
        return "-";
    }
}

// ============================================
// API CALLS
// ============================================

async function fetchRecentReadings() {
    try {
        const res = await fetch(`${API_BASE}/recent-readings`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return Array.isArray(data) ? data : [];
    } catch (err) {
        console.error("recent-readings failed:", err);
        return [];
    }
}

async function fetchCityStats() {
    try {
        const res = await fetch(`${API_BASE}/city-stats`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cityStats = Array.isArray(data) ? data : [];
        return cityStats;
    } catch (err) {
        console.error("city-stats failed:", err);
        cityStats = [];
        return [];
    }
}

async function fetchRealtime() {
    try {
        const res = await fetch(`${API_BASE}/realtime`);
        if (!res.ok) {
            console.log("Realtime not available:", res.status);
            return [];
        }
        const data = await res.json();
        realtimeData = Array.isArray(data) ? data : [];
        return realtimeData;
    } catch (err) {
        console.error("realtime failed:", err);
        realtimeData = [];
        return [];
    }
}

// -------- MODEL PREDICTION --------
async function predictAQI(payload) {
    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            console.warn("predict HTTP", res.status);
            return {};
        }
        return await res.json();
    } catch (err) {
        console.error("predict error:", err);
        return {};
    }
}

// -------- MODEL METRICS --------
// -------- MODEL METRICS --------
async function fetchModelMetrics() {
    try {
        const res = await fetch(`${API_BASE}/model-metrics`);
        if (!res.ok) {
            console.log("model-metrics HTTP", res.status);
            return {};
        }

        modelMetrics = await res.json();

        // Extract raw metrics
        const rfRaw = modelMetrics.rf_accuracy;
        const xgbClfRaw = modelMetrics.xgb_clf_accuracy;
        const xgbMaeRaw = modelMetrics.xgb_reg_mae;
        const gbMaeRaw = modelMetrics.gb_reg_mae;

        // ===== FORMAT VALUES =====
        const rfAcc = rfRaw != null ? (rfRaw * 100).toFixed(0) + "%" : "-";
        const xgbClfAcc = xgbClfRaw != null ? (xgbClfRaw * 100).toFixed(0) + "%" : "-";

        const xgbMae = xgbMaeRaw != null ? xgbMaeRaw.toFixed(2) + " AQI" : "-";
        const gbMae = gbMaeRaw != null ? gbMaeRaw.toFixed(2) + " AQI" : "-";

        // ===== UPDATE UI =====
        const rfAccEl = document.getElementById("rf-acc");
        const xgbClfEl = document.getElementById("xgb-clf-acc");
        const xgbMaeEl = document.getElementById("xgb-reg-mae");
        const gbMaeEl = document.getElementById("gb-reg-mae");

        if (rfAccEl) rfAccEl.textContent = rfAcc;
        if (xgbClfEl) xgbClfEl.textContent = xgbClfAcc;
        if (xgbMaeEl) xgbMaeEl.textContent = xgbMae;
        if (gbMaeEl) gbMaeEl.textContent = gbMae;

        return modelMetrics;
    } catch (err) {
        console.error("model-metrics error:", err);
        return {};
    }
}


// ============================================
// REALTIME + PREDICTION PIPELINE
// ============================================

async function fetchRealtimeAndPredict() {
    const realtime = await fetchRealtime();
    baseRows = realtime.length ? realtime : await fetchRecentReadings();

    if (!baseRows.length) {
        predictedRows = [];
        return;
    }

    const subset = baseRows.slice(0, 200);

    const results = await Promise.all(
        subset.map(async (r) => {
            const d = new Date(r.last_update || r.datetime || new Date());
            const payload = {
                "PM2.5": r["PM2.5"] ?? r.pm25 ?? null,
                "PM10": r["PM10"] ?? r.pm10 ?? null,
                "NO2": r["NO2"] ?? r.no2 ?? null,
                "SO2": r["SO2"] ?? r.so2 ?? null,
                "OZONE": r["OZONE"] ?? r.o3 ?? null,
                "CO": r["CO"] ?? r.co ?? null,
                "NH3": r["NH3"] ?? r.nh3 ?? null,
                "hour_of_day": d.getHours(),
                "day_of_week": d.getDay()
            };

            const pred = await predictAQI(payload);

            return {
                ...r,
                prediction: pred,
                rf_pred: pred.rf_pred,
                xgb_clf_pred: pred.xgb_clf_pred,
                xgb_reg_pred: pred.xgb_reg_pred,
                gb_reg_pred: pred.gb_reg_pred
            };
        })
    );

    predictedRows = results;
}

// ============================================
// RENDER UI
// ============================================

function renderStatsOverview() {
    const stats = {
        good: 0,
        satisfactory: 0,
        moderate: 0,
        poor: 0,
        veryPoor: 0,
        severe: 0,
        mlUnsafe: 0,
        mlAnomaly: 0
    };

    // City-level category distribution
    cityStats.forEach(c => {
        const cat = getAQICategory(c.avg_aqi);
        switch (cat) {
            case "Good": stats.good++; break;
            case "Satisfactory": stats.satisfactory++; break;
            case "Moderate": stats.moderate++; break;
            case "Poor": stats.poor++; break;
            case "Very Poor": stats.veryPoor++; break;
            case "Severe": stats.severe++; break;
        }
    });

    // ML insights from prediction API
    predictedRows.forEach(r => {
        const p = r.prediction || {};
        if (p.unsafe_label === 1) stats.mlUnsafe++;
        if (p.anomaly === true) stats.mlAnomaly++;
    });

    const g = document.getElementById("stat-good");
    const m = document.getElementById("stat-moderate");
    const p = document.getElementById("stat-poor");
    const s = document.getElementById("stat-severe");
    const mlU = document.getElementById("stat-ml-unsafe");
    const mlA = document.getElementById("stat-ml-anomaly");

    if (g) g.textContent = stats.good;
    if (m) m.textContent = stats.satisfactory + stats.moderate;
    if (p) p.textContent = stats.poor + stats.veryPoor;
    if (s) s.textContent = stats.severe;
    if (mlU) mlU.textContent = stats.mlUnsafe;
    if (mlA) mlA.textContent = stats.mlAnomaly;
}

// ------------------ CITY CARDS ------------------
function renderCityCards() {
    const grid = document.getElementById("cityGrid");
    if (!grid) return;

    let cities = [];

    if (cityStats.length) {
        cities = cityStats.map(c => ({
            name: c.city,
            state: "India",
            aqi: Math.round(c.avg_aqi || 0),
            pm25: Math.round(c.avg_pm25 || 0),
            pm10: Math.round(c.avg_pm10 || 0),
            no2: Math.round(c.avg_no2 || 0)
        }));
    }

    if (!cities.length) {
        grid.innerHTML = '<p style="text-align:center; color:#64748B; grid-column:1/-1;">No city data available.</p>';
        return;
    }

    grid.innerHTML = cities.slice(0, 6).map(city => {
        const category = getAQICategory(city.aqi);
        const color = getCategoryColor(category);

        return `
            <div class="city-card" onclick="viewCityDetails('${encodeURIComponent(city.name)}')">
                <div class="city-header">
                    <div class="city-info">
                        <h3>${city.name}</h3>
                        <p>${city.state}</p>
                    </div>
                    <div class="city-aqi">
                        <div class="city-aqi-value" style="color:${color}">${city.aqi}</div>
                        <div class="city-aqi-label" style="color:${color}">${category}</div>
                    </div>
                </div>
                <div class="pollutant-list">
                    <div class="pollutant">
                        <div class="pollutant-name">PM2.5</div>
                        <div class="pollutant-value">${city.pm25}</div>
                    </div>
                    <div class="pollutant">
                        <div class="pollutant-name">PM10</div>
                        <div class="pollutant-value">${city.pm10}</div>
                    </div>
                    <div class="pollutant">
                        <div class="pollutant-name">NO₂</div>
                        <div class="pollutant-value">${city.no2}</div>
                    </div>
                </div>
            </div>
        `;
    }).join("");
}

// ------------------ RECENT TABLE ------------------
function renderRecentTable() {
    const tbody = document.getElementById("tableBody");
    if (!tbody) return;

    const rows = predictedRows.length ? predictedRows : baseRows;
    if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color:#64748B;">No data available.</td></tr>`;
        return;
    }

    tbody.innerHTML = rows.slice(0, 50).map(r => {
        const aqi = r.AQI_official || r["PM2.5"] || 0;
        const cat = getAQICategory(aqi);
        const catClass = cat.toLowerCase().replace(" ", "-");

        const p = r.prediction || {};
        const unsafeLabel = p.unsafe_label === 1 ? "Unsafe" : "Safe";
        const unsafeProb = (p.unsafe_prob != null)
            ? (p.unsafe_prob * 100).toFixed(0) + "%"
            : "-";
        const anomaly = p.anomaly === true ? "Yes" : "No";

        return `
            <tr>
                <td><strong>${r.city || "-"}</strong></td>
                <td>${r.station || r.location || "N/A"}</td>
                <td><strong style="color:${getCategoryColor(cat)}">${Math.round(aqi)}</strong></td>
                <td><span class="badge ${catClass}">${cat}</span></td>
                <td>${Math.round(r["PM2.5"] || 0)}</td>
                <td>${Math.round(r["PM10"] || 0)}</td>
                <td>${formatDate(r.last_update)}</td>
                <td>${unsafeLabel}${unsafeProb !== "-" ? ` (${unsafeProb})` : ""}</td>
                <td>${anomaly}</td>
            </tr>
        `;
    }).join("");
}

// ------------------ MODEL COMPARISON ------------------
function renderModelComparison() {
    const tbody = document.getElementById("modelCompareBody");
    if (!tbody) return;

    if (!predictedRows.length) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:#64748B;">No prediction data available.</td></tr>`;
        return;
    }

    tbody.innerHTML = predictedRows.slice(0, 50).map(r => `
        <tr>
            <td>${r.city || "-"}</td>
            <td>${Math.round(r["PM2.5"] || 0)}</td>
            <td>${r.rf_pred != null ? r.rf_pred : "-"}</td>
            <td>${r.xgb_clf_pred != null ? r.xgb_clf_pred : "-"}</td>
            <td>${r.xgb_reg_pred != null ? r.xgb_reg_pred.toFixed(1) : "-"}</td>
            <td>${r.gb_reg_pred != null ? r.gb_reg_pred.toFixed(1) : "-"}</td>
        </tr>
    `).join("");
}

// ============================================
// FORECAST (CARD-BASED)
// ============================================

function generateForecast(type) {
    const grid = document.getElementById("forecastGrid");
    if (!grid) return;

    let num, labels;
    if (type === "7day") {
        num = 7;
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    } else if (type === "24hr") {
        num = 8;
        labels = ["Now", "3h", "6h", "9h", "12h", "15h", "18h", "21h"];
    } else {
        num = 8;
        labels = ["6h", "12h", "18h", "24h", "30h", "36h", "42h", "48h"];
    }

    let avgAQI = 150;
    if (cityStats.length > 0) {
        const s = cityStats.reduce((sum, c) => sum + (c.avg_aqi || 0), 0);
        avgAQI = s / cityStats.length;
    } else if (baseRows.length > 0) {
        const s = baseRows.reduce(
            (sum, r) => sum + (r.AQI_official || r["PM2.5"] || 0),
            0
        );
        avgAQI = s / baseRows.length;
    }

    const forecasts = Array.from({ length: num }, (_, i) => {
        const variation = (Math.random() - 0.5) * 80;
        const aqi = Math.max(20, Math.round(avgAQI + variation));
        const category = getAQICategory(aqi);
        return {
            label: labels[i] || `Step ${i + 1}`,
            aqi,
            category,
            pm25: Math.round(aqi * 0.5),
            pm10: Math.round(aqi * 0.8)
        };
    });

    grid.innerHTML = forecasts.map(f => `
        <div class="forecast-card">
            <div class="forecast-day">${f.label}</div>
            <div class="forecast-aqi" style="color:${getCategoryColor(f.category)}">${f.aqi}</div>
            <span class="forecast-category badge ${f.category.toLowerCase().replace(" ", "-")}">
                ${f.category}
            </span>
            <div class="forecast-details">
                <div>PM2.5: ${f.pm25}</div>
                <div>PM10: ${f.pm10}</div>
            </div>
        </div>
    `).join("");
}

function switchForecast(type, btn) {
    const buttons = document.querySelectorAll(".time-btn");
    buttons.forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    generateForecast(type);
}

// ============================================
// SEARCH + CITY NAVIGATION
// ============================================

function setupSearchAutocomplete() {
    const input = document.getElementById("citySearch");
    const list = document.getElementById("suggestions");
    if (!input || !list) return;

    input.addEventListener("input", (e) => {
        const value = e.target.value.toLowerCase();
        if (!value) {
            list.classList.remove("active");
            return;
        }

        const fromStats = cityStats.map(c => c.city);
        const fromBase = baseRows.map(r => r.city);
        const fromReal = realtimeData.map(r => r.city);

        allCities = [...new Set([...fromStats, ...fromBase, ...fromReal])].filter(Boolean);

        const matches = allCities
            .filter(c => c.toLowerCase().includes(value))
            .slice(0, 10);

        if (!matches.length) {
            list.classList.remove("active");
            return;
        }

        list.innerHTML = matches.map(city =>
            `<div class="suggestion-item" onclick="selectCity('${encodeURIComponent(city)}')">${city}</div>`
        ).join("");

        list.classList.add("active");
    });

    document.addEventListener("click", (e) => {
        if (!e.target.closest(".search-container")) {
            list.classList.remove("active");
        }
    });
}

function selectCity(city) {
    const decoded = decodeURIComponent(city);
    const input = document.getElementById("citySearch");
    const list = document.getElementById("suggestions");
    if (input) input.value = decoded;
    if (list) list.classList.remove("active");
    viewCityDetails(decoded);
}

function searchCity() {
    const input = document.getElementById("citySearch");
    if (!input) return;
    const city = input.value.trim();
    if (city) {
        viewCityDetails(city);
    }
}

function viewCityDetails(city) {
    window.location.href = `/city/${encodeURIComponent(city)}`;
}

// ============================================
// PAGE INIT
// ============================================

async function refreshData() {
    showLoading();
    try {
        await fetchCityStats();
        await fetchRealtimeAndPredict();
        await fetchModelMetrics();

        renderStatsOverview();
        renderCityCards();
        renderRecentTable();
        renderModelComparison();
        generateForecast("7day");
        setupSearchAutocomplete();
    } catch (err) {
        console.error("Manual refresh failed:", err);
        alert("Failed to refresh data. Check backend and try again.");
    } finally {
        hideLoading();
    }
}

async function initDashboard() {
    showLoading();
    try {
        console.log("🚀 Initializing dashboard...");
        await fetchCityStats();
        await fetchRealtimeAndPredict();
        await fetchModelMetrics();

        renderStatsOverview();
        renderCityCards();
        renderRecentTable();
        renderModelComparison();
        generateForecast("7day");
        setupSearchAutocomplete();

        // Auto-refresh every 5 minutes
        setInterval(async () => {
            console.log("🔄 Auto refresh...");
            await fetchCityStats();
            await fetchRealtimeAndPredict();
            await fetchModelMetrics();
            renderStatsOverview();
            renderCityCards();
            renderRecentTable();
            renderModelComparison();
            generateForecast("7day");
        }, 300000);
    } catch (err) {
        console.error("Dashboard init failed:", err);
    } finally {
        hideLoading();
    }
}

document.addEventListener("DOMContentLoaded", initDashboard);
