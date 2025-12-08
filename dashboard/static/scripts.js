// dashboard/static/scripts.js

const API_BASE = "/api";

// Global state
let baseRows = [];
let predictedRows = [];
let cityStats = [];
let realtimeData = [];
let allCities = [];
let modelMetrics = {};
let transformerForecast = [];
let transformerMode = "7day"; // default tab

// ============================================
// HELPERS
// ============================================

function getAQICategory(aqi) {
    if (aqi === null || aqi === undefined || Number.isNaN(aqi)) return "Unknown";
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
        "Severe": "#4A148C",
        "Unknown": "#64748B"
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

function formatForecastLabel(ts, mode) {
    if (!ts) return mode === "7day" ? "Day" : "T+";
    const d = new Date(ts);
    if (mode === "7day") {
        return d.toLocaleDateString("en-IN", { weekday: "short" });
    }
    return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}

// Bucketing helper for transformer forecast
function bucketTransformerForecast(targetCount) {
    const points = transformerForecast;
    const n = points.length;
    if (!n) return [];

    if (targetCount >= n) {
        // nothing to bucket, just return as-is
        return points.slice(0, targetCount);
    }

    const step = n / targetCount;
    const result = [];
    for (let i = 0; i < targetCount; i++) {
        const start = Math.floor(i * step);
        const end = Math.max(start + 1, Math.floor((i + 1) * step));
        const slice = points.slice(start, end);
        const avgAqi = slice.reduce((sum, p) => sum + p.aqi, 0) / slice.length;
        const ts = slice[0].timestamp;
        result.push({ aqi: avgAqi, timestamp: ts });
    }
    return result;
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
async function fetchModelMetrics() {
    try {
        const res = await fetch(`${API_BASE}/model-metrics`);
        if (!res.ok) {
            console.log("model-metrics HTTP", res.status);
            return {};
        }

        modelMetrics = await res.json();

        const rfRaw = modelMetrics.rf_accuracy;
        const xgbClfRaw = modelMetrics.xgb_clf_accuracy;
        const xgbMaeRaw = modelMetrics.xgb_reg_mae;
        const gbMaeRaw = modelMetrics.gb_reg_mae;

        const rfAcc = rfRaw != null ? (rfRaw * 100).toFixed(0) + "%" : "--";
        const xgbClfAcc = xgbClfRaw != null ? (xgbClfRaw * 100).toFixed(0) + "%" : "--";

        const xgbMae = xgbMaeRaw != null ? xgbMaeRaw.toFixed(2) + " AQI" : "--";
        const gbMae = gbMaeRaw != null ? gbMaeRaw.toFixed(2) + " AQI" : "--";

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

// -------- TRANSFORMER FORECAST --------
async function fetchTransformerForecast() {
    try {
        const res = await fetch(`${API_BASE}/transformer-forecast`);
        if (!res.ok) {
            console.log("transformer-forecast HTTP", res.status);
            transformerForecast = [];
            return [];
        }

        const data = await res.json();
        const raw = Array.isArray(data.forecast) ? data.forecast : [];

        transformerForecast = raw
            .map((p, idx) => {
                const aqiVal = (p.aqi === null || p.aqi === undefined || Number.isNaN(p.aqi))
                    ? null
                    : Number(p.aqi);
                return {
                    step: p.step ?? idx + 1,
                    aqi: aqiVal,
                    timestamp: p.timestamp || null
                };
            })
            .filter(p => p.aqi !== null);

        return transformerForecast;
    } catch (err) {
        console.error("transformer-forecast error:", err);
        transformerForecast = [];
        return [];
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
// TRANSFORMER FORECAST RENDERING
// ============================================

function renderTransformerForecast(mode) {
    transformerMode = mode;

    const grid = document.getElementById("forecastGrid");
    const titleEl = document.getElementById("transformer-summary-title");
    const subtitleEl = document.getElementById("transformer-summary-subtitle");
    const mainEl = document.getElementById("transformer-summary-main");
    const aqiEl = document.getElementById("transformer-summary-aqi");
    const catEl = document.getElementById("transformer-summary-category");
    const detailEl = document.getElementById("transformer-summary-detail");

    if (!grid) return;

    if (!transformerForecast.length) {
        grid.innerHTML = `<p style="text-align:center; color:#64748B; grid-column:1/-1;">
            No transformer forecast data available.
        </p>`;
        if (titleEl) titleEl.textContent = "Forecast unavailable";
        if (subtitleEl) subtitleEl.textContent = "Transformer model output is not available right now.";
        if (mainEl) mainEl.style.display = "none";
        return;
    }

    let targetCount;
    if (mode === "24hr") {
        targetCount = Math.min(24, transformerForecast.length);
    } else if (mode === "48hr") {
        targetCount = Math.min(16, transformerForecast.length); // grouped a bit coarser
    } else {
        targetCount = Math.min(7, transformerForecast.length);
    }

    const buckets = bucketTransformerForecast(targetCount);

    grid.innerHTML = buckets.map((f, idx) => {
        const aqiVal = Math.round(f.aqi);
        const cat = getAQICategory(aqiVal);
        const color = getCategoryColor(cat);
        const label = formatForecastLabel(f.timestamp, mode);

        return `
            <div class="forecast-card">
                <div class="forecast-day">${label}</div>
                <div class="forecast-aqi" style="color:${color}">
                    ${Number.isFinite(aqiVal) ? aqiVal : "-"}
                </div>
                <span class="forecast-category badge ${cat.toLowerCase().replace(" ", "-")}">
                    ${cat}
                </span>
                <div class="forecast-details">
                    <div>Step: ${idx + 1}</div>
                    <div>Mode: ${mode === "7day" ? "Daily" : "Hourly"}</div>
                </div>
            </div>
        `;
    }).join("");

    // Summary card from first bucket
    const first = buckets[0];
    const sumAqi = Math.round(first.aqi);
    const sumCat = getAQICategory(sumAqi);

    if (titleEl) {
        const modeLabel = mode === "7day" ? "coming days" :
            (mode === "48hr" ? "next 48 hours" : "next 24 hours");
        titleEl.textContent = `Predicted AQI – ${modeLabel}`;
    }
    if (subtitleEl) {
        subtitleEl.textContent = "Sequence-to-sequence Transformer forecast using past AQI_official values.";
    }
    if (mainEl) mainEl.style.display = "block";
    if (aqiEl) aqiEl.textContent = Number.isFinite(sumAqi) ? sumAqi : "--";
    if (catEl) {
        catEl.textContent = sumCat;
        catEl.style.color = getCategoryColor(sumCat);
    }
    if (detailEl) {
        detailEl.textContent = `Model suggests ${sumCat} air quality around an AQI of ~${Number.isFinite(sumAqi) ? sumAqi : "N/A"}.`;
    }
}

function switchTransformerForecast(type, btn) {
    const buttons = document.querySelectorAll(".time-btn");
    buttons.forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    renderTransformerForecast(type);
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
        await fetchTransformerForecast();

        renderStatsOverview();
        renderCityCards();
        renderRecentTable();
        renderModelComparison();
        renderTransformerForecast(transformerMode);
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
        await fetchTransformerForecast();

        renderStatsOverview();
        renderCityCards();
        renderRecentTable();
        renderModelComparison();
        renderTransformerForecast("7day");
        setupSearchAutocomplete();

        // Auto-refresh every 5 minutes
        setInterval(async () => {
            console.log("🔄 Auto refresh...");
            await fetchCityStats();
            await fetchRealtimeAndPredict();
            await fetchModelMetrics();
            await fetchTransformerForecast();
            renderStatsOverview();
            renderCityCards();
            renderRecentTable();
            renderModelComparison();
            renderTransformerForecast(transformerMode);
        }, 300000);
    } catch (err) {
        console.error("Dashboard init failed:", err);
    } finally {
        hideLoading();
    }
}

document.addEventListener("DOMContentLoaded", initDashboard);
