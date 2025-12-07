// dashboard/static/scripts.js

const API_BASE = "/api";

// Global data storage
let baseRows = [];          // From realtime or recent-readings
let predictedRows = [];     // baseRows + prediction field
let cityStats = [];
let realtimeData = [];
let allCities = [];

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
        'Good': '#00C853',
        'Satisfactory': '#FFD600',
        'Moderate': '#FF6D00',
        'Poor': '#DD2C00',
        'Very Poor': '#7B1FA2',
        'Severe': '#4A148C'
    };
    return colors[category] || '#757575';
}

function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.remove('active');
}

function formatDate(dateStr) {
    if (!dateStr) return '-';
    try {
        const date = new Date(dateStr);
        return date.toLocaleString('en-IN', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return '-';
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
        console.log("Loaded recent-readings:", data.length);
        return Array.isArray(data) ? data : [];
    } catch (err) {
        console.error("Failed to fetch recent-readings:", err);
        return [];
    }
}

async function fetchCityStats() {
    try {
        const res = await fetch(`${API_BASE}/city-stats`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        cityStats = Array.isArray(data) ? data : [];
        console.log("Loaded city-stats:", cityStats.length);
        return cityStats;
    } catch (err) {
        console.error("Failed to fetch city-stats:", err);
        cityStats = [];
        return [];
    }
}

async function fetchRealtime() {
    try {
        const res = await fetch(`${API_BASE}/realtime`);
        if (!res.ok) {
            console.log("Realtime not available, status:", res.status);
            return [];
        }
        const data = await res.json();
        realtimeData = Array.isArray(data) ? data : [];
        console.log("Loaded realtime rows:", realtimeData.length);
        return realtimeData;
    } catch (err) {
        console.error("Failed to fetch realtime:", err);
        realtimeData = [];
        return [];
    }
}

async function predictAQI(payload) {
    try {
        const res = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!res.ok) {
            console.warn("Predict HTTP", res.status);
            return { anomaly: null, unsafe_prob: null, unsafe_label: null };
        }
        return await res.json();
    } catch (err) {
        console.error("Prediction error:", err);
        return { anomaly: null, unsafe_prob: null, unsafe_label: null };
    }
}

// Uses realtime if available, else recent-readings.
// For each row (limited) calls /api/predict and builds predictedRows.
async function fetchRealtimeAndPredict() {
    // choose base data
    const realtime = await fetchRealtime();
    if (realtime.length > 0) {
        baseRows = realtime;
    } else {
        baseRows = await fetchRecentReadings();
    }

    if (!baseRows.length) {
        predictedRows = [];
        return;
    }

    // Limit number of prediction calls for performance
    const subset = baseRows.slice(0, 200);

    const withPredictions = await Promise.all(
        subset.map(async (r) => {
            const lastUpdate = r.last_update || r.datetime || null;
            const d = lastUpdate ? new Date(lastUpdate) : new Date();
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

            const prediction = await predictAQI(payload);
            return { ...r, prediction };
        })
    );

    predictedRows = withPredictions;
}

// ============================================
// RENDER FUNCTIONS
// ============================================

function renderStatsOverview() {
    const stats = {
        good: 0,
        satisfactory: 0,
        moderate: 0,
        poor: 0,
        veryPoor: 0,
        severe: 0,
        unsafeCount: 0,
        anomalyCount: 0
    };

    // Use cityStats for city-level categories
    if (cityStats.length > 0) {
        cityStats.forEach(city => {
            const cat = getAQICategory(city.avg_aqi);
            switch (cat) {
                case 'Good': stats.good++; break;
                case 'Satisfactory': stats.satisfactory++; break;
                case 'Moderate': stats.moderate++; break;
                case 'Poor': stats.poor++; break;
                case 'Very Poor': stats.veryPoor++; break;
                case 'Severe': stats.severe++; break;
            }
        });
    }

    // Use predictions for ML-specific stats
    if (predictedRows.length > 0) {
        predictedRows.forEach(row => {
            const p = row.prediction || {};
            if (p.unsafe_label === 1) stats.unsafeCount++;
            if (p.anomaly === true) stats.anomalyCount++;
        });
    }

    document.getElementById("stat-good").textContent = stats.good;
    document.getElementById("stat-moderate").textContent = stats.satisfactory + stats.moderate;
    document.getElementById("stat-poor").textContent = stats.poor + stats.veryPoor;
    document.getElementById("stat-severe").textContent = stats.severe;

    document.getElementById("stat-ml-unsafe").textContent = stats.unsafeCount;
    document.getElementById("stat-ml-anomaly").textContent = stats.anomalyCount;
}

function renderCityCards() {
    const grid = document.getElementById("cityGrid");
    if (!grid) return;

    let cityData = [];

    if (cityStats.length > 0) {
        cityData = cityStats.slice(0, 6).map(city => ({
            name: city.city,
            state: 'India',
            aqi: Math.round(city.avg_aqi || 0),
            pm25: Math.round(city.avg_pm25 || 0),
            pm10: Math.round(city.avg_pm10 || 0),
            no2: Math.round(city.avg_no2 || 0)
        }));
    } else if (baseRows.length > 0) {
        const map = {};
        baseRows.forEach(r => {
            if (!r.city) return;
            if (!map[r.city]) {
                map[r.city] = {
                    name: r.city,
                    state: 'India',
                    aqi: r.AQI_official || r["PM2.5"] || 0,
                    pm25: r["PM2.5"] || 0,
                    pm10: r["PM10"] || 0,
                    no2: r["NO2"] || 0
                };
            }
        });
        cityData = Object.values(map).slice(0, 6);
    }

    if (!cityData.length) {
        grid.innerHTML = '<p style="text-align: center; color: #64748B; grid-column: 1/-1;">No city data available. Click "Refresh" to load data.</p>';
        return;
    }

    grid.innerHTML = cityData.map(city => {
        const category = getAQICategory(city.aqi);
        return `
            <div class="city-card" onclick="viewCityDetails('${encodeURIComponent(city.name)}')">
                <div class="city-header">
                    <div class="city-info">
                        <h3>${city.name}</h3>
                        <p>${city.state}</p>
                    </div>
                    <div class="city-aqi">
                        <div class="city-aqi-value" style="color: ${getCategoryColor(category)}">${city.aqi}</div>
                        <div class="city-aqi-label" style="color: ${getCategoryColor(category)}">${category}</div>
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
    }).join('');
}

function renderRecentTable() {
    const tbody = document.getElementById("tableBody");
    if (!tbody) return;

    const rows = predictedRows.length > 0 ? predictedRows : baseRows.slice(0, 50);

    if (!rows.length) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: #64748B;">No data available. Click "Refresh" to load data.</td></tr>';
        return;
    }

    tbody.innerHTML = rows.slice(0, 50).map(row => {
        const aqi = row.AQI_official || row["PM2.5"] || 0;
        const category = getAQICategory(aqi);
        const categoryClass = category.toLowerCase().replace(' ', '-');

        const p = row.prediction || {};
        const unsafeLabel = p.unsafe_label === 1 ? "Unsafe" : "Safe";
        const unsafeProb = p.unsafe_prob != null ? (p.unsafe_prob * 100).toFixed(0) + "%" : "-";
        const anomaly = p.anomaly === true ? "Yes" : "No";

        return `
            <tr>
                <td><strong>${row.city || 'Unknown'}</strong></td>
                <td>${row.station || row.location || 'N/A'}</td>
                <td><strong style="color: ${getCategoryColor(category)}">${Math.round(aqi)}</strong></td>
                <td><span class="badge ${categoryClass}">${category}</span></td>
                <td>${Math.round(row["PM2.5"] || 0)}</td>
                <td>${Math.round(row["PM10"] || 0)}</td>
                <td>${formatDate(row.last_update)}</td>
                <td>${unsafeLabel} ${unsafeProb !== "-" ? `(${unsafeProb})` : ""}</td>
                <td>${anomaly}</td>
            </tr>
        `;
    }).join('');
}

function generateForecast(type) {
    const grid = document.getElementById('forecastGrid');
    if (!grid) return;

    let num, labels;
    if (type === "7day") {
        num = 7;
        labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
    } else if (type === "24hr") {
        num = 8;
        labels = ['Now','3h','6h','9h','12h','15h','18h','21h'];
    } else {
        num = 8;
        labels = ['6h','12h','18h','24h','30h','36h','42h','48h'];
    }

    let avgAQI = 150;
    if (cityStats.length > 0) {
        const s = cityStats.reduce((sum, c) => sum + (c.avg_aqi || 0), 0);
        avgAQI = s / cityStats.length;
    } else if (baseRows.length > 0) {
        const s = baseRows.reduce((sum, r) => sum + (r.AQI_official || r["PM2.5"] || 0), 0);
        avgAQI = s / baseRows.length;
    }

    const forecasts = Array.from({ length: num }, (_, i) => {
        const variation = (Math.random() - 0.5) * 80;
        const aqi = Math.max(20, Math.round(avgAQI + variation));
        const cat = getAQICategory(aqi);
        return {
            label: labels[i] || `Step ${i+1}`,
            aqi,
            category: cat,
            pm25: Math.round(aqi * 0.5),
            pm10: Math.round(aqi * 0.8)
        };
    });

    grid.innerHTML = forecasts.map(f => `
        <div class="forecast-card">
            <div class="forecast-day">${f.label}</div>
            <div class="forecast-aqi" style="color: ${getCategoryColor(f.category)}">${f.aqi}</div>
            <span class="forecast-category badge ${f.category.toLowerCase().replace(' ', '-')}">${f.category}</span>
            <div class="forecast-details">
                <div>PM2.5: ${f.pm25}</div>
                <div>PM10: ${f.pm10}</div>
            </div>
        </div>
    `).join('');
}

function switchForecast(type) {
    document.querySelectorAll('.time-btn').forEach(btn => btn.classList.remove('active'));
    if (event && event.target) {
        event.target.classList.add('active');
    }
    generateForecast(type);
}

// ============================================
// SEARCH
// ============================================

function setupSearchAutocomplete() {
    const input = document.getElementById('citySearch');
    const suggestions = document.getElementById('suggestions');
    if (!input || !suggestions) return;

    input.addEventListener('input', (e) => {
        const value = e.target.value.toLowerCase();
        if (!value) {
            suggestions.classList.remove('active');
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
            suggestions.classList.remove('active');
            return;
        }

        suggestions.innerHTML = matches.map(city =>
            `<div class="suggestion-item" onclick="selectCity('${encodeURIComponent(city)}')">${city}</div>`
        ).join('');
        suggestions.classList.add('active');
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            suggestions.classList.remove('active');
        }
    });
}

function selectCity(city) {
    const decoded = decodeURIComponent(city);
    const input = document.getElementById('citySearch');
    if (input) input.value = decoded;
    const suggestions = document.getElementById('suggestions');
    if (suggestions) suggestions.classList.remove('active');
    viewCityDetails(decoded);
}

function searchCity() {
    const input = document.getElementById('citySearch');
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
// REFRESH / INIT
// ============================================

async function refreshData() {
    const btn = event?.target?.closest('button');
    if (btn) btn.disabled = true;
    showLoading();
    try {
        await fetchCityStats();
        await fetchRealtimeAndPredict();
        renderStatsOverview();
        renderCityCards();
        renderRecentTable();
        generateForecast('7day');
        setupSearchAutocomplete();
        console.log("✅ Manual refresh complete");
    } catch (err) {
        console.error("Refresh failed:", err);
        alert("Failed to refresh data. Check backend and try again.");
    } finally {
        hideLoading();
        if (btn) btn.disabled = false;
    }
}

async function initDashboard() {
    showLoading();
    try {
        console.log("🚀 Initializing dashboard...");
        await fetchCityStats();
        await fetchRealtimeAndPredict();
        renderStatsOverview();
        renderCityCards();
        renderRecentTable();
        generateForecast('7day');
        setupSearchAutocomplete();
        console.log("✅ Dashboard initialized");

        setInterval(async () => {
            console.log("🔄 Auto-refreshing base + predictions...");
            await fetchCityStats();
            await fetchRealtimeAndPredict();
            renderStatsOverview();
            renderCityCards();
            renderRecentTable();
        }, 300000); // 5 min
    } catch (err) {
        console.error("Dashboard init failed:", err);
    } finally {
        hideLoading();
    }
}

document.addEventListener("DOMContentLoaded", initDashboard);
