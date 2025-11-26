/**
 * SMART AQI MONITORING SYSTEM - MAIN JAVASCRIPT
 * Handles data fetching, visualization, and interactivity
 */

// ================================
// GLOBAL STATE
// ================================
let aqiData = [];
let anomalyData = [];
let pcaData = [];
let currentView = '2d';
let currentAnomalyFilter = 'both';

// ================================
// API ENDPOINTS (Update these to match your Flask backend)
// ================================
const API_BASE = 'http://localhost:5000/api';
const ENDPOINTS = {
    recentReadings: `${API_BASE}/recent-readings`,
    anomalies: `${API_BASE}/anomalies`,
    pcaData: `${API_BASE}/pca-data`,
    cityStats: `${API_BASE}/city-stats`,
    reports: `${API_BASE}/reports`,
    generateReport: `${API_BASE}/generate-report`
};

// ================================
// AQI CATEGORY HELPERS
// ================================
const AQI_CATEGORIES = {
    'Good': { color: '#00C853', range: [0, 50] },
    'Satisfactory': { color: '#FFD600', range: [51, 100] },
    'Moderate': { color: '#FF6D00', range: [101, 200] },
    'Poor': { color: '#DD2C00', range: [201, 300] },
    'Very Poor': { color: '#7B1FA2', range: [301, 400] },
    'Severe': { color: '#4A148C', range: [401, 500] }
};

function getAQICategory(aqi) {
    if (aqi <= 50) return 'Good';
    if (aqi <= 100) return 'Satisfactory';
    if (aqi <= 200) return 'Moderate';
    if (aqi <= 300) return 'Poor';
    if (aqi <= 400) return 'Very Poor';
    return 'Severe';
}

function getAQIColor(category) {
    return AQI_CATEGORIES[category]?.color || '#757575';
}

// ================================
// INITIALIZATION
// ================================
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
});

async function initializeApp() {
    showLoading();
    try {
        await loadAllData();
        renderDashboard();
        updateLastUpdateTime();
    } catch (error) {
        console.error('Initialization error:', error);
        showError('Failed to load data. Using sample data instead.');
        loadSampleData();
        renderDashboard();
    } finally {
        hideLoading();
    }
}

// ================================
// DATA LOADING
// ================================
async function loadAllData() {
    try {
        // Load data from backend API
        const [readings, anomalies, pca] = await Promise.all([
            fetchData(ENDPOINTS.recentReadings),
            fetchData(ENDPOINTS.anomalies),
            fetchData(ENDPOINTS.pcaData)
        ]);
        
        aqiData = readings || [];
        anomalyData = anomalies || [];
        pcaData = pca || [];
    } catch (error) {
        console.error('Data loading error:', error);
        throw error;
    }
}

async function fetchData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (error) {
        console.warn(`Failed to fetch from ${url}, using sample data`);
        return null;
    }
}

function loadSampleData() {
    // Sample data for demonstration
    const cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Pune'];
    const stations = ['Station A', 'Station B', 'Station C'];
    
    aqiData = Array.from({ length: 50 }, (_, i) => ({
        city: cities[Math.floor(Math.random() * cities.length)],
        station: stations[Math.floor(Math.random() * stations.length)],
        AQI_official: Math.floor(Math.random() * 400) + 20,
        PM25: Math.floor(Math.random() * 150) + 10,
        PM10: Math.floor(Math.random() * 200) + 20,
        NO2: Math.floor(Math.random() * 100) + 10,
        SO2: Math.floor(Math.random() * 80) + 5,
        OZONE: Math.floor(Math.random() * 150) + 10,
        CO: (Math.random() * 5 + 0.5).toFixed(1),
        last_update: new Date(Date.now() - Math.random() * 86400000).toISOString(),
        iforest_anomaly: Math.random() > 0.95 ? 1 : 0,
        z_anomaly: Math.random() > 0.95,
        z_score: (Math.random() * 6 - 3).toFixed(2)
    })).map(row => ({
        ...row,
        AQI_category: getAQICategory(row.AQI_official)
    }));

    // Generate anomaly data
    anomalyData = aqiData.filter(row => row.iforest_anomaly === 1 || row.z_anomaly);

    // Generate PCA data
    pcaData = aqiData.map(row => ({
        ...row,
        PC1: Math.random() * 4 - 2,
        PC2: Math.random() * 4 - 2,
        PC3: Math.random() * 4 - 2
    }));
}

// ================================
// DASHBOARD RENDERING
// ================================
function renderDashboard() {
    renderStats();
    renderRecentReadingsTable();
    renderAnomalySection();
    renderPCASection();
    renderCityGrid();
    renderReports();
}

// ================================
// STATS CARDS
// ================================
function renderStats() {
    const stats = {
        good: 0,
        moderate: 0,
        poor: 0,
        severe: 0
    };

    aqiData.forEach(row => {
        const cat = row.AQI_category.toLowerCase().replace(' ', '-');
        if (cat === 'good' || cat === 'satisfactory') stats.good++;
        else if (cat === 'moderate') stats.moderate++;
        else if (cat === 'poor' || cat === 'very-poor') stats.poor++;
        else if (cat === 'severe') stats.severe++;
    });

    document.getElementById('stat-good').textContent = stats.good;
    document.getElementById('stat-moderate').textContent = stats.moderate;
    document.getElementById('stat-poor').textContent = stats.poor;
    document.getElementById('stat-severe').textContent = stats.severe;
}

// ================================
// RECENT READINGS TABLE
// ================================
function renderRecentReadingsTable() {
    const tbody = document.getElementById('readings-tbody');
    const sortedData = [...aqiData].sort((a, b) => 
        new Date(b.last_update) - new Date(a.last_update)
    ).slice(0, 20);

    tbody.innerHTML = sortedData.map(row => `
        <tr>
            <td><strong>${row.city}</strong></td>
            <td>${row.station}</td>
            <td><strong style="color: ${getAQIColor(row.AQI_category)}">${row.AQI_official}</strong></td>
            <td><span class="aqi-badge ${row.AQI_category.toLowerCase().replace(' ', '-')}">${row.AQI_category}</span></td>
            <td>${row.PM25 || 'N/A'}</td>
            <td>${row.PM10 || 'N/A'}</td>
            <td>${formatDate(row.last_update)}</td>
            <td>
                <button class="btn-icon" onclick="viewCityDetails('${row.city}')" title="View Details">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        </tr>
    `).join('');
}

// ================================
// ANOMALY DETECTION SECTION
// ================================
function renderAnomalySection() {
    // Calculate anomaly stats
    const last24h = anomalyData.filter(row => {
        const diff = Date.now() - new Date(row.last_update).getTime();
        return diff < 86400000; // 24 hours
    });

    const anomalyRate = ((anomalyData.length / aqiData.length) * 100).toFixed(1);

    document.getElementById('anomaly-count').textContent = last24h.length;
    document.getElementById('anomaly-rate').textContent = `${anomalyRate}%`;

    renderAnomalyTable();
    renderAnomalyTimeline();
}

function renderAnomalyTable() {
    const tbody = document.getElementById('anomaly-tbody');
    const filtered = filterAnomalies(anomalyData);

    tbody.innerHTML = filtered.slice(0, 15).map(row => {
        const methods = [];
        if (row.iforest_anomaly === 1) methods.push('Isolation Forest');
        if (row.z_anomaly) methods.push('Z-Score');

        return `
            <tr>
                <td>${formatDate(row.last_update)}</td>
                <td><strong>${row.city}</strong></td>
                <td>${row.station}</td>
                <td><strong style="color: ${getAQIColor(row.AQI_category)}">${row.AQI_official}</strong></td>
                <td><span class="aqi-badge moderate">${methods.join(', ')}</span></td>
                <td>${row.z_score || 'N/A'}</td>
                <td>
                    <button class="btn-icon" onclick="viewAnomalyDetails(${JSON.stringify(row).replace(/"/g, '&quot;')})" title="Details">
                        <i class="fas fa-info-circle"></i>
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function filterAnomalies(data) {
    const filter = document.getElementById('anomaly-filter')?.value || 'all';
    
    if (filter === 'all') return data;
    if (filter === 'iforest') return data.filter(r => r.iforest_anomaly === 1);
    if (filter === 'zscore') return data.filter(r => r.z_anomaly);
    if (filter === 'both') return data.filter(r => r.iforest_anomaly === 1 && r.z_anomaly);
    
    return data;
}

function renderAnomalyTimeline() {
    // Group anomalies by date
    const grouped = {};
    anomalyData.forEach(row => {
        const date = new Date(row.last_update).toLocaleDateString();
        grouped[date] = (grouped[date] || 0) + 1;
    });

    const dates = Object.keys(grouped).sort();
    const counts = dates.map(d => grouped[d]);

    const ctx = document.getElementById('anomaly-timeline-chart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Anomalies Detected',
                data: counts,
                borderColor: '#FF6D00',
                backgroundColor: 'rgba(255, 109, 0, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}

// ================================
// PCA SECTION
// ================================
function renderPCASection() {
    renderPCAStats();
    renderPCAScatter();
}

function renderPCAStats() {
    // Calculate variance (simulated)
    document.getElementById('pc1-var').textContent = '45.2%';
    document.getElementById('pc2-var').textContent = '28.7%';
    document.getElementById('pc3-var').textContent = '15.3%';
    document.getElementById('cumulative-var').textContent = '89.2%';

    renderVarianceChart();
}

function renderVarianceChart() {
    const ctx = document.getElementById('variance-chart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['PC1', 'PC2', 'PC3'],
            datasets: [{
                label: 'Explained Variance',
                data: [45.2, 28.7, 15.3],
                backgroundColor: ['#1976D2', '#42A5F5', '#90CAF9']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    max: 50,
                    title: { display: true, text: 'Variance (%)' }
                }
            }
        }
    });
}

function renderPCAScatter() {
    if (!pcaData || pcaData.length === 0) return;

    const container = document.getElementById('pca-scatter');
    if (!container) return;

    if (currentView === '2d') {
        renderPCA2D();
    } else {
        renderPCA3D();
    }
}

function renderPCA2D() {
    const traces = {};
    
    pcaData.forEach(row => {
        const cat = row.AQI_category;
        if (!traces[cat]) {
            traces[cat] = {
                x: [],
                y: [],
                mode: 'markers',
                type: 'scatter',
                name: cat,
                marker: {
                    color: getAQIColor(cat),
                    size: 8,
                    opacity: 0.7
                }
            };
        }
        traces[cat].x.push(row.PC1);
        traces[cat].y.push(row.PC2);
    });

    const data = Object.values(traces);

    const layout = {
        xaxis: { title: 'PC1 (45.2% variance)' },
        yaxis: { title: 'PC2 (28.7% variance)' },
        hovermode: 'closest',
        showlegend: true,
        legend: { x: 1, y: 1 }
    };

    Plotly.newPlot('pca-scatter', data, layout, { responsive: true });
}

function renderPCA3D() {
    const traces = {};
    
    pcaData.forEach(row => {
        const cat = row.AQI_category;
        if (!traces[cat]) {
            traces[cat] = {
                x: [],
                y: [],
                z: [],
                mode: 'markers',
                type: 'scatter3d',
                name: cat,
                marker: {
                    color: getAQIColor(cat),
                    size: 4,
                    opacity: 0.7
                }
            };
        }
        traces[cat].x.push(row.PC1);
        traces[cat].y.push(row.PC2);
        traces[cat].z.push(row.PC3);
    });

    const data = Object.values(traces);

    const layout = {
        scene: {
            xaxis: { title: 'PC1' },
            yaxis: { title: 'PC2' },
            zaxis: { title: 'PC3' }
        },
        showlegend: true
    };

    Plotly.newPlot('pca-scatter', data, layout, { responsive: true });
}

// ================================
// CITY GRID
// ================================
function renderCityGrid() {
    const cityStats = {};
    
    aqiData.forEach(row => {
        if (!cityStats[row.city]) {
            cityStats[row.city] = {
                city: row.city,
                totalAQI: 0,
                count: 0,
                pm25: 0,
                pm10: 0,
                no2: 0
            };
        }
        cityStats[row.city].totalAQI += row.AQI_official;
        cityStats[row.city].count++;
        cityStats[row.city].pm25 += row.PM25 || 0;
        cityStats[row.city].pm10 += row.PM10 || 0;
        cityStats[row.city].no2 += row.NO2 || 0;
    });

    const cities = Object.values(cityStats).map(city => ({
        ...city,
        avgAQI: Math.round(city.totalAQI / city.count),
        avgPM25: Math.round(city.pm25 / city.count),
        avgPM10: Math.round(city.pm10 / city.count),
        avgNO2: Math.round(city.no2 / city.count)
    }));

    const grid = document.getElementById('city-grid');
    grid.innerHTML = cities.map(city => {
        const category = getAQICategory(city.avgAQI);
        return `
            <div class="city-card" onclick="viewCityDetails('${city.city}')">
                <div class="city-header">
                    <div class="city-name">${city.city}</div>
                    <span class="aqi-badge ${category.toLowerCase().replace(' ', '-')}">${category}</span>
                </div>
                <div class="city-aqi" style="color: ${getAQIColor(category)}">${city.avgAQI}</div>
                <div class="city-pollutants">
                    <div class="pollutant-item">
                        <span>PM2.5</span>
                        <strong>${city.avgPM25}</strong>
                    </div>
                    <div class="pollutant-item">
                        <span>PM10</span>
                        <strong>${city.avgPM10}</strong>
                    </div>
                    <div class="pollutant-item">
                        <span>NO2</span>
                        <strong>${city.avgNO2}</strong>
                    </div>
                    <div class="pollutant-item">
                        <span>Stations</span>
                        <strong>${city.count}</strong>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ================================
// REPORTS
// ================================
function renderReports() {
    const reportsList = document.getElementById('reports-list');
    const sampleReports = [
        { date: '2024-11-25', cities: 15, anomalies: 23, severe: 3 },
        { date: '2024-11-24', cities: 15, anomalies: 18, severe: 2 },
        { date: '2024-11-23', cities: 15, anomalies: 31, severe: 5 }
    ];

    reportsList.innerHTML = sampleReports.map(report => `
        <div class="report-item">
            <div class="report-info">
                <i class="fas fa-file-pdf"></i>
                <div class="report-details">
                    <h4>Daily Report - ${report.date}</h4>
                    <p>${report.cities} cities · ${report.anomalies} anomalies · ${report.severe} severe alerts</p>
                </div>
            </div>
            <div class="report-actions">
                <button class="btn-icon" title="Download">
                    <i class="fas fa-download"></i>
                </button>
                <button class="btn-icon" title="View">
                    <i class="fas fa-eye"></i>
                </button>
            </div>
        </div>
    `).join('');
}

// ================================
// EVENT LISTENERS
// ================================
function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('href');
            document.querySelector(target)?.scrollIntoView({ behavior: 'smooth' });
            
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Refresh button
    document.getElementById('refresh-btn')?.addEventListener('click', async () => {
        await initializeApp();
    });

    // PCA view toggle
    document.querySelectorAll('[data-view]').forEach(btn => {
        btn.addEventListener('click', () => {
            currentView = btn.dataset.view;
            document.querySelectorAll('[data-view]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderPCAScatter();
        });
    });

    // Anomaly filter
    document.getElementById('anomaly-filter')?.addEventListener('change', () => {
        renderAnomalyTable();
    });

    // Search
    document.getElementById('search-city')?.addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase();
        document.querySelectorAll('#readings-tbody tr').forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(term) ? '' : 'none';
        });
    });

    // Generate report
    document.getElementById('generate-report-btn')?.addEventListener('click', generateReport);
}

// ================================
// UTILITY FUNCTIONS
// ================================
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function showLoading() {
    document.getElementById('loading-overlay')?.classList.add('active');
}

function hideLoading() {
    document.getElementById('loading-overlay')?.classList.remove('active');
}

function showError(message) {
    alert(message); // Replace with better error handling
}

function updateLastUpdateTime() {
    document.getElementById('last-update-time').textContent = new Date().toLocaleString('en-IN');
}

function viewCityDetails(city) {
    alert(`Viewing details for ${city}\n\nThis would navigate to city_detail.html with data for ${city}`);
    // In production: window.location.href = `city_detail.html?city=${encodeURIComponent(city)}`;
}

function viewAnomalyDetails(row) {
    alert(`Anomaly Details:\n\nCity: ${row.city}\nStation: ${row.station}\nAQI: ${row.AQI_official}\nTimestamp: ${formatDate(row.last_update)}`);
}

async function generateReport() {
    showLoading();
    try {
        // Simulate report generation
        await new Promise(resolve => setTimeout(resolve, 2000));
        alert('Report generated successfully!');
        renderReports();
    } catch (error) {
        showError('Failed to generate report');
    } finally {
        hideLoading();
    }
}

// Auto-refresh every 5 minutes
setInterval(() => {
    initializeApp();
}, 300000);