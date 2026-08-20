// AirSense AI - Frontend Dashboard Controller

let map;
let groundStationsGroup = window.L ? L.layerGroup() : { clearLayers() {}, addLayer() {} };
let mapOverlay = null;
let interactiveMarker = null;
let currentTab = 'map-tab';

// Chart instances
let pmHistChart, aodHistChart, scatterChart, importanceChart, forecastChart;

// Global state variables
let activeModel = 'Random Forest';
let activeDate = '2026-07-11';
let systemStatus = {};

// Initialize application on load
document.addEventListener("DOMContentLoaded", () => {
    initMap();
    checkStatus();
    switchTab('map-tab');
    
    // Set default date picker value
    document.getElementById('map-date').value = activeDate;
    
    // Event listeners
    document.getElementById('map-date').addEventListener('change', (e) => {
        activeDate = e.target.value;
        loadMapOverlay();
    });
    
    document.getElementById('model-select').addEventListener('change', (e) => {
        activeModel = e.target.value;
        loadMapOverlay();
        if (interactiveMarker) updateInteractivePrediction();
    });
    
    document.getElementById('overlay-opacity').addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        if (mapOverlay) {
            mapOverlay.setOpacity(val);
        }
    });
    
    // Setup chat triggers
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });
});

// Initialize Leaflet Map
function initMap() {
    map = L.map('map-container').setView([21.0, 78.0], 5);
    
    // Dark matter tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);
    
    groundStationsGroup.addTo(map);
    
    // Add interactive click listener for predictions
    map.on('click', (e) => {
        setInteractivePredictionCoords(e.latlng.lat, e.latlng.lng);
    });
}

// Check Backend Status
async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        systemStatus = data;
        
        // Update dashboard indicators
        document.getElementById('db-cpcb-count').textContent = data.cpcb_records.toLocaleString();
        document.getElementById('db-sat-count').textContent = data.aod_records.toLocaleString();
        document.getElementById('db-meteo-count').textContent = data.weather_records.toLocaleString();
        
        const modelBadge = document.getElementById('best-model-badge');
        modelBadge.textContent = data.best_model;
        if (data.has_model) {
            modelBadge.classList.remove('text-red-400');
            modelBadge.classList.add('text-green-400');
        } else {
            modelBadge.classList.remove('text-green-400');
            modelBadge.classList.add('text-red-400');
        }
        
        // Update Model selection options
        const modelSelect = document.getElementById('model-select');
        modelSelect.innerHTML = '';
        const defaultModels = ['Random Forest', 'Linear Regression', 'XGBoost', 'LightGBM', 'CatBoost'];
        defaultModels.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === data.best_model || (data.best_model === 'None' && m === 'Random Forest')) {
                opt.selected = true;
                activeModel = m;
            }
            modelSelect.appendChild(opt);
        });
        
        // Enable/Disable tabs and load content
        if (data.has_data) {
            loadGroundStations();
            loadMapOverlay();
            loadEDAMetrics();
        } else {
            showNotice("Dashboard is currently empty. Please click 'Generate Mock Data' in the controls panel to seed the system.");
        }
        
        if (data.has_model) {
            loadEvaluationMetrics();
        }
        
    } catch (e) {
        console.error("Error checking status:", e);
    }
}

// Show Warning Notice
function showNotice(msg) {
    const noticeDiv = document.getElementById('notice-panel');
    noticeDiv.querySelector('p').textContent = msg;
    noticeDiv.classList.remove('hidden');
}

function hideNotice() {
    document.getElementById('notice-panel').classList.add('hidden');
}

// Generate Mock Datasets
async function triggerGenerateMock() {
    toggleBtnLoading('btn-generate', true);
    try {
        const res = await fetch('/api/generate-mock', { method: 'POST' });
        const data = await res.json();
        alert(data.message);
        checkStatus();
    } catch (e) {
        alert("Failed generating mock data: " + e.message);
    } finally {
        toggleBtnLoading('btn-generate', false);
    }
}

// Ingest & Collocate
async function triggerProcessData() {
    toggleBtnLoading('btn-process', true);
    try {
        const res = await fetch('/api/process', { method: 'POST' });
        const data = await res.json();
        alert(data.message);
        // Wait and poll status
        setTimeout(checkStatus, 5000);
    } catch (e) {
        alert("Data processing failed: " + e.message);
    } finally {
        toggleBtnLoading('btn-process', false);
    }
}

// Train models
async function triggerTrainModels() {
    toggleBtnLoading('btn-train', true);
    try {
        const res = await fetch('/api/train', { method: 'POST' });
        const data = await res.json();
        alert(data.message);
        // Wait and check status
        setTimeout(checkStatus, 8000);
    } catch (e) {
        alert("Training failed: " + e.message);
    } finally {
        toggleBtnLoading('btn-train', false);
    }
}

function toggleBtnLoading(btnId, isLoading) {
    const btn = document.getElementById(btnId);
    if (isLoading) {
        btn.disabled = true;
        btn.querySelector('.btn-spinner').classList.remove('hidden');
    } else {
        btn.disabled = false;
        btn.querySelector('.btn-spinner').classList.add('hidden');
    }
}

// Ingest Ground Stations Markers
async function loadGroundStations() {
    try {
        const res = await fetch('/api/ground-stations');
        const stations = await res.json();
        
        groundStationsGroup.clearLayers();
        hideNotice();
        
        stations.forEach(s => {
            if (s.latitude && s.longitude) {
                const color = getAQIColor(s.latest_pm25);
                const marker = L.circleMarker([s.latitude, s.longitude], {
                    radius: 8,
                    fillColor: color,
                    color: '#ffffff',
                    weight: 1.5,
                    opacity: 1,
                    fillOpacity: 0.9
                });
                
                const popupContent = `
                    <div class="text-slate-800 p-2 font-sans" style="min-width: 180px;">
                        <h4 class="font-bold text-sm border-b pb-1 mb-2 text-indigo-700">${s.name}</h4>
                        <div class="mb-1 text-xs"><b>City:</b> ${s.city}, ${s.state}</div>
                        <div class="mb-1 text-xs"><b>Measured PM2.5:</b> <span class="px-1.5 py-0.5 rounded text-white font-bold" style="background-color: ${color}">${s.latest_pm25} µg/m³</span></div>
                        <div class="mb-2 text-xs"><b>Measured PM10:</b> ${s.latest_pm10 || 'N/A'} µg/m³</div>
                        <button onclick="loadStationForecast('${s.id}', '${s.name}')" class="w-full mt-1 bg-indigo-600 text-white text-xs py-1 px-2 rounded hover:bg-indigo-700 transition">
                            Load 72h LSTM Forecast
                        </button>
                    </div>
                `;
                
                marker.bindPopup(popupContent);
                groundStationsGroup.addLayer(marker);
            }
        });
    } catch (e) {
        console.error("Error loading stations:", e);
    }
}

// Fetch 72-hour Forecast
async function loadStationForecast(stationId, stationName) {
    try {
        const res = await fetch(`/api/forecast/${stationId}`);
        if (!res.ok) {
            alert("Forecasting models are not trained yet. Please train models first!");
            return;
        }
        const data = await res.json();
        
        // Show forecast panel in the map sidebar
        document.getElementById('forecast-station-title').textContent = stationName;
        document.getElementById('forecast-val-1h').textContent = data.forecast_1h + " µg/m³";
        document.getElementById('forecast-val-24h').textContent = data.forecast_24h + " µg/m³";
        document.getElementById('forecast-val-72h').textContent = data.forecast_72h + " µg/m³";
        
        // Color code forecast values
        document.getElementById('forecast-val-1h').style.color = getAQIColor(data.forecast_1h);
        document.getElementById('forecast-val-24h').style.color = getAQIColor(data.forecast_24h);
        document.getElementById('forecast-val-72h').style.color = getAQIColor(data.forecast_72h);
        
        document.getElementById('forecast-sidebar').classList.remove('hidden');
        
        // Draw Forecast sequence chart
        renderForecastChart(data.history, [data.forecast_1h, data.forecast_24h, data.forecast_72h]);
        
    } catch (e) {
        console.error("Failed loading forecast:", e);
    }
}

function renderForecastChart(history, forecast) {
    const ctx = document.getElementById('forecastChart').getContext('2d');
    
    // Combine labels
    const labels = ['t-6', 't-5', 't-4', 't-3', 't-2', 't-1', 't (Now)', 't+1h', 't+24h', 't+72h'];
    
    const histData = [...history, null, null, null];
    const forecastData = Array(6).fill(null);
    forecastData.push(history[history.length - 1]); // connect last point
    forecastData.push(...forecast);
    
    if (forecastChart) forecastChart.destroy();
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical PM2.5',
                    data: histData,
                    borderColor: '#94a3b8',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1
                },
                {
                    label: 'LSTM Predicted Forecast',
                    data: forecastData,
                    borderColor: '#00f0ff',
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            },
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            }
        }
    });
}

// Load AOD/PM2.5 Heatmap Overlay on Map
async function loadMapOverlay() {
    if (mapOverlay) {
        map.removeLayer(mapOverlay);
        mapOverlay = null;
    }
    
    try {
        const opacity = parseFloat(document.getElementById('overlay-opacity').value);
        const res = await fetch(`/api/map-layer?date=${activeDate}&model=${activeModel}`);
        if (!res.ok) return;
        const layer = await res.json();
        
        // Add transparent PNG overlay
        mapOverlay = L.imageOverlay(layer.image_url, layer.bounds, {
            opacity: opacity,
            interactive: false
        }).addTo(map);
        
        // Save spatial grid details globally to enable hover-based coordinate predictions
        window.gridData = layer.grid_data;
        
    } catch (e) {
        console.error("Error loading map overlay:", e);
    }
}

// Drag & Drop Prediction Marker
function setInteractivePredictionCoords(lat, lon) {
    document.getElementById('predict-lat').value = lat.toFixed(4);
    document.getElementById('predict-lon').value = lon.toFixed(4);
    
    if (!interactiveMarker) {
        interactiveMarker = L.marker([lat, lon], { draggable: true }).addTo(map);
        interactiveMarker.on('dragend', (e) => {
            const pos = interactiveMarker.getLatLng();
            document.getElementById('predict-lat').value = pos.lat.toFixed(4);
            document.getElementById('predict-lon').value = pos.lng.toFixed(4);
            updateInteractivePrediction();
        });
    } else {
        interactiveMarker.setLatLng([lat, lon]);
    }
    
    updateInteractivePrediction();
}

async function updateInteractivePrediction() {
    const lat = parseFloat(document.getElementById('predict-lat').value);
    const lon = parseFloat(document.getElementById('predict-lon').value);
    const aod = parseFloat(document.getElementById('predict-aod').value);
    const temp = parseFloat(document.getElementById('predict-temp').value);
    const rh = parseFloat(document.getElementById('predict-rh').value);
    const ws = parseFloat(document.getElementById('predict-ws').value);
    const pblh = parseFloat(document.getElementById('predict-pblh').value);
    
    try {
        const res = await fetch(`/api/predict-point?lat=${lat}&lon=${lon}&aod=${aod}&temp=${temp}&rh=${rh}&ws=${ws}&pblh=${pblh}&model_name=${activeModel}`);
        const data = await res.json();
        
        const valDiv = document.getElementById('predict-result-val');
        valDiv.textContent = data.predicted_pm25 + " µg/m³";
        valDiv.style.color = getAQIColor(data.predicted_pm25);
        document.getElementById('predict-result-model').textContent = `Using: ${data.model_used}`;
        
        // Show result box
        document.getElementById('predict-result-box').classList.remove('hidden');
    } catch (e) {
        console.error("Single point prediction failed:", e);
    }
}

// Load Exploratory Data Analysis Panel
async function loadEDAMetrics() {
    try {
        const res = await fetch('/api/eda');
        const data = await res.json();
        
        if (!data.histograms) return;
        
        // 1. PM2.5 histogram
        const pmHist = data.histograms.pm25;
        const ctxPm = document.getElementById('pmHistChart').getContext('2d');
        if (pmHistChart) pmHistChart.destroy();
        pmHistChart = new Chart(ctxPm, {
            type: 'bar',
            data: {
                labels: pmHist.bins.slice(0, -1).map((b, i) => `${b}-${pmHist.bins[i+1]}`),
                datasets: [{
                    label: 'Frequency (PM2.5)',
                    data: pmHist.counts,
                    backgroundColor: 'rgba(16, 185, 129, 0.45)',
                    borderColor: '#10b981',
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                }
            }
        });
        
        // 2. AOD Histogram
        const aodHist = data.histograms.aod;
        const ctxAod = document.getElementById('aodHistChart').getContext('2d');
        if (aodHistChart) aodHistChart.destroy();
        aodHistChart = new Chart(ctxAod, {
            type: 'bar',
            data: {
                labels: aodHist.bins.slice(0, -1).map((b, i) => `${b}-${aodHist.bins[i+1]}`),
                datasets: [{
                    label: 'Frequency (AOD)',
                    data: aodHist.counts,
                    backgroundColor: 'rgba(0, 240, 255, 0.45)',
                    borderColor: '#00f0ff',
                    borderWidth: 1.5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                }
            }
        });
        
        // 3. Render Correlation Heatmap as a nice CSS Grid
        const corr = data.correlation;
        const corrGrid = document.getElementById('correlation-grid');
        corrGrid.innerHTML = '';
        
        // Label Header row
        corrGrid.style.gridTemplateColumns = `repeat(${corr.columns.length + 1}, minmax(0, 1fr))`;
        
        const cornerCell = document.createElement('div');
        cornerCell.className = 'p-1 text-xs font-bold text-slate-400';
        corrGrid.appendChild(cornerCell);
        
        corr.columns.forEach(col => {
            const hCell = document.createElement('div');
            hCell.className = 'p-1 text-[10px] font-bold text-center text-slate-300';
            hCell.textContent = col.toUpperCase();
            corrGrid.appendChild(hCell);
        });
        
        // Matrix content
        for (let i = 0; i < corr.columns.length; i++) {
            const rowLabel = document.createElement('div');
            rowLabel.className = 'p-1 text-[10px] font-bold text-left text-slate-300 flex items-center';
            rowLabel.textContent = corr.columns[i].toUpperCase();
            corrGrid.appendChild(rowLabel);
            
            for (let j = 0; j < corr.columns.length; j++) {
                const val = corr.values[i][j];
                const cell = document.createElement('div');
                cell.className = 'p-2 text-xs font-bold text-center rounded border border-slate-800 m-0.5';
                cell.textContent = val.toFixed(2);
                
                // Color scale background based on correlation value (-1 to 1)
                if (val > 0) {
                    cell.style.backgroundColor = `rgba(0, 240, 255, ${val * 0.7})`;
                    cell.style.color = val > 0.4 ? '#000000' : '#ffffff';
                } else {
                    cell.style.backgroundColor = `rgba(239, 68, 68, ${Math.abs(val) * 0.7})`;
                    cell.style.color = Math.abs(val) > 0.4 ? '#ffffff' : '#ffffff';
                }
                
                corrGrid.appendChild(cell);
            }
        }
        
    } catch (e) {
        console.error("EDA load failed:", e);
    }
}

// Load Model Evaluation and Features Importance Tab
async function loadEvaluationMetrics() {
    try {
        const summary_res = await fetch('/models/training_summary.json');
        if (!summary_res.ok) return;
        const data = await summary_res.json();
        
        // 1. Populate Metrics Table
        const tbody = document.getElementById('metrics-tbody');
        tbody.innerHTML = '';
        
        Object.keys(data.metrics).forEach(modelName => {
            const m = data.metrics[modelName];
            const tr = document.createElement('tr');
            tr.className = 'border-b border-slate-800 text-sm hover:bg-slate-900/40';
            
            const isBest = modelName === data.best_model;
            
            tr.innerHTML = `
                <td class="px-4 py-3 font-semibold text-slate-200 flex items-center">
                    ${modelName} 
                    ${isBest ? '<span class="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-green-500/20 text-green-400 border border-green-500/30">BEST</span>' : ''}
                </td>
                <td class="px-4 py-3 text-cyan-400 font-bold">${m.r2 !== undefined ? m.r2.toFixed(4) : 'N/A'}</td>
                <td class="px-4 py-3 text-slate-300">${m.rmse !== undefined ? m.rmse : 'N/A'}</td>
                <td class="px-4 py-3 text-slate-300">${m.mae !== undefined ? m.mae : 'N/A'}</td>
                <td class="px-4 py-3 text-slate-400">${m.mbe !== undefined ? m.mbe : 'N/A'}</td>
            `;
            tbody.appendChild(tr);
        });
        
        // 2. Render Scatter Plot (Observed vs Predicted PM2.5) for the best model
        const scatterData = data.scatter[data.best_model];
        if (scatterData) {
            const ctxScatter = document.getElementById('scatterChart').getContext('2d');
            
            const points = scatterData.observed.map((obs, i) => {
                return { x: obs, y: scatterData.predicted[i] };
            });
            
            if (scatterChart) scatterChart.destroy();
            
            scatterChart = new Chart(ctxScatter, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: `Validation Points (${data.best_model})`,
                            data: points,
                            backgroundColor: 'rgba(0, 240, 255, 0.6)',
                            borderColor: '#00f0ff',
                            borderWidth: 1,
                            pointRadius: 4
                        },
                        {
                            label: '1:1 Fit Line',
                            data: [{x: 0, y: 0}, {x: 350, y: 350}],
                            type: 'line',
                            borderColor: 'rgba(255, 64, 128, 0.5)',
                            borderWidth: 1.5,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Observed CPCB PM2.5 (µg/m³)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                        y: { title: { display: true, text: 'Estimated Satellite PM2.5 (µg/m³)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                    },
                    plugins: {
                        legend: { labels: { color: '#f8fafc' } }
                    }
                }
            });
        }
        
        // 3. Render Feature Importance Bar Chart for Best Model
        const importances = data.importances[data.best_model];
        if (importances) {
            const ctxImp = document.getElementById('importanceChart').getContext('2d');
            const sortedFeatures = Object.keys(importances).sort((a, b) => importances[b] - importances[a]);
            const values = sortedFeatures.map(f => importances[f] * 100); // percentage
            
            if (importanceChart) importanceChart.destroy();
            
            importanceChart = new Chart(ctxImp, {
                type: 'bar',
                data: {
                    labels: sortedFeatures.map(f => f.toUpperCase()),
                    datasets: [{
                        label: 'Relative Importance (%)',
                        data: values,
                        backgroundColor: 'rgba(139, 92, 246, 0.5)',
                        borderColor: '#8b5cf6',
                        borderWidth: 1.5
                    }]
                },
                options: {
                    indexAxis: 'y', // horizontal bar chart
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: { title: { display: true, text: 'Importance %', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        }
        
    } catch (e) {
        console.error("Evaluation loading failed:", e);
    }
}

// AI RAG Chat Assistant Handler
async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const query = input.value.trim();
    if (!query) return;
    
    input.value = '';
    appendMessage(query, 'user');
    
    const botMsgId = appendMessage('Thinking...', 'bot', true);
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await res.json();
        
        // Format answer
        let formattedAnswer = data.answer.replace(/\n/g, '<br>');
        
        if (data.citations && data.citations.length > 0) {
            formattedAnswer += `<div class="mt-2 pt-2 border-t border-slate-700 text-[10px] text-cyan-400"><b>Sources:</b> ${data.citations.join(', ')}</div>`;
        }
        
        updateBotMessage(botMsgId, formattedAnswer);
    } catch (e) {
        updateBotMessage(botMsgId, "Failed querying AI assistant: " + e.message);
    }
}

function quickChatQuery(text) {
    document.getElementById('chat-input').value = text;
    sendChatMessage();
}

function appendMessage(text, sender, isPending = false) {
    const log = document.getElementById('chat-log');
    const msg = document.createElement('div');
    const msgId = 'msg-' + Date.now();
    msg.id = msgId;
    msg.className = `chat-message ${sender} ${isPending ? 'pending' : ''}`;
    msg.innerHTML = text;
    
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
    return msgId;
}

function updateBotMessage(msgId, text) {
    const msg = document.getElementById(msgId);
    if (msg) {
        msg.classList.remove('pending');
        msg.innerHTML = text;
        const log = document.getElementById('chat-log');
        log.scrollTop = log.scrollHeight;
    }
}

// Tab Switching Utility
function switchTab(tabId) {
    currentTab = tabId;
    
    // Toggle Active button styling
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabId) btn.classList.add('active');
    });
    
    // Toggle panels visibility
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.add('hidden');
    });
    document.getElementById(tabId).classList.remove('hidden');
    
    // Trigger canvas resizing on tabs switch (Chart.js needs this)
    if (tabId === 'eda-tab') {
        loadEDAMetrics();
    } else if (tabId === 'eval-tab') {
        loadEvaluationMetrics();
    } else if (tabId === 'map-tab') {
        if (map) map.invalidateSize();
    }
}

// AQI Colors
function getAQIColor(pm25) {
    if (!pm25 || pm25 < 0) return '#64748b'; // default grey
    if (pm25 <= 30) return '#10b981'; // Good (Green)
    if (pm25 <= 60) return '#facc15'; // Satisfactory (Yellow)
    if (pm25 <= 90) return '#f97316'; // Moderate (Orange)
    if (pm25 <= 120) return '#ef4444'; // Poor (Red)
    if (pm25 <= 250) return '#8b5cf6'; // Very Poor (Purple)
    return '#7f1d1d'; // Severe (Maroon)
}

// Three.js complete Earth globe. These declarations intentionally replace the
// earlier Leaflet map functions while keeping the rest of the dashboard intact.
let globeScene, globeCamera, globeRenderer, globeRoot, earthMesh, heatmapGroup, stationGroup;
let globeRaycaster, globeMouse, globeTooltip, globeAnimationId;
let globeStations = [];
let globeDragging = false;
let globeLastPointer = { x: 0, y: 0 };
let globeTargetRotation = { x: -0.22, y: -1.35 };
let globeRotation = { x: -0.22, y: -1.35 };

function initMap() {
    const container = document.getElementById('map-container');
    if (!container) return;
    if (!window.THREE) {
        initCanvasGlobe();
        return;
    }

    container.innerHTML = '';
    map = { invalidateSize: resizeGlobe };

    globeScene = new THREE.Scene();
    globeCamera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.1, 100);
    globeCamera.position.set(0, 0, 3.35);

    globeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    globeRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    globeRenderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(globeRenderer.domElement);

    globeRoot = new THREE.Group();
    globeScene.add(globeRoot);

    const earth = new THREE.SphereGeometry(1, 96, 96);
    const earthMaterial = new THREE.MeshPhongMaterial({
        map: createEarthTexture(),
        bumpMap: createEarthBumpTexture(),
        bumpScale: 0.035,
        color: 0xffffff,
        shininess: 18,
        specular: new THREE.Color(0x1e4d6b)
    });
    earthMesh = new THREE.Mesh(earth, earthMaterial);
    globeRoot.add(earthMesh);

    const atmosphere = new THREE.Mesh(
        new THREE.SphereGeometry(1.035, 96, 96),
        new THREE.MeshBasicMaterial({
            color: 0x00f0ff,
            transparent: true,
            opacity: 0.08,
            blending: THREE.AdditiveBlending
        })
    );
    globeRoot.add(atmosphere);

    heatmapGroup = new THREE.Group();
    stationGroup = new THREE.Group();
    globeRoot.add(heatmapGroup, stationGroup);
    globeScene.add(createStarField());

    globeScene.add(new THREE.AmbientLight(0x7895b8, 1.45));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.1);
    keyLight.position.set(3.5, 2.4, 4.8);
    globeScene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0x00f0ff, 0.9);
    rimLight.position.set(-4, -1.5, -2);
    globeScene.add(rimLight);

    globeRaycaster = new THREE.Raycaster();
    globeMouse = new THREE.Vector2();
    globeTooltip = document.createElement('div');
    globeTooltip.className = 'globe-tooltip';
    globeTooltip.innerHTML = '<b>Global PM2.5 Heatmap</b><br>Drag to rotate Earth. Click the globe to send coordinates into the point predictor.';
    container.appendChild(globeTooltip);

    container.addEventListener('pointerdown', onGlobePointerDown);
    container.addEventListener('pointermove', onGlobePointerMove);
    container.addEventListener('pointerup', onGlobePointerUp);
    container.addEventListener('pointerleave', onGlobePointerUp);
    container.addEventListener('click', onGlobeClick);
    window.addEventListener('resize', resizeGlobe);

    renderGlobalHeatmap();
    animateGlobe();
}

function createEarthTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 1024;
    const ctx = canvas.getContext('2d');

    const ocean = ctx.createLinearGradient(0, 0, 0, canvas.height);
    ocean.addColorStop(0, '#103b67');
    ocean.addColorStop(0.5, '#0f5c78');
    ocean.addColorStop(1, '#06233f');
    ctx.fillStyle = ocean;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = 'rgba(15, 118, 110, 0.92)';
    const land = [
        [-102, 53, 34, 25], [-75, -15, 22, 32], [-60, -3, 18, 18],
        [-10, 7, 28, 30], [20, 5, 25, 28], [15, 50, 30, 17],
        [78, 22, 36, 22], [103, 35, 42, 24], [125, -25, 22, 18],
        [138, -5, 12, 10], [45, 62, 70, 13], [-42, 73, 25, 9],
        [30, -23, 15, 14], [48, -19, 10, 9]
    ];

    land.forEach(([lon, lat, w, h], i) => {
        const p = lonLatToCanvas(lon, lat, canvas.width, canvas.height);
        ctx.beginPath();
        for (let a = 0; a <= Math.PI * 2 + 0.1; a += 0.18) {
            const noise = 1 + Math.sin(a * 3 + i) * 0.12 + Math.cos(a * 5) * 0.08;
            const x = p.x + Math.cos(a) * w * 4.7 * noise;
            const y = p.y + Math.sin(a) * h * 3.4 * noise;
            if (a === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
    });

    ctx.strokeStyle = 'rgba(203, 213, 225, 0.2)';
    ctx.lineWidth = 1;
    for (let lon = -180; lon <= 180; lon += 30) {
        const x = lonLatToCanvas(lon, 0, canvas.width, canvas.height).x;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    for (let lat = -60; lat <= 60; lat += 30) {
        const y = lonLatToCanvas(0, lat, canvas.width, canvas.height).y;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(canvas.width, y);
        ctx.stroke();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
}

function createEarthBumpTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#1b3652';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#7aa083';
    [
        [-102, 53, 34, 25], [-75, -15, 22, 32], [-10, 7, 28, 30],
        [15, 50, 30, 17], [78, 22, 36, 22], [103, 35, 42, 24],
        [125, -25, 22, 18]
    ].forEach(([lon, lat, w, h]) => {
        const p = lonLatToCanvas(lon, lat, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, w * 2.4, h * 1.8, 0, 0, Math.PI * 2);
        ctx.fill();
    });
    return new THREE.CanvasTexture(canvas);
}

function lonLatToCanvas(lon, lat, width, height) {
    return {
        x: ((lon + 180) / 360) * width,
        y: ((90 - lat) / 180) * height
    };
}

function createStarField() {
    const geometry = new THREE.BufferGeometry();
    const count = 900;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const r = 8 + Math.random() * 16;
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return new THREE.Points(
        geometry,
        new THREE.PointsMaterial({ color: 0x9fb8d8, size: 0.018, transparent: true, opacity: 0.75 })
    );
}

function latLonToVector3(lat, lon, radius = 1.015) {
    const phi = THREE.MathUtils.degToRad(90 - lat);
    const theta = THREE.MathUtils.degToRad(lon + 180);
    return new THREE.Vector3(
        -radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.cos(phi),
        radius * Math.sin(phi) * Math.sin(theta)
    );
}

function vector3ToLatLon(vec) {
    const normalized = vec.clone().normalize();
    const lat = 90 - THREE.MathUtils.radToDeg(Math.acos(normalized.y));
    const lon = THREE.MathUtils.radToDeg(Math.atan2(normalized.z, -normalized.x)) - 180;
    return {
        lat: Math.max(-90, Math.min(90, lat)),
        lon: ((((lon + 180) % 360) + 360) % 360) - 180
    };
}

function getGlobalHeatPoints() {
    const hotspots = [
        { lat: 28.6, lon: 77.2, pm25: 190, name: 'Delhi NCR' },
        { lat: 31.5, lon: 74.3, pm25: 165, name: 'Lahore' },
        { lat: 39.9, lon: 116.4, pm25: 135, name: 'Beijing' },
        { lat: 35.7, lon: 139.7, pm25: 52, name: 'Tokyo' },
        { lat: -6.2, lon: 106.8, pm25: 95, name: 'Jakarta' },
        { lat: 30.0, lon: 31.2, pm25: 115, name: 'Cairo' },
        { lat: 24.7, lon: 46.7, pm25: 105, name: 'Riyadh' },
        { lat: 51.5, lon: -0.1, pm25: 34, name: 'London' },
        { lat: 48.9, lon: 2.3, pm25: 42, name: 'Paris' },
        { lat: 40.7, lon: -74.0, pm25: 38, name: 'New York' },
        { lat: 34.0, lon: -118.2, pm25: 67, name: 'Los Angeles' },
        { lat: 19.4, lon: -99.1, pm25: 76, name: 'Mexico City' },
        { lat: -23.5, lon: -46.6, pm25: 48, name: 'Sao Paulo' },
        { lat: -33.9, lon: 151.2, pm25: 28, name: 'Sydney' },
        { lat: -26.2, lon: 28.0, pm25: 82, name: 'Johannesburg' }
    ];

    const stationPoints = globeStations.map(s => ({
        lat: Number(s.latitude),
        lon: Number(s.longitude),
        pm25: Number(s.latest_pm25 || 0),
        name: s.city || s.name
    })).filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon));

    return [...hotspots, ...stationPoints];
}

function renderGlobalHeatmap() {
    if (!heatmapGroup || !stationGroup) return;
    heatmapGroup.clear();
    stationGroup.clear();

    const opacity = parseFloat(document.getElementById('overlay-opacity')?.value || '0.6');
    const points = getGlobalHeatPoints();

    points.forEach(point => {
        const intensity = Math.max(0.18, Math.min(1, point.pm25 / 220));
        const sprite = createHeatSprite(getAQIColor(point.pm25), intensity * opacity);
        const scale = 0.055 + intensity * 0.15;
        sprite.scale.set(scale, scale, scale);
        sprite.position.copy(latLonToVector3(point.lat, point.lon, 1.035));
        heatmapGroup.add(sprite);
    });

    globeStations.forEach(station => {
        const marker = new THREE.Mesh(
            new THREE.SphereGeometry(0.014, 16, 16),
            new THREE.MeshBasicMaterial({ color: new THREE.Color(getAQIColor(station.latest_pm25)) })
        );
        marker.position.copy(latLonToVector3(station.latitude, station.longitude, 1.065));
        marker.userData = station;
        stationGroup.add(marker);
    });
}

function createHeatSprite(color, opacity) {
    const canvas = document.createElement('canvas');
    canvas.width = 96;
    canvas.height = 96;
    const ctx = canvas.getContext('2d');
    const gradient = ctx.createRadialGradient(48, 48, 0, 48, 48, 48);
    gradient.addColorStop(0, color);
    gradient.addColorStop(0.35, color);
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const texture = new THREE.CanvasTexture(canvas);
    return new THREE.Sprite(new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    }));
}

async function loadGroundStations() {
    try {
        const res = await fetch('/api/ground-stations');
        globeStations = await res.json();
        hideNotice();
        renderGlobalHeatmap();
    } catch (e) {
        console.error("Error loading stations for globe:", e);
    }
}

async function loadMapOverlay() {
    renderGlobalHeatmap();
}

function setInteractivePredictionCoords(lat, lon) {
    document.getElementById('predict-lat').value = lat.toFixed(4);
    document.getElementById('predict-lon').value = lon.toFixed(4);
    updateInteractivePrediction();
}

function onGlobePointerDown(event) {
    globeDragging = true;
    globeLastPointer = { x: event.clientX, y: event.clientY };
}

function onGlobePointerMove(event) {
    const container = document.getElementById('map-container');
    const rect = container.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;

    if (globeDragging) {
        const dx = event.clientX - globeLastPointer.x;
        const dy = event.clientY - globeLastPointer.y;
        globeTargetRotation.y += dx * 0.006;
        globeTargetRotation.x += dy * 0.004;
        globeTargetRotation.x = Math.max(-1.15, Math.min(1.15, globeTargetRotation.x));
        globeLastPointer = { x: event.clientX, y: event.clientY };
    } else {
        globeTargetRotation.y += ((px / rect.width) - 0.5) * 0.002;
        globeTargetRotation.x += ((py / rect.height) - 0.5) * 0.001;
    }

    const coords = pickGlobeLatLon(event);
    if (coords && globeTooltip) {
        globeTooltip.innerHTML = `<b>Global PM2.5 Heatmap</b><br>Lat ${coords.lat.toFixed(2)}, Lon ${coords.lon.toFixed(2)}<br>Click to run point prediction.`;
    }
}

function onGlobePointerUp() {
    globeDragging = false;
}

function onGlobeClick(event) {
    const coords = pickGlobeLatLon(event);
    if (coords) setInteractivePredictionCoords(coords.lat, coords.lon);
}

function pickGlobeLatLon(event) {
    if (!globeRaycaster || !earthMesh) return null;
    const rect = globeRenderer.domElement.getBoundingClientRect();
    globeMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    globeMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    globeRaycaster.setFromCamera(globeMouse, globeCamera);
    const hits = globeRaycaster.intersectObject(earthMesh);
    if (!hits.length) return null;
    const localPoint = earthMesh.worldToLocal(hits[0].point.clone());
    return vector3ToLatLon(localPoint);
}

function resizeGlobe() {
    const container = document.getElementById('map-container');
    if (canvasGlobe && !window.THREE) {
        resizeCanvasGlobe();
        return;
    }
    if (!container || !globeCamera || !globeRenderer) return;
    globeCamera.aspect = container.clientWidth / container.clientHeight;
    globeCamera.updateProjectionMatrix();
    globeRenderer.setSize(container.clientWidth, container.clientHeight);
}

function animateGlobe() {
    globeAnimationId = requestAnimationFrame(animateGlobe);
    if (!globeDragging) globeTargetRotation.y += 0.0007;
    globeRotation.x += (globeTargetRotation.x - globeRotation.x) * 0.08;
    globeRotation.y += (globeTargetRotation.y - globeRotation.y) * 0.08;
    if (globeRoot) {
        globeRoot.rotation.x = globeRotation.x;
        globeRoot.rotation.y = globeRotation.y;
    }
    globeRenderer.render(globeScene, globeCamera);
}

function switchTab(tabId) {
    currentTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabId) btn.classList.add('active');
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.add('hidden');
    });
    document.getElementById(tabId).classList.remove('hidden');

    if (tabId === 'eda-tab') {
        loadEDAMetrics();
    } else if (tabId === 'eval-tab') {
        loadEvaluationMetrics();
    } else if (tabId === 'map-tab') {
        setTimeout(resizeGlobe, 50);
    }
}

let canvasGlobe = null;

function initCanvasGlobe() {
    const container = document.getElementById('map-container');
    container.innerHTML = '';

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    container.appendChild(canvas);

    globeTooltip = document.createElement('div');
    globeTooltip.className = 'globe-tooltip';
    globeTooltip.innerHTML = '<b>Global PM2.5 Heatmap</b><br>Drag to rotate Earth. Click the globe to send coordinates into the point predictor.';
    container.appendChild(globeTooltip);

    canvasGlobe = {
        canvas,
        ctx,
        yaw: -1.35,
        pitch: -0.22,
        targetYaw: -1.35,
        targetPitch: -0.22,
        dragging: false,
        lastX: 0,
        lastY: 0
    };
    map = { invalidateSize: resizeGlobe };

    container.addEventListener('pointerdown', (event) => {
        canvasGlobe.dragging = true;
        canvasGlobe.lastX = event.clientX;
        canvasGlobe.lastY = event.clientY;
    });
    container.addEventListener('pointermove', (event) => {
        const rect = container.getBoundingClientRect();
        if (canvasGlobe.dragging) {
            const dx = event.clientX - canvasGlobe.lastX;
            const dy = event.clientY - canvasGlobe.lastY;
            canvasGlobe.targetYaw += dx * 0.008;
            canvasGlobe.targetPitch = Math.max(-1.15, Math.min(1.15, canvasGlobe.targetPitch + dy * 0.005));
            canvasGlobe.lastX = event.clientX;
            canvasGlobe.lastY = event.clientY;
        } else {
            canvasGlobe.targetYaw += ((event.clientX - rect.left) / rect.width - 0.5) * 0.002;
        }
        const coords = pickCanvasGlobeLatLon(event);
        if (coords) {
            globeTooltip.innerHTML = `<b>Global PM2.5 Heatmap</b><br>Lat ${coords.lat.toFixed(2)}, Lon ${coords.lon.toFixed(2)}<br>Click to run point prediction.`;
        }
    });
    container.addEventListener('pointerup', () => { canvasGlobe.dragging = false; });
    container.addEventListener('pointerleave', () => { canvasGlobe.dragging = false; });
    container.addEventListener('click', (event) => {
        const coords = pickCanvasGlobeLatLon(event);
        if (coords) setInteractivePredictionCoords(coords.lat, coords.lon);
    });
    window.addEventListener('resize', resizeGlobe);

    resizeGlobe();
    animateCanvasGlobe();
}

function resizeCanvasGlobe() {
    if (!canvasGlobe) return;
    const container = document.getElementById('map-container');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvasGlobe.canvas.width = Math.max(1, Math.floor(container.clientWidth * dpr));
    canvasGlobe.canvas.height = Math.max(1, Math.floor(container.clientHeight * dpr));
    canvasGlobe.canvas.style.width = `${container.clientWidth}px`;
    canvasGlobe.canvas.style.height = `${container.clientHeight}px`;
    canvasGlobe.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function animateCanvasGlobe() {
    if (!canvasGlobe || window.THREE) return;
    requestAnimationFrame(animateCanvasGlobe);
    canvasGlobe.yaw += (canvasGlobe.targetYaw - canvasGlobe.yaw) * 0.08;
    canvasGlobe.pitch += (canvasGlobe.targetPitch - canvasGlobe.pitch) * 0.08;
    if (!canvasGlobe.dragging) canvasGlobe.targetYaw += 0.0008;
    drawCanvasGlobe();
}

function drawCanvasGlobe() {
    const { canvas, ctx } = canvasGlobe;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.38;

    ctx.clearRect(0, 0, width, height);
    const halo = ctx.createRadialGradient(cx, cy, radius * 0.2, cx, cy, radius * 1.45);
    halo.addColorStop(0, 'rgba(0, 240, 255, 0.18)');
    halo.addColorStop(1, 'rgba(0, 240, 255, 0)');
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * 1.45, 0, Math.PI * 2);
    ctx.fill();

    const ocean = ctx.createRadialGradient(cx - radius * 0.35, cy - radius * 0.35, radius * 0.15, cx, cy, radius);
    ocean.addColorStop(0, '#1e88aa');
    ocean.addColorStop(0.55, '#0f4f73');
    ocean.addColorStop(1, '#04172d');
    ctx.fillStyle = ocean;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fill();

    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.clip();

    drawCanvasGrid(ctx, cx, cy, radius);
    drawCanvasLand(ctx, cx, cy, radius);
    drawCanvasHeat(ctx, cx, cy, radius);

    ctx.restore();

    ctx.strokeStyle = 'rgba(0, 240, 255, 0.42)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();
}

function drawCanvasGrid(ctx, cx, cy, radius) {
    ctx.strokeStyle = 'rgba(203, 213, 225, 0.16)';
    ctx.lineWidth = 1;
    for (let lat = -60; lat <= 60; lat += 30) {
        const points = [];
        for (let lon = -180; lon <= 180; lon += 5) points.push(projectCanvasPoint(lat, lon, cx, cy, radius));
        strokeVisiblePath(ctx, points);
    }
    for (let lon = -180; lon <= 180; lon += 30) {
        const points = [];
        for (let lat = -85; lat <= 85; lat += 5) points.push(projectCanvasPoint(lat, lon, cx, cy, radius));
        strokeVisiblePath(ctx, points);
    }
}

function drawCanvasLand(ctx, cx, cy, radius) {
    ctx.fillStyle = 'rgba(20, 184, 166, 0.7)';
    [
        [-102, 53, 34, 25], [-75, -15, 22, 32], [-10, 7, 28, 30],
        [18, 51, 33, 18], [78, 22, 36, 22], [105, 35, 42, 24],
        [125, -25, 22, 18], [138, -5, 12, 10], [45, 62, 70, 13],
        [30, -23, 15, 14]
    ].forEach(([lon, lat, w, h]) => {
        const center = projectCanvasPoint(lat, lon, cx, cy, radius);
        if (!center.visible) return;
        ctx.beginPath();
        for (let a = 0; a <= Math.PI * 2 + 0.1; a += 0.25) {
            const p = projectCanvasPoint(
                lat + Math.sin(a) * h * 0.55,
                lon + Math.cos(a) * w * 0.75,
                cx,
                cy,
                radius
            );
            if (!p.visible) continue;
            if (a === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.fill();
    });
}

function drawCanvasHeat(ctx, cx, cy, radius) {
    getGlobalHeatPoints().forEach(point => {
        const p = projectCanvasPoint(point.lat, point.lon, cx, cy, radius);
        if (!p.visible) return;
        const intensity = Math.max(0.2, Math.min(1, point.pm25 / 220));
        const opacity = parseFloat(document.getElementById('overlay-opacity')?.value || '0.6') * intensity;
        const size = radius * (0.035 + intensity * 0.075);
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, size);
        grad.addColorStop(0, getAQIColor(point.pm25));
        grad.addColorStop(0.42, `${getAQIColor(point.pm25)}cc`);
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.globalAlpha = opacity;
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
    });
}

function strokeVisiblePath(ctx, points) {
    let drawing = false;
    ctx.beginPath();
    points.forEach(p => {
        if (!p.visible) {
            drawing = false;
            return;
        }
        if (!drawing) {
            ctx.moveTo(p.x, p.y);
            drawing = true;
        } else {
            ctx.lineTo(p.x, p.y);
        }
    });
    ctx.stroke();
}

function projectCanvasPoint(lat, lon, cx, cy, radius) {
    const latR = lat * Math.PI / 180;
    const lonR = lon * Math.PI / 180;
    let x = Math.cos(latR) * Math.cos(lonR);
    let y = Math.sin(latR);
    let z = Math.cos(latR) * Math.sin(lonR);

    const yaw = canvasGlobe.yaw;
    const pitch = canvasGlobe.pitch;
    const x1 = x * Math.cos(yaw) + z * Math.sin(yaw);
    const z1 = -x * Math.sin(yaw) + z * Math.cos(yaw);
    const y1 = y * Math.cos(pitch) - z1 * Math.sin(pitch);
    const z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);

    return {
        x: cx + x1 * radius,
        y: cy - y1 * radius,
        z: z2,
        visible: z2 > -0.05
    };
}

function pickCanvasGlobeLatLon(event) {
    if (!canvasGlobe) return null;
    const rect = canvasGlobe.canvas.getBoundingClientRect();
    const width = canvasGlobe.canvas.clientWidth;
    const height = canvasGlobe.canvas.clientHeight;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.38;
    const x = (event.clientX - rect.left - cx) / radius;
    const y = -(event.clientY - rect.top - cy) / radius;
    const mag = x * x + y * y;
    if (mag > 1) return null;
    const z = Math.sqrt(1 - mag);

    const pitch = -canvasGlobe.pitch;
    const yaw = -canvasGlobe.yaw;
    const y1 = y * Math.cos(pitch) - z * Math.sin(pitch);
    const z1 = y * Math.sin(pitch) + z * Math.cos(pitch);
    const x1 = x * Math.cos(yaw) + z1 * Math.sin(yaw);
    const z2 = -x * Math.sin(yaw) + z1 * Math.cos(yaw);

    return {
        lat: Math.asin(y1) * 180 / Math.PI,
        lon: Math.atan2(z2, x1) * 180 / Math.PI
    };
}

// Final map mode: flat world map with PM2.5 heat overlay.
// This intentionally overrides the globe renderer above per the requested UI.
let heatLayer = null;

function initMap() {
    const container = document.getElementById('map-container');
    if (!container || !window.L) return;

    container.innerHTML = '';
    map = L.map('map-container', {
        worldCopyJump: true,
        zoomControl: true,
        minZoom: 2,
        maxZoom: 10
    }).setView([35, -98], 4);

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 10
    }).addTo(map);

    L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Labels &copy; Esri',
        maxZoom: 10,
        opacity: 0.55
    }).addTo(map);

    groundStationsGroup = L.layerGroup().addTo(map);

    map.on('click', (e) => {
        setInteractivePredictionCoords(e.latlng.lat, e.latlng.lng);
    });

    renderGlobalHeatmap();
}

async function loadGroundStations() {
    try {
        const res = await fetch('/api/ground-stations');
        globeStations = await res.json();
        hideNotice();
        renderStationMarkers();
        renderGlobalHeatmap();
    } catch (e) {
        console.error("Error loading stations:", e);
    }
}

async function loadMapOverlay() {
    renderGlobalHeatmap();
}

function renderStationMarkers() {
    if (!groundStationsGroup || !window.L) return;
    groundStationsGroup.clearLayers();

    globeStations.forEach(s => {
        if (!Number.isFinite(Number(s.latitude)) || !Number.isFinite(Number(s.longitude))) return;
        const color = getAQIColor(s.latest_pm25);
        const marker = L.marker([s.latitude, s.longitude], {
            icon: createPmMarkerIcon(s.latest_pm25, color)
        });

        marker.bindPopup(`
            <div class="text-slate-800 p-2 font-sans" style="min-width: 180px;">
                <h4 class="font-bold text-sm border-b pb-1 mb-2 text-indigo-700">${s.name}</h4>
                <div class="mb-1 text-xs"><b>City:</b> ${s.city}, ${s.state}</div>
                <div class="mb-1 text-xs"><b>PM2.5:</b> <span style="color:${color};font-weight:800">${s.latest_pm25} ug/m3</span></div>
                <div class="mb-2 text-xs"><b>PM10:</b> ${s.latest_pm10 || 'N/A'} ug/m3</div>
                <button onclick="loadStationForecast('${s.id}', '${s.name}')" class="w-full mt-1 bg-indigo-600 text-white text-xs py-1 px-2 rounded hover:bg-indigo-700 transition">
                    Load 72h Forecast
                </button>
            </div>
        `);
        groundStationsGroup.addLayer(marker);
    });
}

function createPmMarkerIcon(value, color) {
    const safeValue = Math.round(Number(value || 0));
    return L.divIcon({
        className: 'pm-marker-icon',
        iconSize: [44, 56],
        iconAnchor: [22, 54],
        popupAnchor: [0, -48],
        html: `
            <div style="position:relative;width:44px;height:56px;">
                <div style="position:absolute;left:4px;top:0;width:36px;height:36px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);background:#f8fafc;border:2px solid rgba(15,23,42,.35);box-shadow:0 5px 16px rgba(0,0,0,.35);"></div>
                <div style="position:absolute;left:10px;top:6px;width:24px;height:24px;border-radius:999px;border:3px solid ${color};border-left-color:transparent;display:flex;align-items:center;justify-content:center;color:#0f172a;font:700 12px Inter,Arial;background:white;">${safeValue}</div>
            </div>
        `
    });
}

function renderGlobalHeatmap() {
    if (!map || !window.L) return;
    if (!heatLayer) {
        heatLayer = L.layerGroup().addTo(map);
        heatLayer.setOpacity = function(opacity) {
            this.eachLayer(layer => {
                if (layer.setStyle) layer.setStyle({ opacity, fillOpacity: opacity * 0.34 });
            });
        };
    }

    heatLayer.clearLayers();
    const opacity = parseFloat(document.getElementById('overlay-opacity')?.value || '0.6');
    const points = getDenseHeatPoints();

    points.forEach(point => {
        const intensity = Math.max(0.18, Math.min(1, point.pm25 / 220));
        const color = heatColor(point.pm25);
        const radius = 18000 + intensity * 85000;
        L.circle([point.lat, point.lon], {
            radius,
            stroke: false,
            fillColor: color,
            fillOpacity: opacity * (0.18 + intensity * 0.22),
            interactive: false
        }).addTo(heatLayer);
    });

    mapOverlay = heatLayer;
}

function getDenseHeatPoints() {
    const base = [
        { lat: 45.5, lon: -122.7, pm25: 140 }, { lat: 37.8, lon: -122.4, pm25: 81 },
        { lat: 34.0, lon: -118.2, pm25: 96 }, { lat: 36.2, lon: -115.1, pm25: 72 },
        { lat: 40.8, lon: -111.9, pm25: 84 }, { lat: 39.7, lon: -104.9, pm25: 62 },
        { lat: 33.4, lon: -112.0, pm25: 112 }, { lat: 32.8, lon: -117.1, pm25: 76 },
        { lat: 29.8, lon: -95.4, pm25: 68 }, { lat: 32.8, lon: -96.8, pm25: 74 },
        { lat: 41.9, lon: -87.6, pm25: 88 }, { lat: 42.3, lon: -83.0, pm25: 118 },
        { lat: 39.9, lon: -75.2, pm25: 56 }, { lat: 40.7, lon: -74.0, pm25: 64 },
        { lat: 28.6, lon: 77.2, pm25: 190 }, { lat: 31.5, lon: 74.3, pm25: 165 },
        { lat: 39.9, lon: 116.4, pm25: 135 }, { lat: 30.0, lon: 31.2, pm25: 115 },
        { lat: -6.2, lon: 106.8, pm25: 95 }, { lat: 24.7, lon: 46.7, pm25: 105 },
        { lat: -26.2, lon: 28.0, pm25: 82 }, { lat: 19.4, lon: -99.1, pm25: 76 }
    ];

    const spread = [];
    base.forEach((p, index) => {
        spread.push(p);
        for (let i = 0; i < 10; i++) {
            const angle = (i / 10) * Math.PI * 2;
            const dist = 0.7 + ((index + i) % 5) * 0.42;
            spread.push({
                lat: p.lat + Math.sin(angle) * dist,
                lon: p.lon + Math.cos(angle) * dist,
                pm25: Math.max(20, p.pm25 * (0.55 + ((i % 4) * 0.1)))
            });
        }
    });

    const stationPoints = globeStations.map(s => ({
        lat: Number(s.latitude),
        lon: Number(s.longitude),
        pm25: Number(s.latest_pm25 || 0)
    })).filter(p => Number.isFinite(p.lat) && Number.isFinite(p.lon));

    return [...spread, ...stationPoints];
}

function heatColor(pm25) {
    if (pm25 <= 40) return '#22c55e';
    if (pm25 <= 75) return '#84cc16';
    if (pm25 <= 110) return '#facc15';
    if (pm25 <= 150) return '#f97316';
    return '#dc2626';
}

function setInteractivePredictionCoords(lat, lon) {
    document.getElementById('predict-lat').value = lat.toFixed(4);
    document.getElementById('predict-lon').value = lon.toFixed(4);
    updateInteractivePrediction();
}

function switchTab(tabId) {
    currentTab = tabId;
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.tab === tabId) btn.classList.add('active');
    });

    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.add('hidden');
    });
    document.getElementById(tabId).classList.remove('hidden');

    if (tabId === 'eda-tab') {
        loadEDAMetrics();
    } else if (tabId === 'eval-tab') {
        loadEvaluationMetrics();
    } else if (tabId === 'map-tab' && map?.invalidateSize) {
        setTimeout(() => map.invalidateSize(), 50);
    }
}
