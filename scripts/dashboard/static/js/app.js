/**
 * app.js — Dashboard application bootstrap & WebSocket manager
 * Connects to the telemetry WebSocket, parses messages,
 * and routes data to each panel module.
 */

(function () {
    'use strict';

    // ---- Configuration ----
    const WS_PORT = 8001;
    const WS_URL = `ws://${window.location.hostname || 'localhost'}:${WS_PORT}`;
    const RECONNECT_DELAY = 2000;

    // ---- DOM references ----
    const statusBadge = document.getElementById('sim-status');
    const statusText = statusBadge.querySelector('.status-text');
    const levelPills = document.querySelectorAll('.pill');
    const episodeTimeEl = document.getElementById('episode-time');
    const episodeDurationEl = document.getElementById('episode-duration');
    const tickCounterEl = document.getElementById('tick-counter');

    // ---- Initialize panel modules ----
    const cameraFeeds = new CameraFeeds();
    const metricsPanel = new MetricsPanel();
    const attitudeHUD = new AttitudeHUD();
    const liveCharts = new LiveCharts();
    const navScene = new NavigationScene();

    // ---- WebSocket connection ----
    let ws = null;
    let reconnectTimer = null;

    function setStatus(state) {
        statusBadge.className = 'header-badge ' + state;
        switch (state) {
            case 'connected':
                statusText.textContent = 'CONNECTED';
                break;
            case 'disconnected':
                statusText.textContent = 'DISCONNECTED';
                break;
            default:
                statusText.textContent = 'CONNECTING…';
        }
    }

    function updateHeader(data) {
        // Level pills
        const level = data.level || 1;
        levelPills.forEach(pill => {
            const pillLevel = parseInt(pill.dataset.level);
            pill.classList.remove('active', 'completed');
            if (pillLevel === level) {
                pill.classList.add('active');
            } else if (pillLevel < level) {
                pill.classList.add('completed');
            }
        });

        // Episode time
        if (data.level_time !== undefined) {
            episodeTimeEl.textContent = data.level_time.toFixed(2) + 's';
        }
        if (data.level_duration !== undefined) {
            episodeDurationEl.textContent = data.level_duration.toFixed(2) + 's';
        }

        // Tick counter
        if (data.tick !== undefined) {
            tickCounterEl.textContent = data.tick.toLocaleString();
        }
    }

    function onMessage(event) {
        try {
            const data = JSON.parse(event.data);

            // Route to all panels
            updateHeader(data);
            cameraFeeds.update(data.images);
            metricsPanel.update(data);
            attitudeHUD.update(data);
            liveCharts.update(data);
            navScene.update(data);
        } catch (e) {
            console.error('[Dashboard] Failed to parse message:', e);
        }
    }

    function connect() {
        if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) {
            return;
        }

        setStatus('connecting');
        console.log('[Dashboard] Connecting to', WS_URL);

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('[Dashboard] WebSocket connected');
            setStatus('connected');
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
        };

        ws.onmessage = onMessage;

        ws.onclose = () => {
            console.log('[Dashboard] WebSocket disconnected');
            setStatus('disconnected');
            scheduleReconnect();
        };

        ws.onerror = (err) => {
            console.error('[Dashboard] WebSocket error:', err);
            ws.close();
        };
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connect();
        }, RECONNECT_DELAY);
    }

    // ---- Handle window resize for 3D viewport ----
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            navScene._handleResize();
        }, 100);
    });

    // ---- Start ----
    console.log('[Dashboard] RL Drone Play-Mode Dashboard initialized');
    connect();
})();
