/**
 * app.js — Dashboard bootstrap, tab switching, level selection, WebSocket manager
 */

(function () {
    'use strict';

    const WS_PORT = 8001;
    const WS_URL = `ws://${window.location.hostname || 'localhost'}:${WS_PORT}`;
    const RECONNECT_DELAY = 2000;

    // ---- DOM ----
    const statusBadge = document.getElementById('sim-status');
    const statusText = statusBadge.querySelector('.status-text');
    const levelPills = document.querySelectorAll('.pill');
    const autoBtn = document.getElementById('auto-btn');
    const episodeTimeEl = document.getElementById('episode-time');
    const episodeDurationEl = document.getElementById('episode-duration');
    const tickCounterEl = document.getElementById('tick-counter');
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    // ---- Modules ----
    const cameraFeeds = new CameraFeeds();
    const metricsPanel = new MetricsPanel();
    const liveCharts = new LiveCharts();
    const navScene = new NavigationScene();

    // ---- State ----
    let ws = null;
    let reconnectTimer = null;
    let levelMode = 'auto'; // 'auto' or 'forced'
    let forcedLevel = null;

    // ---- Tab Switching ----
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            tabContents.forEach(tc => tc.classList.remove('active'));
            const el = document.getElementById('tab-' + target);
            if (el) el.classList.add('active');

            // Resize 3D viewport when switching to navigation tab
            if (target === 'navigation') {
                setTimeout(() => navScene._handleResize(), 50);
            }
        });
    });

    // ---- Level Selection ----
    levelPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const level = parseInt(pill.dataset.level);
            forcedLevel = level;
            levelMode = 'forced';
            autoBtn.classList.remove('active');
            sendCommand({ command: 'set_level', level: level });
        });
    });

    autoBtn.addEventListener('click', () => {
        levelMode = 'auto';
        forcedLevel = null;
        autoBtn.classList.add('active');
        sendCommand({ command: 'set_level', level: 'auto' });
    });

    // ---- WebSocket ----
    function setStatus(state) {
        statusBadge.className = 'header-badge ' + state;
        statusText.textContent = state === 'connected' ? 'CONNECTED'
            : state === 'disconnected' ? 'DISCONNECTED' : 'CONNECTING…';
    }

    function sendCommand(cmd) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(cmd));
        }
    }

    function updateHeader(data) {
        const level = data.level || 1;
        const mode = data.level_mode || levelMode;

        levelPills.forEach(pill => {
            const pl = parseInt(pill.dataset.level);
            pill.classList.remove('active', 'completed', 'forced');
            if (mode === 'forced' && pl === level) {
                pill.classList.add('forced');
            } else if (pl === level) {
                pill.classList.add('active');
            } else if (pl < level && mode === 'auto') {
                pill.classList.add('completed');
            }
        });

        if (data.level_time !== undefined) episodeTimeEl.textContent = data.level_time.toFixed(2) + 's';
        if (data.level_duration !== undefined) episodeDurationEl.textContent = data.level_duration.toFixed(2) + 's';
        if (data.tick !== undefined) tickCounterEl.textContent = data.tick.toLocaleString();
    }

    function onMessage(event) {
        try {
            const data = JSON.parse(event.data);
            updateHeader(data);
            cameraFeeds.update(data.images);
            metricsPanel.update(data);
            liveCharts.update(data);
            navScene.update(data);
        } catch (e) {
            console.error('[Dashboard] Parse error:', e);
        }
    }

    function connect() {
        if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
        setStatus('connecting');
        ws = new WebSocket(WS_URL);
        ws.onopen = () => {
            setStatus('connected');
            if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        };
        ws.onmessage = onMessage;
        ws.onclose = () => { setStatus('disconnected'); scheduleReconnect(); };
        ws.onerror = () => ws.close();
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, RECONNECT_DELAY);
    }

    // ---- Resize & Minimize ----
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => navScene._handleResize(), 100);
    });

    document.addEventListener('click', (e) => {
        const btn = e.target.closest('.minimize-btn');
        if (!btn) return;
        const panel = btn.closest('.panel');
        if (!panel) return;

        panel.classList.toggle('minimized');
        const isMin = panel.classList.contains('minimized');
        btn.textContent = isMin ? '+' : '−';
        btn.title = isMin ? 'Expand' : 'Minimize';

        // Dispatch a window resize event to force Three.js and Chart.js to recalculate dimensions
        window.dispatchEvent(new Event('resize'));
    });

    // ---- Start ----
    console.log('[Dashboard] RL Drone Play-Mode Dashboard initialized');
    connect();
})();
