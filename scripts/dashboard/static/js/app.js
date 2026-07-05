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
    const attitudeHud = document.getElementById('canvas-hud') ? new AttitudeHUD() : null;
    const slam3dScene = new SlamScene3D('slam-3d-container');
    const slam2dMap = document.getElementById('cam-slam-main') ? new SlamMap2D('cam-slam-main') : null;
    const yoloHud = document.getElementById('cam-yolo-hud') ? new YoloHudView('cam-yolo-hud') : null;

    // SLAM tab stat elements
    const slamEls = {
        state:     document.getElementById('slam-state-val'),
        explored:  document.getElementById('slam-explored-val'),
        frontiers: document.getElementById('slam-frontiers-val'),
        person:    document.getElementById('slam-person-val'),
        pos:       document.getElementById('slam-pos-val'),
        alt:       document.getElementById('slam-alt-val'),
        hdg:       document.getElementById('slam-hdg-val'),
        spd:       document.getElementById('slam-spd-val'),
        time:      document.getElementById('slam-time-val'),
    };

    function updateSlamStats(data) {
        if (!data) return;
        const s = slamEls;
        if (s.state)     s.state.textContent  = data.slam_state || '—';
        if (s.explored)  s.explored.textContent = (data.map_explored_pct || 0).toFixed(1) + '%';
        if (s.frontiers) s.frontiers.textContent = data.frontier_count || 0;
        if (s.person)    s.person.textContent  = data.people_found ? 'FOUND' : 'Not found';
        if (data.pos) {
            if (s.pos) s.pos.textContent = data.pos[0].toFixed(2) + ' / ' + data.pos[1].toFixed(2);
            if (s.alt) s.alt.textContent = data.pos[2].toFixed(2) + ' m';
        }
        if (s.hdg && data.yaw !== undefined) {
            let deg = (data.yaw * 180 / Math.PI) % 360;
            if (deg < 0) deg += 360;
            s.hdg.textContent = deg.toFixed(1) + '°';
        }
        if (s.spd && data.lin_vel) {
            const [vx, vy, vz] = data.lin_vel;
            s.spd.textContent = Math.sqrt(vx*vx + vy*vy + vz*vz).toFixed(2) + ' m/s';
        }
        if (s.time && data.level_time !== undefined) {
            s.time.textContent = data.level_time.toFixed(0) + 's';
        }
        // Colour-code state
        if (s.state) {
            const col = data.slam_state === 'SCAN' ? '#22d3ee'
                      : data.slam_state === 'EXPLORE' ? '#34d399'
                      : data.slam_state === 'COMPLETE' ? '#fbbf24'
                      : '#f1f5f9';
            s.state.style.color = col;
        }
        if (s.person) {
            s.person.style.color = data.people_found ? '#a78bfa' : '#94a3b8';
        }
    }

    const yoloEls = {
        status:  document.getElementById('yolo-status-val'),
        conf:    document.getElementById('yolo-conf-val'),
        live:    document.getElementById('yolo-live-val'),
        thresh:  document.getElementById('yolo-thresh-val'),
        frame:   document.getElementById('yolo-frame-val'),
        person:  document.getElementById('yolo-person-val'),
        alert:      document.getElementById('yolo-hud-alert'),
        alertIcon:  document.getElementById('yolo-alert-icon'),
        alertLabel: document.getElementById('yolo-alert-label'),
        alertConf:  document.getElementById('yolo-alert-conf'),
        alertMeter: document.getElementById('yolo-alert-meter'),
        alertThresh:document.getElementById('yolo-alert-thresh'),
    };

    function updateYoloStats(data) {
        const ys = data && data.yolo_stats;
        if (!ys) return;
        const e = yoloEls;

        // best_conf now reflects the persistent detection peak (never collapses to
        // 0 mid-scan); current_conf is the instantaneous per-frame value.
        const best = (typeof ys.best_conf === 'number') ? ys.best_conf : 0;
        const live = (typeof ys.current_conf === 'number') ? ys.current_conf : best;
        const thresh = ys.conf_threshold || 0.7;
        const bestPct = best * 100;

        if (e.conf)   e.conf.textContent = bestPct.toFixed(1) + '%';
        if (e.live)   e.live.textContent = (live * 100).toFixed(1) + '%';
        if (e.thresh) e.thresh.textContent = (thresh * 100).toFixed(0) + '%';
        if (e.frame)  e.frame.textContent = ys.detection_count || 0;
        if (e.person) {
            e.person.textContent = ys.person_found ? 'FOUND' : 'Not found';
            e.person.style.color = ys.person_found ? 'var(--neon-magenta)' : 'var(--text-secondary)';
        }

        // ---- Resolve detection state ----
        let label, state;
        if (ys.person_found) {
            label = 'TARGET CONFIRMED'; state = 'confirmed';
        } else if (best >= thresh) {
            label = 'HUMAN DETECTED';   state = 'detected';
        } else if (best > 0) {
            label = 'CONTACT · TRACKING'; state = 'seen';
        } else {
            label = 'SCANNING'; state = 'idle';
        }

        if (e.status) {
            e.status.textContent = label;
            e.status.dataset.state = state;
        }

        // ---- Drive the HUD alert overlay ----
        if (e.alert) e.alert.dataset.state = state;
        if (e.alertLabel) {
            e.alertLabel.textContent = state === 'idle' ? 'SCANNING FOR HUMANS' : label;
        }
        if (e.alertConf) {
            e.alertConf.innerHTML = bestPct.toFixed(1) + '<span class="hud-alert-pct">%</span>';
        }
        if (e.alertIcon) {
            e.alertIcon.textContent = state === 'idle' ? '◎'
                : (state === 'confirmed' ? '✓' : '⚠');
        }
        if (e.alertMeter) {
            e.alertMeter.style.width = Math.max(0, Math.min(100, bestPct)).toFixed(1) + '%';
        }
        if (e.alertThresh) {
            e.alertThresh.style.left = (thresh * 100).toFixed(1) + '%';
        }
    }

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

            if (target === 'navigation') {
                setTimeout(() => navScene._handleResize(), 50);
            } else if (target === 'slam') {
                setTimeout(() => {
                    slam3dScene._handleResize();
                    if (slam2dMap && slam2dMap._onResize) slam2dMap._onResize();
                }, 50);
            } else if (target === 'yolo') {
                setTimeout(() => {
                    if (yoloHud && yoloHud._onResize) yoloHud._onResize();
                }, 50);
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
            if (attitudeHud) attitudeHud.update(data);
            updateSlamStats(data);
            updateYoloStats(data);
            slam3dScene.update(data.slam_3d || null, data.room_bounds || null);
            if (slam2dMap && data.images && data.images.slam_map) {
                slam2dMap.update(data.images.slam_map);
            }
            if (yoloHud && data.images && data.images.yolo_hud) {
                yoloHud.update(data.images.yolo_hud);
            }
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
