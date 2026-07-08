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
    const slam2dMap = document.getElementById('slam2d-canvas') ? new SlamMap2D('slam2d-canvas') : null;
    const yoloHud = document.getElementById('yolo-hud') ? new YoloHud('yolo-hud') : null;

    // SLAM tab stat elements
    const slamEls = {
        state:     document.getElementById('slam-state-val'),
        explored:  document.getElementById('slam-explored-val'),
        frontiers: document.getElementById('slam-frontiers-val'),
        person:    document.getElementById('slam-person-val'),
        x:         document.getElementById('slam-x-val'),
        y:         document.getElementById('slam-y-val'),
        alt:       document.getElementById('slam-alt-val'),
        hdg:       document.getElementById('slam-hdg-val'),
        spd:       document.getElementById('slam-spd-val'),
        time:      document.getElementById('slam-time-val'),
        goal:      document.getElementById('slam-goal-val'),
        astar:     document.getElementById('slam-astar-val'),
        dist:      document.getElementById('slam-dist-val'),
        level:     document.getElementById('slam-level-val'),
    };

    function updateSlamStats(data) {
        if (!data) return;
        const s = slamEls;
        if (s.state)     s.state.textContent  = data.slam_state || '—';
        if (s.explored)  s.explored.textContent = (data.map_explored_pct || 0).toFixed(1) + '%';
        if (s.frontiers) s.frontiers.textContent = data.frontier_count || 0;
        if (s.person)    s.person.textContent  = data.people_found ? 'FOUND' : 'Not found';
        if (data.pos) {
            if (s.x)   s.x.textContent   = data.pos[0].toFixed(2) + ' m';
            if (s.y)   s.y.textContent   = data.pos[1].toFixed(2) + ' m';
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
            s.time.textContent = data.level_time.toFixed(1) + ' s';
        }
        // OpenCV "Mapped Targets" column
        if (s.goal) {
            if (data.slam_goal && data.slam_goal.length >= 2) {
                s.goal.textContent = '(' + data.slam_goal[0].toFixed(1) + ', ' + data.slam_goal[1].toFixed(1) + ')';
            } else {
                s.goal.textContent = 'None';
            }
        }
        if (s.astar) s.astar.textContent = data.astar_nodes != null ? data.astar_nodes : 0;
        if (s.dist)  s.dist.textContent  = (data.dist_to_goal != null ? data.dist_to_goal : 0).toFixed(2) + ' m';
        if (s.level) s.level.textContent = data.level || 1;
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
        if (s.explored) {
            s.explored.style.color = (data.map_explored_pct || 0) > 50 ? '#34d399' : '#eafcff';
        }
        if (s.goal && data.slam_goal) {
            s.goal.style.color = '#2ff3ff';
        } else if (s.goal) {
            s.goal.style.color = '#94a3b8';
        }
    }

    function updateYoloStats(data) {
        if (yoloHud && data && data.yolo_stats) {
            yoloHud.update(data.yolo_stats);
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
                }, 60);
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
            if (slam2dMap) slam2dMap.update(data.slam_3d || null);
            if (yoloHud && data.images) {
                yoloHud.updateFrames(
                    data.images.yolo_frame || null,
                    data.images.yolo_frame_left || null,
                    data.images.yolo_frame_right || null
                );
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
