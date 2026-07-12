/**
 * app.js — Dashboard bootstrap, tab switching, level selection, WebSocket manager
 */

(function () {
    'use strict';

    const qs = new URLSearchParams(window.location.search);
    const wsPortQuery = Number.parseInt(qs.get('ws_port') || '', 10);
    const WS_PORT = Number.isFinite(wsPortQuery) && wsPortQuery > 0 ? wsPortQuery : 8001;
    const WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const WS_URL = `${WS_PROTOCOL}://${window.location.hostname || 'localhost'}:${WS_PORT}`;
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
    const flightControlPanel = document.getElementById('panel-flight-control')
        ? new FlightControlPanel() : null;
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
        crash:     document.getElementById('slam-crash-val'),
    };

    const brainEls = {
        state:    document.getElementById('brain-state-val'),
        segment:  document.getElementById('brain-seg-val'),
        waypoint: document.getElementById('brain-wp-val'),
        target:   document.getElementById('brain-target-val'),
        stuck:    document.getElementById('brain-stuck-val'),
        explore:  document.getElementById('brain-explore-val'),
        mission:  document.getElementById('brain-mission-val'),
    };

    const missionEls = {
        badge:   document.getElementById('mission-status'),
        state:   document.getElementById('mission-state-val'),
        detail:  document.getElementById('mission-detail-val'),
        targets: document.getElementById('mission-targets-val'),
    };

    function updateMissionHeader(data) {
        const ms = (data && data.mission_status) || {};
        const spawn = (data && data.spawn_info) || {};
        const status = ms.status || data.slam_state || '—';
        let found = (ms.targets_found != null && ms.targets_found !== '')
            ? String(ms.targets_found)
            : `${spawn.detected || 0}/${spawn.total || 0}`;
        if (playbackMode && loadedRecordingMeta) {
            const metaTotal = Number(loadedRecordingMeta.targets_total);
            const metaFound = Number(loadedRecordingMeta.targets_found);
            if (Number.isFinite(metaTotal) && Number.isFinite(metaFound) && metaTotal >= 0) {
                found = `${metaFound}/${metaTotal}`;
            }
        }
        const crashReason = ms.crash_reason || ms.detail || '';

        if (missionEls.state) {
            missionEls.state.textContent = status;
            const col = status === 'COMPLETE' ? '#fbbf24'
                : status === 'STUCK' ? '#fb923c'
                : status === 'CRASH' ? '#f87171'
                : status === 'EXPLORE' ? '#34d399'
                : status === 'SCAN' ? '#22d3ee'
                : '#e2e8f0';
            missionEls.state.style.color = col;
        }
        if (missionEls.detail) {
            if (status === 'CRASH' && crashReason) {
                missionEls.detail.hidden = false;
                missionEls.detail.textContent = crashReason;
                missionEls.detail.style.color = '#fca5a5';
            } else if (status === 'STUCK' && ms.detail) {
                missionEls.detail.hidden = false;
                missionEls.detail.textContent = ms.detail;
                missionEls.detail.style.color = '#fdba74';
            } else {
                missionEls.detail.hidden = true;
                missionEls.detail.textContent = '';
            }
        }
        if (missionEls.targets) {
            missionEls.targets.textContent = found;
        }
        if (missionEls.badge) {
            const tip = status === 'CRASH' && crashReason
                ? `MODE ${status} — ${crashReason} · FOUND ${found}`
                : `MODE ${status} · FOUND ${found}`;
            missionEls.badge.title = tip;
        }
    }

    function updateBrainStats(data) {
        const bt = (data && data.brain_telemetry) || {};
        const ms = (data && data.mission_status) || {};
        const b = brainEls;
        if (b.state) {
            b.state.textContent = bt.state || '—';
            b.state.style.color = bt.state === 'COMPLETE' ? '#fbbf24'
                : bt.state === 'EXPLORE' ? '#34d399'
                : bt.state === 'SCAN' ? '#22d3ee' : '#f1f5f9';
        }
        if (b.segment) {
            const lbl = bt.segment_label ? ` · ${bt.segment_label}` : '';
            b.segment.textContent = `${bt.segment_idx ?? 0}${lbl}`;
        }
        if (b.waypoint) {
            const total = bt.waypoint_total || bt.path_nodes || 0;
            b.waypoint.textContent = total > 0
                ? `${bt.waypoint_idx || 0} / ${total}`
                : `${bt.path_nodes || 0} nodes`;
        }
        if (b.target) {
            if (bt.nav_target && bt.nav_target.length >= 2) {
                b.target.textContent = `(${bt.nav_target[0].toFixed(1)}, ${bt.nav_target[1].toFixed(1)})`;
                b.target.style.color = '#2ff3ff';
            } else if (data.slam_goal && data.slam_goal.length >= 2) {
                b.target.textContent = `(${data.slam_goal[0].toFixed(1)}, ${data.slam_goal[1].toFixed(1)})`;
                b.target.style.color = '#2ff3ff';
            } else {
                b.target.textContent = 'None';
                b.target.style.color = '#94a3b8';
            }
        }
        if (b.stuck) {
            const stuck = Math.max(bt.stuck_steps || 0, bt.stuck_ticks || 0);
            b.stuck.textContent = String(stuck);
            b.stuck.style.color = stuck > 90 ? '#fb923c' : '#f1f5f9';
        }
        if (b.explore) b.explore.textContent = String(bt.explore_steps || 0);
        if (b.mission) {
            const crash = ms.crash_reason || '';
            if (ms.status === 'CRASH' && crash) {
                b.mission.textContent = crash;
                b.mission.style.color = '#f87171';
            } else {
                b.mission.textContent = ms.status || (bt.mission_finished ? 'COMPLETE' : 'ACTIVE');
                b.mission.style.color = (ms.status === 'COMPLETE' || bt.mission_finished)
                    ? '#fbbf24' : '#34d399';
            }
        }
    }

    function updateSlamStats(data) {
        if (!data) return;
        const s = slamEls;
        if (s.state)     s.state.textContent  = data.slam_state || '—';
        if (s.explored)  s.explored.textContent = (data.map_explored_pct || 0).toFixed(1) + '%';
        if (s.frontiers) s.frontiers.textContent = data.frontier_count || 0;
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
        if (s.crash) {
            const ms = (data && data.mission_status) || {};
            const crash = ms.crash_reason || ms.detail || '';
            if ((ms.status === 'CRASH' || ms.crashed) && crash) {
                s.crash.textContent = crash;
                s.crash.style.color = '#f87171';
            } else {
                s.crash.textContent = '—';
                s.crash.style.color = '#64748b';
            }
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
            const spawn = data.spawn_info || {};
            if (spawn.total > 0) {
                s.person.textContent = `${spawn.detected || 0}/${spawn.total} found`;
            } else {
                s.person.textContent = data.people_found ? 'FOUND' : 'Not found';
            }
            s.person.style.color = (spawn.detected > 0 || data.people_found) ? '#a78bfa' : '#94a3b8';
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
            } else if (target === 'cameras') {
                setTimeout(() => {
                    if (cameraFeeds) {
                        if (typeof cameraFeeds.resetToGrid === 'function') cameraFeeds.resetToGrid();
                        else if (typeof cameraFeeds._onResize === 'function') cameraFeeds._onResize();
                    }
                }, 50);
            } else if (target === 'slam') {
                setTimeout(() => {
                    slam3dScene._handleResize();
                    if (slam2dMap && slam2dMap._onResize) slam2dMap._onResize();
                }, 50);
            } else if (target === 'yolo') {
                setTimeout(() => {
                    if (yoloHud && yoloHud._onResize) yoloHud._onResize();
                }, 60);
            } else if (target === 'recordings') {
                if (typeof loadRecordingsList === 'function') loadRecordingsList();
                setTimeout(() => {
                    if (playbackFrames.length) drawFlightLogTimeline(playbackFrames, currentFrameIndex);
                }, 80);
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

    // ---- Spawn Targets panel (enabled only when sim is live) ----
    const spawnBtn = document.getElementById('btn-spawn-targets');
    const spawnCountInput = document.getElementById('input-spawn-count');
    const spawnStatusEl = document.getElementById('spawn-status');
    let simRunning = false;

    function updateSpawnPanel(data) {
        simRunning = Boolean(data && data.sim_running);
        const spawn = (data && data.spawn_info) || {};
        const canSpawn = simRunning && ws && ws.readyState === WebSocket.OPEN;
        if (spawnBtn) {
            spawnBtn.disabled = !canSpawn;
        }
        if (spawnCountInput) {
            spawnCountInput.disabled = !canSpawn;
        }
        if (!spawnStatusEl) return;
        if (!simRunning) {
            spawnStatusEl.textContent = 'Waiting for sim…';
        } else if (spawn.pending != null) {
            spawnStatusEl.textContent = `Spawning ${spawn.pending}…`;
        } else if (spawn.active && spawn.total > 0) {
            spawnStatusEl.textContent = `${spawn.detected}/${spawn.total} found · ${Math.round(data.map_explored_pct || 0)}% map`;
        } else if (canSpawn) {
            spawnStatusEl.textContent = 'Ready — click Spawn';
        } else {
            spawnStatusEl.textContent = 'Connect to sim';
        }
    }

    if (spawnBtn) {
        spawnBtn.addEventListener('click', () => {
            if (spawnBtn.disabled) return;
            const count = parseInt(spawnCountInput?.value || '2', 10);
            sendCommand({
                command: 'spawn_random_targets',
                count: Math.max(1, Math.min(15, count)),
            });
            spawnBtn.textContent = 'Sent';
            spawnBtn.disabled = true;
            if (spawnStatusEl) spawnStatusEl.textContent = 'Command sent…';
            setTimeout(() => {
                spawnBtn.textContent = 'Spawn';
                if (simRunning) spawnBtn.disabled = false;
            }, 1500);
        });
    }

    // ---- WebSocket ----
    function setStatus(state) {
        statusBadge.className = 'header-badge ' + state;
        statusText.textContent = state === 'connected' ? 'CONNECTED'
            : state === 'disconnected' ? 'DISCONNECTED'
            : state === 'playback' ? 'PLAYBACK' : 'CONNECTING…';
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

        if (data.level_time !== undefined) episodeTimeEl.textContent = formatTime(data.level_time, true);
        if (data.level_duration !== undefined) episodeDurationEl.textContent = formatTime(data.level_duration, false);
        if (data.tick !== undefined) tickCounterEl.textContent = data.tick.toLocaleString();
    }

    function renderTelemetryFrame(data, index = 0, isPlayback = false) {
        updateHeader(data);
        updateMissionHeader(data);
        updateSpawnPanel(data);
        cameraFeeds.update(data.images);
        metricsPanel.update(data);
        if (flightControlPanel) flightControlPanel.update(data);
        updateBrainStats(data);
        
        if (isPlayback) {
            liveCharts.showHistory(playbackFrames, index);
        } else {
            liveCharts.update(data);
        }
        
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
    }

    function onMessage(event) {
        if (playbackMode) return;
        try {
            const data = JSON.parse(event.data);
            renderTelemetryFrame(data, 0, false);
        } catch (e) {
            console.error('[Dashboard] Parse error:', e);
        }
    }

    function connect() {
        if (playbackMode) return;
        if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
        setStatus('connecting');
        ws = new WebSocket(WS_URL);
        ws.onopen = () => {
            setStatus('connected');
            if (spawnStatusEl && !simRunning) {
                spawnStatusEl.textContent = 'Waiting for sim…';
            }
            if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        };
        ws.onmessage = onMessage;
        ws.onclose = () => { setStatus('disconnected'); scheduleReconnect(); };
        ws.onerror = () => ws.close();
    }

    function scheduleReconnect() {
        if (playbackMode) return;
        if (reconnectTimer) return;
        reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, RECONNECT_DELAY);
    }

    // ---- Telemetry Playback Controls ----
    let playbackMode = false;
    let playbackFrames = [];
    let currentFrameIndex = 0;
    let playbackIntervalId = null;
    let playbackSpeed = 1;
    let wasPlayingBeforeDrag = false;

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const playbackControls = document.getElementById('playback-controls');
    const btnPlay = document.getElementById('btn-play');
    const btnPause = document.getElementById('btn-pause');
    const btnStop = document.getElementById('btn-stop');
    const timelineSlider = document.getElementById('timeline-slider');
    const timelineTime = document.getElementById('timeline-time');
    const playbackSpeedSelect = document.getElementById('playback-speed');
    const playbackStatusBadge = document.getElementById('playback-status-badge');

    // Global Playback Controls
    const globalPlaybackBar = document.getElementById('global-playback-bar');
    const globalBtnPlay = document.getElementById('global-btn-play');
    const globalBtnPause = document.getElementById('global-btn-pause');
    const globalBtnStop = document.getElementById('global-btn-stop');
    const globalBtnPrev = document.getElementById('global-btn-prev');
    const globalBtnNext = document.getElementById('global-btn-next');
    const globalTimelineSlider = document.getElementById('global-timeline-slider');
    const globalTimelineTime = document.getElementById('global-timeline-time');
    const globalPlaybackSpeed = document.getElementById('global-playback-speed');

    // Directory list elements
    const btnRefreshRecordings = document.getElementById('btn-refresh-recordings');
    const recordingsListBody = document.getElementById('recordings-list-body');
    const recCountVal = document.getElementById('rec-count-val');
    const recLoadedVal = document.getElementById('rec-loaded-val');
    const recFramesVal = document.getElementById('rec-frames-val');
    const recNowName = document.getElementById('rec-now-name');
    const recDeckTime = document.getElementById('rec-deck-time');
    const recFdrCanvas = document.getElementById('rec-fdr-canvas');
    const recFdrAxis = document.getElementById('rec-fdr-axis');
    const recFdrSource = document.getElementById('rec-fdr-source');
    const recLiveEls = {
        alt:   document.getElementById('rec-live-alt'),
        gs:    document.getElementById('rec-live-gs'),
        hdg:   document.getElementById('rec-live-hdg'),
        mode:  document.getElementById('rec-live-mode'),
        pos:   document.getElementById('rec-live-pos'),
        event: document.getElementById('rec-live-event'),
    };
    const inspEls = {
        filename: document.getElementById('insp-filename'),
        duration: document.getElementById('insp-duration'),
        frames:   document.getElementById('insp-frames'),
        size:     document.getElementById('insp-size'),
        level:    document.getElementById('insp-level'),
        status:   document.getElementById('insp-status'),
        targets:  document.getElementById('insp-targets'),
        coverage: document.getElementById('insp-coverage'),
        crash:    document.getElementById('insp-crash'),
        tick:     document.getElementById('insp-tick'),
        slam:     document.getElementById('insp-slam'),
        pos:      document.getElementById('insp-pos'),
        speed:    document.getElementById('insp-speed'),
    };
    let loadedRecordingMeta = null;
    let loadedRecordingName = '';

    function statusClass(status) {
        const s = (status || '').toUpperCase();
        if (s === 'CRASH') return 'rec-st-crash';
        if (s === 'COMPLETE') return 'rec-st-complete';
        if (s === 'STUCK') return 'rec-st-stuck';
        if (s === 'EXPLORE' || s === 'SCAN') return 'rec-st-active';
        return 'rec-st-idle';
    }

    function _frameThrust(f) {
        const llc = f.llc_outputs || {};
        if (llc.thrust != null) return llc.thrust;
        const fc = f.flight_control || {};
        const la = fc.ll_actions || [];
        if (la.length) return la[0];
        const acts = f.actions || [];
        return acts.length ? acts[0] : 0;
    }

    function _sampleFlightLog(frames) {
        const channels = { alt: [], gs: [], map: [], thr: [], crash: [] };
        const step = Math.max(1, Math.floor(frames.length / 320));
        for (let i = 0; i < frames.length; i += step) {
            const f = frames[i];
            const alt = f.pos ? f.pos[2] : 0;
            const gs = f.lin_vel ? Math.hypot(f.lin_vel[0], f.lin_vel[1], f.lin_vel[2]) : 0;
            const map = f.map_explored_pct || 0;
            const thr = _frameThrust(f);
            const ms = f.mission_status || {};
            const crashed = ms.status === 'CRASH' || ms.crashed;
            channels.alt.push(alt);
            channels.gs.push(gs);
            channels.map.push(map);
            channels.thr.push(thr);
            channels.crash.push(crashed ? 1 : 0);
        }
        return channels;
    }

    function drawFlightLogTimeline(frames, playhead = 0) {
        if (!recFdrCanvas || !frames || !frames.length) return;
        const canvas = recFdrCanvas;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const w = rect.width;
        const h = rect.height;
        const channels = _sampleFlightLog(frames);
        const n = channels.alt.length;
        if (!n) return;

        const chH = h / 4;
        const specs = [
            { key: 'alt', color: '#38bdf8', label: 'ALT' },
            { key: 'gs', color: '#34d399', label: 'GS' },
            { key: 'map', color: '#a78bfa', label: 'MAP' },
            { key: 'thr', color: '#fbbf24', label: 'THR' },
        ];

        ctx.fillStyle = '#0a0e14';
        ctx.fillRect(0, 0, w, h);

        specs.forEach((spec, ci) => {
            const y0 = ci * chH;
            const data = channels[spec.key];
            const max = Math.max(...data, 0.001);
            const min = Math.min(...data, 0);
            const range = Math.max(max - min, 0.001);

            ctx.fillStyle = 'rgba(255,255,255,0.02)';
            ctx.fillRect(0, y0, w, chH);
            for (let gy = y0; gy < y0 + chH; gy += chH / 4) {
                ctx.strokeStyle = 'rgba(255,255,255,0.05)';
                ctx.beginPath();
                ctx.moveTo(0, gy);
                ctx.lineTo(w, gy);
                ctx.stroke();
            }

            ctx.strokeStyle = spec.color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < n; i++) {
                const x = (i / Math.max(n - 1, 1)) * w;
                const norm = (data[i] - min) / range;
                const y = y0 + chH - 4 - norm * (chH - 8);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();

            for (let i = 0; i < n; i++) {
                if (!channels.crash[i]) continue;
                const x = (i / Math.max(n - 1, 1)) * w;
                ctx.fillStyle = '#f87171';
                ctx.beginPath();
                ctx.moveTo(x, y0 + 4);
                ctx.lineTo(x - 4, y0 + 12);
                ctx.lineTo(x + 4, y0 + 12);
                ctx.closePath();
                ctx.fill();
            }
        });

        const ph = (playhead / Math.max(frames.length - 1, 1)) * w;
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(ph, 0);
        ctx.lineTo(ph, h);
        ctx.stroke();
        ctx.setLineDash([]);

        const frame = frames[playhead] || frames[0];
        if (recFdrAxis && frame) {
            recFdrAxis.textContent = `T+ ${formatTime(frame.level_time || 0)} · frame ${playhead + 1}/${frames.length}`;
        }
    }

    function updateFlightLogReadouts(frame) {
        if (!frame) return;
        const r = recLiveEls;
        if (r.alt && frame.pos) r.alt.textContent = frame.pos[2].toFixed(2) + ' m';
        if (r.gs && frame.lin_vel) {
            const [vx, vy, vz] = frame.lin_vel;
            r.gs.textContent = Math.hypot(vx, vy, vz).toFixed(2);
        }
        if (r.hdg && frame.yaw != null) {
            let deg = (frame.yaw * 180 / Math.PI) % 360;
            if (deg < 0) deg += 360;
            r.hdg.textContent = deg.toFixed(0) + '°';
        }
        const ms = frame.mission_status || {};
        if (r.mode) {
            r.mode.textContent = ms.status || frame.slam_state || '—';
            r.mode.style.color = ms.status === 'CRASH' ? '#f87171'
                : frame.slam_state === 'EXPLORE' ? '#34d399' : '#e2e8f0';
        }
        if (r.pos && frame.pos) {
            r.pos.textContent = `${frame.pos[0].toFixed(1)}, ${frame.pos[1].toFixed(1)}`;
        }
        if (r.event) {
            const crash = ms.crash_reason || '';
            if ((ms.status === 'CRASH' || ms.crashed) && crash) {
                r.event.textContent = crash;
                r.event.style.color = '#f87171';
            } else {
                r.event.textContent = 'Nominal';
                r.event.style.color = '#64748b';
            }
        }
    }

    function updateRecordingInspector(frame, index) {
        const meta = loadedRecordingMeta || {};
        const last = playbackFrames.length ? playbackFrames[playbackFrames.length - 1] : null;
        const cur = frame || last;
        const ms = (cur && cur.mission_status) || {};
        const spawn = (cur && cur.spawn_info) || (last && last.spawn_info) || {};
        const totalDur = last ? (last.level_time || 0) : 0;

        if (inspEls.filename) inspEls.filename.textContent = loadedRecordingName || '—';
        if (inspEls.duration) {
            inspEls.duration.textContent = totalDur
                ? formatTime(totalDur)
                : '—';
        }
        if (inspEls.frames) inspEls.frames.textContent = playbackFrames.length ? String(playbackFrames.length) : '—';
        if (inspEls.size) inspEls.size.textContent = meta.file_size || '—';
        if (inspEls.level) inspEls.level.textContent = String((cur && cur.level) || (last && last.level) || 1);

        const inspElapsed = document.getElementById('insp-elapsed');
        if (inspElapsed) {
            if (cur && totalDur) {
                inspElapsed.textContent = `${formatTime(cur.level_time || 0)} / ${formatTime(totalDur)}`;
            } else {
                inspElapsed.textContent = '—';
            }
        }

        if (inspEls.status) {
            const st = ms.status || (cur && cur.slam_state) || '—';
            inspEls.status.textContent = st;
            inspEls.status.className = 'mono rec-insp-val ' + statusClass(st);
        }
        if (inspEls.targets) {
            inspEls.targets.textContent = `${spawn.detected || 0} / ${spawn.total || 0}`;
        }
        if (inspEls.coverage && cur) {
            inspEls.coverage.textContent = (cur.map_explored_pct || 0).toFixed(1) + '%';
        }
        if (inspEls.crash) {
            const crash = ms.crash_reason || '';
            if ((ms.status === 'CRASH' || ms.crashed) && crash) {
                inspEls.crash.textContent = crash;
                inspEls.crash.style.color = '#f87171';
            } else {
                inspEls.crash.textContent = 'None';
                inspEls.crash.style.color = '#64748b';
            }
        }
        if (cur) {
            if (inspEls.tick) inspEls.tick.textContent = cur.tick != null ? String(cur.tick) : String(index);
            if (inspEls.slam) inspEls.slam.textContent = cur.slam_state || '—';
            if (inspEls.pos && cur.pos) {
                inspEls.pos.textContent = `(${cur.pos[0].toFixed(1)}, ${cur.pos[1].toFixed(1)}, ${cur.pos[2].toFixed(1)})`;
            }
            if (inspEls.speed && cur.lin_vel) {
                const [vx, vy, vz] = cur.lin_vel;
                inspEls.speed.textContent = Math.hypot(vx, vy, vz).toFixed(2) + ' m/s';
            }
        }
    }

    function parseRecordingLines(text) {
        const lines = text.split('\n');
        const frames = [];
        let sessionHeader = null;
        for (const line of lines) {
            if (!line.trim()) continue;
            try {
                const obj = JSON.parse(line);
                if (obj._record_type === 'session_header') {
                    sessionHeader = obj;
                    continue;
                }
                if (obj.pos || obj.tick != null || obj.level_time != null) {
                    frames.push(obj);
                }
            } catch (err) { /* skip bad lines */ }
        }
        return { frames, sessionHeader };
    }

    function onRecordingLoaded(name, meta) {
        loadedRecordingName = name;
        loadedRecordingMeta = meta || null;
        if (recLoadedVal) recLoadedVal.textContent = name.length > 22 ? name.slice(0, 20) + '…' : name;
        if (recFramesVal) recFramesVal.textContent = String(playbackFrames.length);
        if (recNowName) recNowName.textContent = name;
        if (dropZone) {
            dropZone.classList.add('rec-import-loaded');
            const txt = dropZone.querySelector('.rec-import-text');
            if (txt) txt.textContent = name;
        }
        drawFlightLogTimeline(playbackFrames, 0);
        updateFlightLogReadouts(playbackFrames[0] || null);
        updateRecordingInspector(playbackFrames[0] || null, 0);
        if (recFdrSource) {
            const n = playbackFrames.length;
            const hasLlc = playbackFrames.some(f => f.llc_outputs && f.llc_outputs.thrust != null);
            const hasMission = playbackFrames.some(f => f.mission_status);
            recFdrSource.textContent = `Source: ${n} JSONL frames · ALT/GS/MAP/THR from log${hasLlc ? '' : ' (THR approx)'}${hasMission ? '' : ' · no mission_status'}`;
        }
    }

    let recordingsCache = {};

    function loadRecordingsList() {
        if (!recordingsListBody) return;
        recordingsListBody.innerHTML = '<div class="rec-lib-empty">Scanning recordings directory…</div>';
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);
        fetch('/api/recordings', { signal: controller.signal })
            .then(res => {
                if (!res.ok) {
                    throw new Error(`Server returned error status ${res.status}`);
                }
                return res.json();
            })
            .then(data => {
                recordingsCache = {};
                if (recCountVal) recCountVal.textContent = String(data ? data.length : 0);
                if (!data || data.length === 0) {
                    recordingsListBody.innerHTML = '<div class="rec-lib-empty">No recorded flights in recordings/</div>';
                    return;
                }

                recordingsListBody.innerHTML = '';
                data.forEach(rec => {
                    recordingsCache[rec.filename] = rec;
                    const row = document.createElement('button');
                    row.type = 'button';
                    row.className = 'rec-lib-row play-remote-btn';
                    row.dataset.file = rec.filename;
                    if (loadedRecordingName === rec.filename) row.classList.add('rec-lib-row-active');

                    const st = rec.status || '—';
                    const crashTip = rec.crash_reason ? ` · ${rec.crash_reason}` : '';
                    row.title = `${rec.filename} · ${st}${crashTip}`;

                    row.innerHTML = `
                        <span class="rec-lib-date">${rec.date}</span>
                        <span class="rec-lib-status ${statusClass(st)}">${st}</span>
                        <span class="rec-lib-found mono">${rec.targets_found}/${rec.targets_total}</span>
                        <span class="rec-lib-cov mono">${rec.coverage.toFixed(1)}%</span>
                        <span class="rec-lib-dur mono">${formatTime(rec.duration)}</span>
                    `;
                    recordingsListBody.appendChild(row);
                });

                recordingsListBody.querySelectorAll('.play-remote-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        loadRemoteRecording(btn.dataset.file);
                    });
                });
            })
            .catch(err => {
                console.error("[Playback] Failed to load recordings list:", err);
                const timedOut = err && err.name === 'AbortError';
                recordingsListBody.innerHTML = timedOut
                    ? '<div class="rec-lib-empty rec-lib-error">Loading recordings timed out — check recordings path/server logs.</div>'
                    : '<div class="rec-lib-empty rec-lib-error">Failed to load — is server.py running?</div>';
            })
            .finally(() => {
                clearTimeout(timeoutId);
            });
    }

    async function loadRemoteRecording(filename) {
        if (recordingsListBody) {
            recordingsListBody.querySelectorAll('.play-remote-btn').forEach(b => {
                b.classList.toggle('rec-lib-loading', b.dataset.file === filename);
                b.style.removeProperty('--load-progress');
            });
        }

        const loadBtn = recordingsListBody ? recordingsListBody.querySelector(`.play-remote-btn[data-file="${filename}"]`) : null;
        const statusSpan = loadBtn ? loadBtn.querySelector('.rec-lib-status') : null;
        if (statusSpan) {
            statusSpan.textContent = '0%';
        }

        const cached = recordingsCache[filename] || {};
        let estimatedTotalBytes = 0;
        if (cached.file_size) {
            const num = parseFloat(cached.file_size);
            if (cached.file_size.includes("MB")) estimatedTotalBytes = num * 1024 * 1024;
            else if (cached.file_size.includes("KB")) estimatedTotalBytes = num * 1024;
            else estimatedTotalBytes = num;
        }

        try {
            const response = await fetch(`/api/recording?file=${encodeURIComponent(filename)}`);
            if (!response.ok) throw new Error("Server returned error status " + response.status);

            const reader = response.body.getReader();
            
            let receivedLength = 0;
            const decoder = new TextDecoder("utf-8");
            let partialLine = "";
            const frames = [];
            let sessionHeader = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                receivedLength += value.length;

                let percent = 0;
                if (estimatedTotalBytes > 0) {
                    percent = Math.min(99, Math.round((receivedLength / estimatedTotalBytes) * 100));
                }

                if (loadBtn) {
                    loadBtn.style.setProperty('--load-progress', `${percent}%`);
                    if (statusSpan) {
                        const mb = (receivedLength / (1024 * 1024)).toFixed(1);
                        statusSpan.textContent = percent > 0 ? `${percent}%` : `${mb}MB`;
                    }
                }

                // Decode chunk and split line-by-line progressively
                const chunkStr = decoder.decode(value, { stream: true });
                const combined = partialLine + chunkStr;
                const lines = combined.split('\n');
                partialLine = lines.pop(); // Save the last incomplete line for next iteration

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const obj = JSON.parse(line);
                        if (obj._record_type === 'session_header') {
                            sessionHeader = obj;
                        } else if (obj.pos || obj.tick != null || obj.level_time != null) {
                            frames.push(obj);
                        }
                    } catch (e) {}
                }
            }

            // Parse the final remaining line
            if (partialLine.trim()) {
                try {
                    const obj = JSON.parse(partialLine);
                    if (obj._record_type === 'session_header') {
                        sessionHeader = obj;
                    } else if (obj.pos || obj.tick != null || obj.level_time != null) {
                        frames.push(obj);
                    }
                } catch (e) {}
            }

            playbackFrames = frames;

            if (playbackFrames.length === 0) {
                alert("No valid telemetry frames found in this recording.");
                loadRecordingsList();
                return;
            }

            console.log(`[Playback] Loaded remote file ${filename} with ${playbackFrames.length} frames.`);
            playbackMode = true;
            currentFrameIndex = 0;
            stopPlaybackTimer();

            if (ws) {
                try { ws.close(); } catch(e) {}
            }
            setStatus('playback');
            document.querySelectorAll('.panel-badge.live').forEach(badge => {
                badge.textContent = 'OFFLINE';
            });

            if (playbackStatusBadge) {
                playbackStatusBadge.textContent = 'PLAYBACK';
                playbackStatusBadge.style.color = '#ff2d95';
                playbackStatusBadge.style.borderColor = 'rgba(255, 45, 149, 0.4)';
                playbackStatusBadge.style.background = 'rgba(255, 45, 149, 0.1)';
            }

            if (timelineSlider) {
                timelineSlider.min = 0;
                timelineSlider.max = playbackFrames.length - 1;
                timelineSlider.value = 0;
            }

            if (globalTimelineSlider) {
                globalTimelineSlider.min = 0;
                globalTimelineSlider.max = playbackFrames.length - 1;
                globalTimelineSlider.value = 0;
            }

            if (globalPlaybackBar) globalPlaybackBar.style.display = 'block';
            if (playbackControls) playbackControls.style.display = 'flex';

            const cached = recordingsCache[filename] || {};
            const meta = {
                ...cached,
                file_size: cached.file_size || formatFileSize(receivedLength),
                session_header: sessionHeader,
            };
            onRecordingLoaded(filename, meta);
            renderFrame(0);
            loadRecordingsList();

        } catch (err) {
            console.error("[Playback] Failed to load remote recording:", err);
            alert("Failed to load recording: " + err.message);
            loadRecordingsList();
        } finally {
            if (loadBtn) {
                loadBtn.classList.remove('rec-lib-loading');
                // The button text will be reset when loadRecordingsList() runs and re-renders the rows.
            }
        }
    }

    if (btnRefreshRecordings) {
        btnRefreshRecordings.addEventListener('click', (e) => {
            e.stopPropagation();
            loadRecordingsList();
        });
    }

    // Time formatter HH:MM:SS or MM:SS (with optional decimals)
    function formatTime(seconds, includeDecimals = false) {
        if (isNaN(seconds) || seconds === null) return includeDecimals ? '0:00.00' : '0:00';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        let timeStr = '';
        if (hrs > 0) {
            timeStr += hrs + ':' + (mins < 10 ? '0' : '') + mins + ':';
        } else {
            timeStr += mins + ':';
        }
        timeStr += (secs < 10 ? '0' : '') + secs;
        
        if (includeDecimals) {
            const decs = Math.floor((seconds % 1) * 100);
            timeStr += '.' + (decs < 10 ? '0' : '') + decs;
        }
        return timeStr;
    }

    // Drag-and-drop handlers
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('rec-import-hover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('rec-import-hover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('rec-import-hover');
            if (e.dataTransfer.files.length > 0) {
                loadRecordingFile(e.dataTransfer.files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                loadRecordingFile(e.target.files[0]);
            }
        });
    }

    function loadRecordingFile(file) {
        const reader = new FileReader();
        reader.onload = function(evt) {
            const text = evt.target.result;
            const parsed = parseRecordingLines(text);
            playbackFrames = parsed.frames;
            if (playbackFrames.length === 0) {
                alert("No valid telemetry frames found in this file.");
                return;
            }

            console.log(`[Playback] Loaded ${playbackFrames.length} frames.`);
            playbackMode = true;
            currentFrameIndex = 0;
            stopPlaybackTimer();
            
            // Suspend WebSocket
            if (ws) {
                try { ws.close(); } catch(e) {}
            }
            setStatus('playback');
            document.querySelectorAll('.panel-badge.live').forEach(badge => {
                badge.textContent = 'OFFLINE';
            });

            if (playbackStatusBadge) {
                playbackStatusBadge.textContent = 'PLAYBACK';
                playbackStatusBadge.style.color = '#ff2d95';
                playbackStatusBadge.style.borderColor = 'rgba(255, 45, 149, 0.4)';
                playbackStatusBadge.style.background = 'rgba(255, 45, 149, 0.1)';
            }
            
            if (timelineSlider) {
                timelineSlider.min = 0;
                timelineSlider.max = playbackFrames.length - 1;
                timelineSlider.value = 0;
            }
            
            if (globalTimelineSlider) {
                globalTimelineSlider.min = 0;
                globalTimelineSlider.max = playbackFrames.length - 1;
                globalTimelineSlider.value = 0;
            }
            
            if (globalPlaybackBar) globalPlaybackBar.style.display = 'block';
            if (playbackControls) playbackControls.style.display = 'flex';
            onRecordingLoaded(file.name, {
                file_size: formatFileSize(file.size),
                session_header: parsed.sessionHeader,
            });
            renderFrame(0);
        };
        reader.readAsText(file);
    }

    function formatFileSize(bytes) {
        if (!bytes || bytes < 1024) return bytes ? bytes + ' B' : '—';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    function renderFrame(index) {
        if (index < 0 || index >= playbackFrames.length) return;
        const frame = playbackFrames[index];
        renderTelemetryFrame(frame, index, true);

        if (timelineSlider) timelineSlider.value = index;
        if (globalTimelineSlider) globalTimelineSlider.value = index;

        const totalDuration = playbackFrames[playbackFrames.length - 1].level_time || 0;
        const formatted = `${formatTime(frame.level_time)} / ${formatTime(totalDuration)}`;
        if (timelineTime) timelineTime.textContent = formatted;
        if (globalTimelineTime) globalTimelineTime.textContent = formatted;
        if (recDeckTime) recDeckTime.textContent = formatted;
        drawFlightLogTimeline(playbackFrames, index);
        updateFlightLogReadouts(frame);
        updateRecordingInspector(frame, index);
    }

    function seekBySeconds(deltaSec) {
        if (!playbackFrames.length) return;
        const cur = playbackFrames[currentFrameIndex] || playbackFrames[0];
        const target = Math.max(0, (cur.level_time || 0) + deltaSec);
        let best = 0;
        for (let i = 0; i < playbackFrames.length; i++) {
            if ((playbackFrames[i].level_time || 0) <= target) best = i;
            else break;
        }
        currentFrameIndex = best;
        renderFrame(best);
    }

    function startPlaybackTimer() {
        stopPlaybackTimer();
        if (btnPlay) btnPlay.style.display = 'none';
        if (btnPause) btnPause.style.display = 'inline-flex';
        if (globalBtnPlay) globalBtnPlay.style.display = 'none';
        if (globalBtnPause) globalBtnPause.style.display = 'inline-block';
        
        const baseInterval = 100; // 10Hz base tick rate
        const delay = baseInterval / playbackSpeed;
        
        playbackIntervalId = setInterval(() => {
            currentFrameIndex++;
            if (currentFrameIndex >= playbackFrames.length) {
                stopPlaybackTimer();
                currentFrameIndex = playbackFrames.length - 1;
                if (btnPlay) btnPlay.style.display = 'inline-flex';
                if (btnPause) btnPause.style.display = 'none';
                if (globalBtnPlay) globalBtnPlay.style.display = 'inline-block';
                if (globalBtnPause) globalBtnPause.style.display = 'none';
            }
            renderFrame(currentFrameIndex);
        }, delay);
    }

    function stopPlaybackTimer() {
        if (playbackIntervalId) {
            clearInterval(playbackIntervalId);
            playbackIntervalId = null;
        }
        if (btnPlay) btnPlay.style.display = 'inline-flex';
        if (btnPause) btnPause.style.display = 'none';
        if (globalBtnPlay) globalBtnPlay.style.display = 'inline-block';
        if (globalBtnPause) globalBtnPause.style.display = 'none';
    }

    if (btnPlay) {
        btnPlay.addEventListener('click', () => {
            if (currentFrameIndex >= playbackFrames.length - 1) {
                currentFrameIndex = 0;
            }
            startPlaybackTimer();
        });
    }

    if (globalBtnPlay) {
        globalBtnPlay.addEventListener('click', () => {
            if (currentFrameIndex >= playbackFrames.length - 1) {
                currentFrameIndex = 0;
            }
            startPlaybackTimer();
        });
    }

    if (btnPause) {
        btnPause.addEventListener('click', () => {
            stopPlaybackTimer();
        });
    }

    if (globalBtnPause) {
        globalBtnPause.addEventListener('click', () => {
            stopPlaybackTimer();
        });
    }

    function stopPlaybackAndReconnect() {
        stopPlaybackTimer();
        playbackMode = false;
        playbackFrames = [];
        currentFrameIndex = 0;
        
        if (globalPlaybackBar) globalPlaybackBar.style.display = 'none';
        
        if (playbackStatusBadge) {
            playbackStatusBadge.textContent = 'OFFLINE';
            playbackStatusBadge.style.color = 'rgba(255, 255, 255, 0.6)';
            playbackStatusBadge.style.borderColor = 'rgba(255, 255, 255, 0.15)';
            playbackStatusBadge.style.background = 'rgba(255, 255, 255, 0.08)';
        }
        
        document.querySelectorAll('.panel-badge.live').forEach(badge => {
            badge.textContent = 'LIVE';
        });
        
        if (liveCharts && typeof liveCharts.reset === 'function') {
            liveCharts.reset();
        }
        
        if (playbackControls) playbackControls.style.display = 'flex';
        if (dropZone) {
            dropZone.classList.remove('rec-import-loaded');
            const txt = dropZone.querySelector('.rec-import-text');
            if (txt) txt.textContent = 'Import flight log (.jsonl)';
        }
        loadedRecordingName = '';
        loadedRecordingMeta = null;
        if (recLoadedVal) recLoadedVal.textContent = 'None';
        if (recFramesVal) recFramesVal.textContent = '—';
        if (recNowName) recNowName.textContent = '—';
        if (recFdrAxis) recFdrAxis.textContent = 'T+ 0:00';
        drawFlightLogTimeline([], 0);

        connect();
    }

    const btnRewind = document.getElementById('btn-rewind');
    const btnFfwd = document.getElementById('btn-ffwd');

    if (btnRewind) btnRewind.addEventListener('click', () => { stopPlaybackTimer(); seekBySeconds(-10); });
    if (btnFfwd) btnFfwd.addEventListener('click', () => { stopPlaybackTimer(); seekBySeconds(10); });

    if (btnStop) btnStop.addEventListener('click', stopPlaybackAndReconnect);
    if (globalBtnStop) globalBtnStop.addEventListener('click', stopPlaybackAndReconnect);

    if (globalBtnPrev) {
        globalBtnPrev.addEventListener('click', () => {
            stopPlaybackTimer();
            if (currentFrameIndex > 0) {
                currentFrameIndex--;
                renderFrame(currentFrameIndex);
            }
        });
    }

    if (globalBtnNext) {
        globalBtnNext.addEventListener('click', () => {
            stopPlaybackTimer();
            if (currentFrameIndex < playbackFrames.length - 1) {
                currentFrameIndex++;
                renderFrame(currentFrameIndex);
            }
        });
    }

    if (recFdrCanvas) {
        window.addEventListener('resize', () => {
            if (playbackFrames.length) drawFlightLogTimeline(playbackFrames, currentFrameIndex);
        });
    }

    function handleSliderScrub(e) {
        if (playbackIntervalId) {
            wasPlayingBeforeDrag = true;
            stopPlaybackTimer();
        } else {
            wasPlayingBeforeDrag = false;
        }
        currentFrameIndex = parseInt(e.target.value);
        renderFrame(currentFrameIndex);
    }

    function handleSliderRelease() {
        if (wasPlayingBeforeDrag) {
            startPlaybackTimer();
        }
    }

    if (timelineSlider) {
        timelineSlider.addEventListener('input', handleSliderScrub);
        timelineSlider.addEventListener('change', handleSliderRelease);
    }
    if (globalTimelineSlider) {
        globalTimelineSlider.addEventListener('input', handleSliderScrub);
        globalTimelineSlider.addEventListener('change', handleSliderRelease);
    }

    function handleSpeedChange(e) {
        playbackSpeed = parseFloat(e.target.value);
        if (playbackSpeedSelect) playbackSpeedSelect.value = playbackSpeed;
        if (globalPlaybackSpeed) globalPlaybackSpeed.value = playbackSpeed;
        if (playbackIntervalId) {
            startPlaybackTimer();
        }
    }

    if (playbackSpeedSelect) {
        playbackSpeedSelect.addEventListener('change', handleSpeedChange);
    }
    if (globalPlaybackSpeed) {
        globalPlaybackSpeed.addEventListener('change', handleSpeedChange);
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
