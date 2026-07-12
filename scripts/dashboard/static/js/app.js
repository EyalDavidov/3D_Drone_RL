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
            } else if (target === 'recordings') {
                if (typeof loadRecordingsList === 'function') {
                    loadRecordingsList();
                }
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

        if (data.level_time !== undefined) episodeTimeEl.textContent = data.level_time.toFixed(2) + 's';
        if (data.level_duration !== undefined) episodeDurationEl.textContent = data.level_duration.toFixed(2) + 's';
        if (data.tick !== undefined) tickCounterEl.textContent = data.tick.toLocaleString();
    }

    function renderTelemetryFrame(data, index = 0, isPlayback = false) {
        updateHeader(data);
        updateSpawnPanel(data);
        cameraFeeds.update(data.images);
        metricsPanel.update(data);
        
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

    // Directory list elements
    const btnRefreshRecordings = document.getElementById('btn-refresh-recordings');
    const recordingsListBody = document.getElementById('recordings-list-body');

    // Load list from backend HTTP server
    function loadRecordingsList() {
        if (!recordingsListBody) return;
        recordingsListBody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; padding: 20px 0; color: rgba(255, 255, 255, 0.4);">Scanning recordings directory...</td>
            </tr>
        `;
        
        fetch('/api/recordings')
            .then(res => res.json())
            .then(data => {
                if (!data || data.length === 0) {
                    recordingsListBody.innerHTML = `
                        <tr>
                            <td colspan="5" style="text-align: center; padding: 20px 0; color: rgba(255, 255, 255, 0.4);">No recorded flights found in recordings/ directory.</td>
                        </tr>
                    `;
                    return;
                }
                
                recordingsListBody.innerHTML = '';
                data.forEach(rec => {
                    const tr = document.createElement('tr');
                    tr.style.borderBottom = '1px solid rgba(255, 255, 255, 0.05)';
                    tr.style.color = 'rgba(255, 255, 255, 0.8)';
                    
                    const formattedDur = formatTime(rec.duration);
                    const formattedCov = rec.coverage.toFixed(1) + '%';
                    
                    tr.innerHTML = `
                        <td style="padding: 10px 8px; font-weight: 500;">${rec.date}</td>
                        <td style="padding: 10px 8px; text-align: center;" class="mono">${rec.targets_found} / ${rec.targets_total}</td>
                        <td style="padding: 10px 8px; text-align: center; color: #34d399;" class="mono">${formattedCov}</td>
                        <td style="padding: 10px 8px; text-align: center;" class="mono">${formattedDur}</td>
                        <td style="padding: 10px 8px; text-align: right;">
                            <button class="action-btn play-remote-btn" data-file="${rec.filename}" style="
                                background: rgba(255, 45, 149, 0.15);
                                color: #ff2d95;
                                border: 1px solid rgba(255, 45, 149, 0.35);
                                padding: 4px 10px;
                                border-radius: 4px;
                                cursor: pointer;
                                font-size: 0.8rem;
                                font-weight: 600;
                                font-family: inherit;
                            ">Select & Play</button>
                        </td>
                    `;
                    recordingsListBody.appendChild(tr);
                });
                
                // Add event listeners
                document.querySelectorAll('.play-remote-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const filename = btn.dataset.file;
                        loadRemoteRecording(filename);
                    });
                });
            })
            .catch(err => {
                console.error("[Playback] Failed to load recordings list:", err);
                recordingsListBody.innerHTML = `
                    <tr>
                        <td colspan="5" style="text-align: center; padding: 20px 0; color: #f43f5e;">Failed to load list. Make sure python server is running.</td>
                    </tr>
                `;
            });
    }

    function loadRemoteRecording(filename) {
        if (recordingsListBody) {
            recordingsListBody.querySelectorAll('.play-remote-btn').forEach(b => {
                b.disabled = true;
                if (b.dataset.file === filename) {
                    b.textContent = 'Loading...';
                }
            });
        }
        
        fetch(`/api/recording?file=${encodeURIComponent(filename)}`)
            .then(res => {
                if (!res.ok) throw new Error("Server returned error status " + res.status);
                return res.text();
            })
            .then(text => {
                const lines = text.split('\n');
                playbackFrames = [];
                for (let line of lines) {
                    if (line.trim()) {
                        try {
                            playbackFrames.push(JSON.parse(line));
                        } catch (err) {}
                    }
                }
                
                if (playbackFrames.length === 0) {
                    alert("No valid telemetry frames found in this recording.");
                    loadRecordingsList(); // reset buttons
                    return;
                }
                
                console.log(`[Playback] Loaded remote file ${filename} with ${playbackFrames.length} frames.`);
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
                
                if (playbackControls) playbackControls.style.display = 'block';
                if (dropZone) {
                    dropZone.style.borderColor = '#22ff66';
                    dropZone.querySelector('p').textContent = `Loaded: ${filename}`;
                }
                
                renderFrame(0);
                loadRecordingsList(); // refresh list buttons to normal state
            })
            .catch(err => {
                alert("Failed to load recording: " + err.message);
                loadRecordingsList();
            });
    }

    if (btnRefreshRecordings) {
        btnRefreshRecordings.addEventListener('click', (e) => {
            e.stopPropagation();
            loadRecordingsList();
        });
    }

    // Time formatter MM:SS
    function formatTime(seconds) {
        if (isNaN(seconds) || seconds === null) return '0:00';
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
    }

    // Drag-and-drop handlers
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.background = 'rgba(255, 45, 149, 0.08)';
            dropZone.style.borderColor = '#ff2d95';
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.style.background = 'rgba(255, 45, 149, 0.02)';
            dropZone.style.borderColor = 'rgba(255, 45, 149, 0.45)';
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.background = 'rgba(255, 45, 149, 0.02)';
            dropZone.style.borderColor = 'rgba(255, 45, 149, 0.45)';
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
            const lines = text.split('\n');
            playbackFrames = [];
            for (let line of lines) {
                if (line.trim()) {
                    try {
                        playbackFrames.push(JSON.parse(line));
                    } catch (err) {
                        // ignore parse errors
                    }
                }
            }
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
            
            if (playbackControls) playbackControls.style.display = 'block';
            if (dropZone) {
                dropZone.style.borderColor = '#22ff66';
                dropZone.querySelector('p').textContent = `Loaded: ${file.name}`;
            }
            
            renderFrame(0);
        };
        reader.readAsText(file);
    }

    function renderFrame(index) {
        if (index < 0 || index >= playbackFrames.length) return;
        const frame = playbackFrames[index];
        renderTelemetryFrame(frame, index, true);
        
        if (timelineSlider) timelineSlider.value = index;
        const totalDuration = playbackFrames[playbackFrames.length - 1].level_time || 0;
        if (timelineTime) {
            timelineTime.textContent = `${formatTime(frame.level_time)} / ${formatTime(totalDuration)}`;
        }
    }

    function startPlaybackTimer() {
        stopPlaybackTimer();
        if (btnPlay) btnPlay.style.display = 'none';
        if (btnPause) btnPause.style.display = 'inline-block';
        
        const baseInterval = 100; // 10Hz base tick rate
        const delay = baseInterval / playbackSpeed;
        
        playbackIntervalId = setInterval(() => {
            currentFrameIndex++;
            if (currentFrameIndex >= playbackFrames.length) {
                stopPlaybackTimer();
                currentFrameIndex = playbackFrames.length - 1;
                if (btnPlay) btnPlay.style.display = 'inline-block';
                if (btnPause) btnPause.style.display = 'none';
            }
            renderFrame(currentFrameIndex);
        }, delay);
    }

    function stopPlaybackTimer() {
        if (playbackIntervalId) {
            clearInterval(playbackIntervalId);
            playbackIntervalId = null;
        }
        if (btnPlay) btnPlay.style.display = 'inline-block';
        if (btnPause) btnPause.style.display = 'none';
    }

    if (btnPlay) {
        btnPlay.addEventListener('click', () => {
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

    if (btnStop) {
        btnStop.addEventListener('click', () => {
            stopPlaybackTimer();
            playbackMode = false;
            playbackFrames = [];
            currentFrameIndex = 0;
            
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
            
            if (playbackControls) playbackControls.style.display = 'none';
            if (dropZone) {
                dropZone.style.borderColor = 'rgba(255, 45, 149, 0.45)';
                dropZone.querySelector('p').textContent = 'Drag & Drop flight recording file here (.jsonl)';
            }
            
            connect();
        });
    }

    if (timelineSlider) {
        timelineSlider.addEventListener('input', (e) => {
            if (playbackIntervalId) {
                wasPlayingBeforeDrag = true;
                stopPlaybackTimer();
            } else {
                wasPlayingBeforeDrag = false;
            }
            currentFrameIndex = parseInt(e.target.value);
            renderFrame(currentFrameIndex);
        });

        timelineSlider.addEventListener('change', () => {
            if (wasPlayingBeforeDrag) {
                startPlaybackTimer();
            }
        });
    }

    if (playbackSpeedSelect) {
        playbackSpeedSelect.addEventListener('change', (e) => {
            playbackSpeed = parseFloat(e.target.value);
            if (playbackIntervalId) {
                startPlaybackTimer();
            }
        });
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
