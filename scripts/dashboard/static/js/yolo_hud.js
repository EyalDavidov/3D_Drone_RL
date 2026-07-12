/**
 * yolo_hud.js — Native YOLO detection HUD.
 *
 * Fully renders the detection HUD in the browser (no OpenCV bitmap): a clean
 * camera frame is drawn to a canvas and bounding boxes + overlays are painted
 * as crisp vectors from structured telemetry. Frame and boxes come from the
 * same YOLO pass server-side, so the overlay is perfectly in sync.
 */
class YoloHud {
    constructor(rootId) {
        this.root = document.getElementById(rootId);
        this.canvas = document.getElementById('yolo-canvas');
        this.canvasLeft = document.getElementById('yolo-canvas-left');
        this.canvasRight = document.getElementById('yolo-canvas-right');
        if (!this.root || !this.canvas || !this.canvasLeft || !this.canvasRight) {
            console.warn('[YoloHud] root/canvas(es) not found');
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        this.ctxLeft = this.canvasLeft.getContext('2d');
        this.ctxRight = this.canvasRight.getContext('2d');

        this.stage = this.canvas.parentElement;
        this.stageLeft = this.canvasLeft.parentElement;
        this.stageRight = this.canvasRight.parentElement;

        // Backdrop images (clean camera frames)
        this._img = new Image();
        this._imgReady = false;
        this._img.onload = () => { this._imgReady = true; };

        this._imgLeft = new Image();
        this._imgLeftReady = false;
        this._imgLeft.onload = () => { this._imgLeftReady = true; };

        this._imgRight = new Image();
        this._imgRightReady = false;
        this._imgRight.onload = () => { this._imgRightReady = true; };

        // Latest telemetry
        this._stats = null;
        this._boxes = [];
        this._boxesLeft = [];
        this._boxesRight = [];
        this._state = 'idle';
        this._threshold = 0.7;

        // Smoothed / animated values
        this._dispBoxes = [];       // eased box geometry for smooth motion
        this._time = 0;

        this._dpr = 1;
        this._cssW = 1;
        this._cssH = 1;
        this._cssWLeft = 1;
        this._cssHLeft = 1;
        this._cssWRight = 1;
        this._cssHRight = 1;

        // Cached DOM
        this.el = {
            statusPill:  document.getElementById('yolo-status-pill'),
            statusVal:   document.getElementById('yolo-status-val'),
            confBig:     document.getElementById('yolo-conf-big'),
            footMeter:   document.getElementById('yolo-foot-meter'),
            footThresh:  document.getElementById('yolo-foot-thresh'),
            threshCap:   document.getElementById('yolo-thresh-cap'),
            frameVal:    document.getElementById('yolo-frame-val'),
            scanChip:    document.getElementById('yolo-scan-chip'),
            scanLabel:   document.getElementById('yolo-scan-label'),
            intel:       document.getElementById('yolo-intel'),
            intelLabel:  document.getElementById('yolo-intel-label'),
            intelConf:   document.getElementById('yolo-intel-conf'),
            intelLat:    document.getElementById('yolo-intel-lat'),
            intelLon:    document.getElementById('yolo-intel-lon'),
            intelDist:   document.getElementById('yolo-intel-dist'),
            // sidebar
            confVal:     document.getElementById('yolo-conf-val'),
            liveVal:     document.getElementById('yolo-live-val'),
            threshVal:   document.getElementById('yolo-thresh-val'),
            boxesVal:    document.getElementById('yolo-boxes-val'),
            personVal:   document.getElementById('yolo-person-val'),
            log:         document.getElementById('yolo-log'),
            logEmpty:    document.getElementById('yolo-log-empty'),
            logCount:    document.getElementById('yolo-log-count'),
            // alert overlay
            alert:       document.getElementById('yolo-hud-alert'),
            alertIcon:   document.getElementById('yolo-alert-icon'),
            alertLabel:  document.getElementById('yolo-alert-label'),
            alertConf:   document.getElementById('yolo-alert-conf'),
            alertMeter:  document.getElementById('yolo-alert-meter'),
            alertThresh: document.getElementById('yolo-alert-thresh'),
        };

        this._logSig = '';   // signature to avoid rebuilding log DOM every frame

        this.TIER = {
            confirmed: { stroke: '#b6ff3c', glow: 'rgba(182,255,60,0.55)', text: '#e8ffc8' },
            noted:     { stroke: '#ffb020', glow: 'rgba(255,176,32,0.5)',  text: '#ffe6bd' },
            low:       { stroke: '#2ff3ff', glow: 'rgba(47,243,255,0.45)', text: '#d5f8ff' },
        };

        this._onResize = this._onResize.bind(this);
        if (window.ResizeObserver && this.stage) {
            new ResizeObserver(this._onResize).observe(this.stage);
        }
        window.addEventListener('resize', this._onResize);
        this._onResize();

        this._loop = this._loop.bind(this);
        requestAnimationFrame(this._loop);
    }

    // ---- Public API ----
    updateFrame(b64) {
        if (!b64) return;
        const mime = b64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg';
        this._img.src = `data:${mime};base64,` + b64;
    }

    updateFrames(frontB64, leftB64, rightB64) {
        if (frontB64) {
            if (frontB64.startsWith('/recordings') || frontB64.startsWith('http')) {
                this._img.src = frontB64;
            } else {
                const mime = frontB64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg';
                this._img.src = `data:${mime};base64,` + frontB64;
            }
        }
        if (leftB64) {
            if (leftB64.startsWith('/recordings') || leftB64.startsWith('http')) {
                this._imgLeft.src = leftB64;
            } else {
                const mime = leftB64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg';
                this._imgLeft.src = `data:${mime};base64,` + leftB64;
            }
        }
        if (rightB64) {
            if (rightB64.startsWith('/recordings') || rightB64.startsWith('http')) {
                this._imgRight.src = rightB64;
            } else {
                const mime = rightB64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg';
                this._imgRight.src = `data:${mime};base64,` + rightB64;
            }
        }
    }

    update(stats) {
        if (!stats) return;
        this._stats = stats;
        this._boxes = Array.isArray(stats.boxes) ? stats.boxes : [];
        this._boxesLeft = Array.isArray(stats.boxes_left) ? stats.boxes_left : [];
        this._boxesRight = Array.isArray(stats.boxes_right) ? stats.boxes_right : [];
        this._threshold = (typeof stats.conf_threshold === 'number') ? stats.conf_threshold : 0.7;

        const best = (typeof stats.best_conf === 'number') ? stats.best_conf : 0;
        const live = (typeof stats.current_conf === 'number') ? stats.current_conf : best;
        const bestPct = best * 100;
        const state = stats.status || this._deriveState(best, this._threshold, stats.person_found);
        const label = stats.status_label || this._stateLabel(state);
        this._state = state;

        if (this.root) this.root.dataset.state = state;

        // Status pill
        if (this.el.statusPill) this.el.statusPill.dataset.state = state;
        if (this.el.statusVal) this.el.statusVal.textContent = label;

        // Footer readout
        if (this.el.confBig) {
            this.el.confBig.innerHTML = bestPct.toFixed(0) + '<span class="yhud-readout-pct">%</span>';
        }
        if (this.el.footMeter) this.el.footMeter.style.width = this._clampPct(bestPct) + '%';
        if (this.el.footThresh) this.el.footThresh.style.left = (this._threshold * 100).toFixed(1) + '%';
        if (this.el.threshCap) this.el.threshCap.textContent = 'THRESHOLD ' + (this._threshold * 100).toFixed(0) + '%';
        if (this.el.frameVal) this.el.frameVal.textContent = '#' + (stats.detection_count || 0);

        // Scan chip
        if (this.el.scanChip) {
            if (stats.scan_label) {
                this.el.scanChip.hidden = false;
                if (this.el.scanLabel) this.el.scanLabel.textContent = String(stats.scan_label).toUpperCase();
            } else {
                this.el.scanChip.hidden = true;
            }
        }

        // Intel card
        this._updateIntel(stats.intel);

        // Sidebar telemetry
        if (this.el.confVal)  this.el.confVal.textContent = bestPct.toFixed(1) + '%';
        if (this.el.liveVal)  this.el.liveVal.textContent = (live * 100).toFixed(1) + '%';
        if (this.el.threshVal) this.el.threshVal.textContent = (this._threshold * 100).toFixed(0) + '%';
        if (this.el.boxesVal) this.el.boxesVal.textContent = this._boxes.length + this._boxesLeft.length + this._boxesRight.length;
        const personActive = stats.person_found || stats.person_seen
            || state === 'detected' || state === 'confirmed' || state === 'seen';
        if (this.el.personVal) {
            this.el.personVal.textContent = personActive ? 'DETECTED' : 'Not found';
            this.el.personVal.dataset.state = personActive ? 'confirmed' : 'idle';
        }

        // Alert overlay
        this._updateAlert(state, label, bestPct);

        // Rescue log
        this._updateLog(stats.rescue_log || []);

        // Captured frames gallery
        this._updateGallery(stats.captured_frames || []);
    }

    // ---- Alert overlay ----
    _updateAlert(state, label, bestPct) {
        const a = this.el;
        if (a.alert) a.alert.dataset.state = state;
        if (a.alertLabel) a.alertLabel.textContent = state === 'idle' ? 'SCANNING FOR HUMANS' : label;
        if (a.alertConf) a.alertConf.innerHTML = bestPct.toFixed(1) + '<span class="hud-alert-pct">%</span>';
        if (a.alertIcon) a.alertIcon.textContent = state === 'idle' ? '◎' : (state === 'confirmed' ? '✓' : '⚠');
        if (a.alertMeter) a.alertMeter.style.width = this._clampPct(bestPct) + '%';
        if (a.alertThresh) a.alertThresh.style.left = (this._threshold * 100).toFixed(1) + '%';
    }

    // ---- Intel card ----
    _updateIntel(intel) {
        const e = this.el;
        if (!e.intel) return;
        if (!intel) { e.intel.hidden = true; return; }
        const norm = YoloHud._normalizeLogEntry({ label: intel.label, key: 'person_detected' });
        e.intel.hidden = false;
        if (e.intelLabel) e.intelLabel.textContent = norm.label || 'TARGET';
        if (e.intelConf)  e.intelConf.textContent = Math.round((intel.conf || 0) * 100) + '%';
        if (e.intelLat)   e.intelLat.textContent = (intel.gps_lat != null) ? Number(intel.gps_lat).toFixed(6) : '—';
        if (e.intelLon)   e.intelLon.textContent = (intel.gps_lon != null) ? Number(intel.gps_lon).toFixed(6) : '—';
        if (e.intelDist)  e.intelDist.textContent = (intel.dist != null) ? Number(intel.dist).toFixed(1) + ' m' : '—';
    }

    // ---- Rescue log ----
    static _normalizeLogEntry(entry) {
        const lab = String(entry?.label || '').trim().toUpperCase().replace(/_/g, ' ');
        const pk = String(entry?.key || '').trim().toLowerCase().replace(/\s+/g, '_');
        if (lab === 'CAMERA VIEW' && (pk === 'camera_view' || pk === 'cameraview' || pk === '')) {
            return { ...entry, label: 'PERSON DETECTED', key: 'person_detected' };
        }
        return entry;
    }

    _updateLog(log) {
        const host = this.el.log;
        if (!host) return;
        const filtered = (log || []).map(e => YoloHud._normalizeLogEntry(e));
        const logCount = (typeof this._stats?.rescue_log_count === 'number')
            ? this._stats.rescue_log_count
            : filtered.length;
        if (this.el.logCount) this.el.logCount.textContent = logCount;

        const sig = filtered.map(e => `${e.key}:${e.conf}`).join('|');
        if (sig === this._logSig) return;   // no change → skip DOM churn
        this._logSig = sig;

        // Clear existing cards (keep empty-state node)
        host.querySelectorAll('.yhud-log-card').forEach(n => n.remove());
        if (this.el.logEmpty) this.el.logEmpty.style.display = filtered.length ? 'none' : '';

        for (const entry of filtered) {
            const conf = entry.conf || 0;
            const tier = conf >= this._threshold ? 'confirmed'
                       : conf >= (this._threshold * 0.7) ? 'noted' : 'low';
            const col = this.TIER[tier];

            const card = document.createElement('div');
            card.className = 'yhud-log-card';
            card.dataset.tier = tier;
            card.style.setProperty('--accent', col.stroke);
            card.style.setProperty('--accent-glow', col.glow);

            const hasGps = entry.gps_lat != null && entry.gps_lon != null;
            const xyz = Array.isArray(entry.xyz) ? entry.xyz : null;
            const hasXyz = xyz && xyz.length >= 3;
            let coordsHtml = '';
            if (hasGps) {
                coordsHtml = `<div class="yhud-log-gps">
                        <span>LAT <b class="mono">${Number(entry.gps_lat).toFixed(6)}</b></span>
                        <span>LONG <b class="mono">${Number(entry.gps_lon).toFixed(6)}</b></span>
                    </div>`;
            } else if (hasXyz) {
                coordsHtml = `<div class="yhud-log-gps">
                        <span>X <b class="mono">${Number(xyz[0]).toFixed(1)}</b></span>
                        <span>Y <b class="mono">${Number(xyz[1]).toFixed(1)}</b></span>
                        <span>Z <b class="mono">${Number(xyz[2]).toFixed(1)}</b></span>
                    </div>`;
            }
            card.innerHTML = `
                <div class="yhud-log-accent"></div>
                <div class="yhud-log-main">
                    <div class="yhud-log-row1">
                        <span class="yhud-log-label">${this._esc(entry.label || 'CONTACT')}</span>
                        <span class="yhud-log-badge">${Math.round(conf * 100)}%</span>
                    </div>
                    <div class="yhud-log-bar"><div class="yhud-log-bar-fill" style="width:${this._clampPct(conf * 100)}%"></div></div>
                    ${coordsHtml}
                </div>`;
            host.appendChild(card);
        }
    }

    // ---- Captured frames gallery ----
    _updateGallery(frames) {
        const host = document.getElementById('yolo-gallery');
        const emptyNode = document.getElementById('yolo-gallery-empty');
        if (!host) return;

        const sig = (frames || []).join('|');
        if (sig === this._gallerySig) return;
        this._gallerySig = sig;

        // Clear all thumbnails except the empty label
        host.querySelectorAll('.yhud-gallery-thumb').forEach(n => n.remove());

        if (!frames || frames.length === 0) {
            if (emptyNode) emptyNode.style.display = 'block';
            return;
        }

        if (emptyNode) emptyNode.style.display = 'none';

        for (const frame of frames) {
            const isConfirmed = frame.startsWith('detection');
            const label = isConfirmed ? 'CONFIRMED' : 'NOTED';
            const camMatch = frame.match(/_(front|left|right)\.jpg$/i);
            const camLabel = camMatch ? camMatch[1].toUpperCase() : '';
            
            const thumb = document.createElement('div');
            thumb.className = 'yhud-gallery-thumb';
            thumb.style.cssText = `
                position: relative;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 6px;
                overflow: hidden;
                background: rgba(0, 0, 0, 0.3);
                aspect-ratio: 16/9;
                transition: transform 0.2s ease, border-color 0.2s ease;
                cursor: pointer;
            `;
            thumb.onmouseenter = () => {
                thumb.style.transform = 'scale(1.03)';
                thumb.style.borderColor = isConfirmed ? 'rgba(182, 255, 60, 0.6)' : 'rgba(255, 176, 32, 0.6)';
            };
            thumb.onmouseleave = () => {
                thumb.style.transform = 'scale(1)';
                thumb.style.borderColor = 'rgba(255, 255, 255, 0.15)';
            };
            
            const imgPath = `yolo_saves/${frame}`;
            
            thumb.innerHTML = `
                <img src="${imgPath}" style="width: 100%; height: 100%; object-fit: cover;" />
                <div style="
                    position: absolute;
                    bottom: 0; left: 0; right: 0;
                    background: rgba(0, 0, 0, 0.7);
                    color: ${isConfirmed ? '#b6ff3c' : '#ffb020'};
                    font-family: monospace;
                    font-size: 0.65rem;
                    padding: 2px 4px;
                    display: flex;
                    justify-content: space-between;
                ">
                    <span>${label}${camLabel ? ` · ${camLabel}` : ''}</span>
                    <span>#${frame.split('_')[1]?.split('.')[0] || ''}</span>
                </div>
            `;
            
            thumb.onclick = () => {
                window.open(imgPath, '_blank');
            };
            
            host.appendChild(thumb);
        }
    }

    // ---- Canvas render loop ----
    _loop(t) {
        this._time = t * 0.001;
        this._draw();
        requestAnimationFrame(this._loop);
    }

    _draw() {
        this._drawSingle(this.ctxLeft, this._imgLeft, this._imgLeftReady, this._boxesLeft, this._cssWLeft, this._cssHLeft);
        this._drawSingle(this.ctx, this._img, this._imgReady, this._boxes, this._cssW, this._cssH);
        this._drawSingle(this.ctxRight, this._imgRight, this._imgRightReady, this._boxesRight, this._cssWRight, this._cssHRight);
    }

    _drawSingle(ctx, img, imgReady, boxes, cssW, cssH) {
        if (!ctx) return;
        const w = cssW, h = cssH, dpr = this._dpr;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#070b14';
        ctx.fillRect(0, 0, w, h);

        // Fit camera frame (contain)
        let ox = 0, oy = 0, dw = w, dh = h;
        if (imgReady && img.naturalWidth) {
            const iw = img.naturalWidth, ih = img.naturalHeight;
            const fit = Math.min(w / iw, h / ih);
            dw = iw * fit; dh = ih * fit;
            ox = (w - dw) / 2; oy = (h - dh) / 2;
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(img, ox, oy, dw, dh);
        } else {
            ctx.fillStyle = 'rgba(120,150,190,0.5)';
            ctx.font = '13px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for camera feed…', w / 2, h / 2);
        }

        this._drawBoxesSingle(ctx, boxes, ox, oy, dw, dh);
    }

    _drawBoxesSingle(ctx, boxes, ox, oy, dw, dh) {
        if (!boxes || !boxes.length) return;

        const pulse = 0.5 + 0.5 * Math.sin(this._time * 4.0);

        boxes.forEach((b, i) => {
            const col = this.TIER[b.tier] || this.TIER.low;
            const x = ox + b.x * dw;
            const y = oy + b.y * dh;
            const bw = b.w * dw;
            const bh = b.h * dh;
            const isBest = i === 0;

            // Soft translucent fill for the primary target
            if (isBest) {
                ctx.fillStyle = col.glow.replace(/[\d.]+\)$/, '0.10)');
                ctx.fillRect(x, y, bw, bh);
            }

            // Thin body rect
            ctx.lineWidth = 1;
            ctx.strokeStyle = this._alpha(col.stroke, 0.35);
            ctx.strokeRect(x, y, bw, bh);

            // Corner brackets
            const arm = Math.max(8, Math.min(bw, bh) * 0.22);
            ctx.lineWidth = isBest ? 2.5 : 2;
            ctx.strokeStyle = col.stroke;
            ctx.shadowColor = col.glow;
            ctx.shadowBlur = isBest ? 12 + pulse * 10 : 6;
            this._corner(ctx, x, y, arm, arm);
            this._corner(ctx, x + bw, y, -arm, arm);
            this._corner(ctx, x, y + bh, arm, -arm);
            this._corner(ctx, x + bw, y + bh, -arm, -arm);
            ctx.shadowBlur = 0;

            // Label chip
            const label = `PERSON ${Math.round(b.conf * 100)}%`;
            ctx.font = '600 11px "JetBrains Mono", monospace';
            const tw = ctx.measureText(label).width;
            const chipH = 17, chipW = tw + 14;
            let cx = x, cy = y - chipH - 4;
            if (cy < oy + 2) cy = y + 4;             // flip inside if clipped at top
            ctx.fillStyle = 'rgba(8,12,20,0.82)';
            this._roundRect(ctx, cx, cy, chipW, chipH, 4);
            ctx.fill();
            ctx.fillStyle = col.text;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.fillText(label, cx + 7, cy + chipH / 2 + 0.5);

            // Best target: crosshair at center
            if (isBest) {
                const mx = x + bw / 2, my = y + bh / 2;
                ctx.strokeStyle = this._alpha(col.stroke, 0.8);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(mx - 9, my); ctx.lineTo(mx + 9, my);
                ctx.moveTo(mx, my - 9); ctx.lineTo(mx, my + 9);
                ctx.stroke();
            }
        });
    }

    _corner(ctx, x, y, dx, dy) {
        ctx.beginPath();
        ctx.moveTo(x + dx, y);
        ctx.lineTo(x, y);
        ctx.lineTo(x, y + dy);
        ctx.stroke();
    }

    _roundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }

    // ---- helpers ----
    _onResize() {
        if (!this.stage || !this.stageLeft || !this.stageRight) return;
        this._dpr = Math.min(window.devicePixelRatio || 1, 2);

        const resizeCanvas = (canvas, stage) => {
            const cssW = Math.max(1, stage.clientWidth);
            const cssH = Math.max(1, stage.clientHeight);
            canvas.width = Math.round(cssW * this._dpr);
            canvas.height = Math.round(cssH * this._dpr);
            canvas.style.width = cssW + 'px';
            canvas.style.height = cssH + 'px';
            return { w: cssW, h: cssH };
        };

        const sizeF = resizeCanvas(this.canvas, this.stage);
        this._cssW = sizeF.w;
        this._cssH = sizeF.h;

        const sizeL = resizeCanvas(this.canvasLeft, this.stageLeft);
        this._cssWLeft = sizeL.w;
        this._cssHLeft = sizeL.h;

        const sizeR = resizeCanvas(this.canvasRight, this.stageRight);
        this._cssWRight = sizeR.w;
        this._cssHRight = sizeR.h;
    }

    _deriveState(best, thresh, personFound) {
        if (personFound) return 'confirmed';
        if (best >= thresh) return 'detected';
        if (best > 0) return 'seen';
        return 'idle';
    }
    _stateLabel(s) {
        return s === 'confirmed' ? 'TARGET CONFIRMED'
             : s === 'detected'  ? 'HUMAN DETECTED'
             : s === 'seen'      ? 'CONTACT · TRACKING' : 'SCANNING';
    }
    _clampPct(p) { return Math.max(0, Math.min(100, p)).toFixed(1); }
    _alpha(hex, a) {
        const n = parseInt(hex.slice(1), 16);
        return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
    }
    _esc(s) { return String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
}
