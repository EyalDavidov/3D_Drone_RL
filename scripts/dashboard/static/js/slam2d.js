/**
 * slam2d.js — Native 2D SLAM occupancy map (top-down radar).
 *
 * Renders directly from the `slam_3d` telemetry grid (no server PNG): occupancy
 * cells are rasterised to an offscreen bitmap that only rebuilds when the map
 * changes (gver), then drawn under crisp neon vector overlays (path, frontiers,
 * person, drone + FOV). Pan / zoom / reset with live world-coordinate readout.
 *
 * Grid cell values: 0 unknown · 1 free · 2 inflated(danger) · 3 wall.
 */
class SlamMap2D {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('[SlamMap2D] canvas not found:', canvasId);
            return;
        }
        this.ctx = this.canvas.getContext('2d');
        this.container = this.canvas.parentElement;

        // Telemetry
        this._d = null;
        this._gridVer = null;
        this._grid = new OffscreenCanvasShim();  // offscreen occupancy bitmap

        // View transform (user zoom/pan on top of the fit)
        this._scale = 1.0;
        this._panX = 0;
        this._panY = 0;
        this._minScale = 0.4;
        this._maxScale = 12.0;

        this._dragging = false;
        this._lastX = 0;
        this._lastY = 0;
        this._mouseWorld = null;

        this._dpr = 1;
        this._cssW = 1;
        this._cssH = 1;
        this._time = 0;

        // Palette — must match live_telemetry grid encoding:
        // 0=unknown 1=free 2=danger 3=dodgeable-obstacle 4=structural-wall
        this.COL = {
            free:     [34, 48, 70, 150],     // explored floor (cool slate)
            inflated: [255, 176, 32, 55],    // amber danger halo (walls only)
            obstacle: [31, 156, 140, 200],   // teal — corridor props (not walls)
            wall:     [255, 204, 51, 240],   // warm yellow — structural walls
        };

        // DOM overlays
        this.coordEl = document.getElementById('slam2d-coord');
        this.zoomEl  = document.getElementById('slam2d-zoom');

        // Bind + wire events
        ['_onWheel', '_onDown', '_onMove', '_onUp', '_onDblClick', '_onResize']
            .forEach(m => { this[m] = this[m].bind(this); });
        this.canvas.addEventListener('wheel', this._onWheel, { passive: false });
        this.canvas.addEventListener('mousedown', this._onDown);
        this.canvas.addEventListener('mousemove', this._onMove);
        this.canvas.addEventListener('mouseup', this._onUp);
        this.canvas.addEventListener('mouseleave', this._onUp);
        this.canvas.addEventListener('dblclick', this._onDblClick);

        if (window.ResizeObserver && this.container) {
            new ResizeObserver(this._onResize).observe(this.container);
        }
        window.addEventListener('resize', this._onResize);
        this.canvas.style.cursor = 'grab';
        this._onResize();

        this._loop = this._loop.bind(this);
        requestAnimationFrame(this._loop);
    }

    // ---- Public API ----
    update(slam3d) {
        if (!slam3d) return;
        this._d = slam3d;
        if (slam3d.grid && slam3d.H && slam3d.W) {
            const ver = (slam3d.gver !== undefined) ? slam3d.gver : slam3d.grid.length;
            if (ver !== this._gridVer) {
                this._gridVer = ver;
                this._rebuildGrid(slam3d);
            }
        }
    }

    resetView() {
        this._scale = 1.0;
        this._panX = 0;
        this._panY = 0;
    }

    _onResize() {
        if (!this.container) return;
        this._dpr = Math.min(window.devicePixelRatio || 1, 2);
        this._cssW = Math.max(1, this.container.clientWidth);
        this._cssH = Math.max(1, this.container.clientHeight);
        this.canvas.width = Math.round(this._cssW * this._dpr);
        this.canvas.height = Math.round(this._cssH * this._dpr);
        this.canvas.style.width = this._cssW + 'px';
        this.canvas.style.height = this._cssH + 'px';
    }

    // ---- Occupancy → offscreen bitmap (rebuilt only on map change) ----
    _rebuildGrid(d) {
        const { grid, H, W } = d;
        const bin = atob(grid);
        const n = bin.length;
        const cv = this._grid.ensure(W, H);
        const ctx = cv.getContext('2d');
        const img = ctx.createImageData(W, H);
        const px = img.data;
        const C = this.COL;

        for (let i = 0; i < n; i++) {
            const v = bin.charCodeAt(i);
            // Flip columns horizontally (mirror on X) so the map's left/right matches
            // the Isaac sim view — a right turn shows as a right turn. Rows (world Y)
            // keep their natural top→bottom order so up/down stays as before.
            const r = (i / W) | 0;
            const c = i - r * W;
            const o = (r * W + (W - 1 - c)) * 4;
            let col = null;
            if (v === 1) col = C.free;
            else if (v === 2) col = C.inflated;
            else if (v === 3) col = C.obstacle;
            else if (v === 4) col = C.wall;
            if (col) {
                px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; px[o + 3] = col[3];
            } else {
                px[o + 3] = 0;   // unknown → transparent
            }
        }
        ctx.putImageData(img, 0, 0);
    }

    // ---- World ↔ screen transform ----
    _fit() {
        const d = this._d;
        const w = this._cssW, h = this._cssH;
        if (!d || d.max_x === undefined) {
            return { ok: false };
        }
        const worldW = Math.max(1e-3, d.max_x - d.min_x);
        const worldH = Math.max(1e-3, d.max_y - d.min_y);
        const pad = 0.92;
        const base = Math.min(w / worldW, h / worldH) * pad;
        const scale = base * this._scale;
        const cxW = (d.min_x + d.max_x) / 2;
        const cyW = (d.min_y + d.max_y) / 2;
        const cx = w / 2 + this._panX;
        const cy = h / 2 + this._panY;
        return { ok: true, scale, cxW, cyW, cx, cy, worldW, worldH };
    }
    _toScreen(t, wx, wy) {
        // Mirror on X (larger world X maps LEFT) so left/right matches the Isaac sim.
        // World Y keeps its Y-down mapping (original up/down orientation preserved).
        return [t.cx - (wx - t.cxW) * t.scale, t.cy + (wy - t.cyW) * t.scale];
    }
    _toWorld(t, sx, sy) {
        return [t.cxW - (sx - t.cx) / t.scale, t.cyW + (sy - t.cy) / t.scale];
    }

    // ---- Render loop ----
    _loop(ts) {
        this._time = ts * 0.001;
        this._draw();
        requestAnimationFrame(this._loop);
    }

    _draw() {
        const ctx = this.ctx;
        if (!ctx) return;
        const w = this._cssW, h = this._cssH, dpr = this._dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#060a12';
        ctx.fillRect(0, 0, w, h);

        const t = this._fit();
        if (!t.ok) {
            ctx.fillStyle = 'rgba(120,150,190,0.5)';
            ctx.font = '13px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Awaiting SLAM map…', w / 2, h / 2);
            return;
        }
        const d = this._d;

        this._drawRadarGrid(ctx, t, d);

        // Occupancy bitmap
        if (this._grid.canvas && this._grid.w) {
            // X mirrored, Y-down → screen top-left corresponds to (max_x, min_y).
            const [x0, y0] = this._toScreen(t, d.max_x, d.min_y);
            const dw = t.worldW * t.scale;
            const dh = t.worldH * t.scale;
            ctx.imageSmoothingEnabled = t.scale < 6;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(this._grid.canvas, x0, y0, dw, dh);
        }

        this._drawPath(ctx, t, d.path);
        this._drawFrontiers(ctx, t, d.frontiers, d.active);
        this._drawActive(ctx, t, d.active);
        this._drawPerson(ctx, t, d.person);
        this._drawDrone(ctx, t, d.drone);

        this._updateHud(d);
    }

    _drawRadarGrid(ctx, t, d) {
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(47,243,255,0.06)';
        const step = 1.0; // 1 metre
        const x0 = Math.ceil(d.min_x / step) * step;
        const y0 = Math.ceil(d.min_y / step) * step;
        ctx.beginPath();
        for (let x = x0; x <= d.max_x; x += step) {
            const [sx, sy1] = this._toScreen(t, x, d.min_y);
            const [, sy2] = this._toScreen(t, x, d.max_y);
            ctx.moveTo(sx, sy1); ctx.lineTo(sx, sy2);
        }
        for (let y = y0; y <= d.max_y; y += step) {
            const [sx1, sy] = this._toScreen(t, d.min_x, y);
            const [sx2] = this._toScreen(t, d.max_x, y);
            ctx.moveTo(sx1, sy); ctx.lineTo(sx2, sy);
        }
        ctx.stroke();
    }

    _drawPath(ctx, t, path) {
        if (!path || path.length < 2) return;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < path.length; i++) {
            const [sx, sy] = this._toScreen(t, path[i][0], path[i][1]);
            i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
        }
        ctx.strokeStyle = 'rgba(60,255,140,0.25)';
        ctx.lineWidth = 6;
        ctx.stroke();
        ctx.strokeStyle = '#3cff8c';
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(60,255,140,0.7)';
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    _drawFrontiers(ctx, t, frontiers, active) {
        if (!frontiers) return;
        for (const f of frontiers) {
            if (active && Math.abs(f[0] - active[0]) < 0.1 && Math.abs(f[1] - active[1]) < 0.1) continue;
            const [sx, sy] = this._toScreen(t, f[0], f[1]);
            ctx.fillStyle = 'rgba(255,176,32,0.9)';
            ctx.shadowColor = 'rgba(255,176,32,0.7)';
            ctx.shadowBlur = 8;
            ctx.beginPath(); ctx.arc(sx, sy, 4.5, 0, Math.PI * 2); ctx.fill();
            ctx.shadowBlur = 0;
            ctx.fillStyle = '#fff';
            ctx.beginPath(); ctx.arc(sx, sy, 1.6, 0, Math.PI * 2); ctx.fill();
        }
    }

    _drawActive(ctx, t, active) {
        if (!active) return;
        const [sx, sy] = this._toScreen(t, active[0], active[1]);
        const pulse = 0.5 + 0.5 * Math.sin(this._time * 4);
        ctx.strokeStyle = '#2ff3ff';
        ctx.lineWidth = 2;
        ctx.shadowColor = 'rgba(47,243,255,0.8)';
        ctx.shadowBlur = 10;
        ctx.beginPath(); ctx.arc(sx, sy, 12 + pulse * 4, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(sx - 16, sy); ctx.lineTo(sx + 16, sy);
        ctx.moveTo(sx, sy - 16); ctx.lineTo(sx, sy + 16);
        ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ff2d95';
        ctx.beginPath(); ctx.arc(sx, sy, 3, 0, Math.PI * 2); ctx.fill();
    }

    _drawPerson(ctx, t, person) {
        if (!person) return;
        const [sx, sy] = this._toScreen(t, person[0], person[1]);
        ctx.save();
        ctx.translate(sx, sy);
        ctx.fillStyle = '#ff2d95';
        ctx.shadowColor = 'rgba(255,45,149,0.8)';
        ctx.shadowBlur = 12;
        // 5-point star
        ctx.beginPath();
        for (let i = 0; i < 10; i++) {
            const r = i % 2 ? 4 : 9;
            const a = -Math.PI / 2 + i * Math.PI / 5;
            const x = Math.cos(a) * r, y = Math.sin(a) * r;
            i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.closePath(); ctx.fill();
        ctx.restore();
    }

    _drawDrone(ctx, t, drone) {
        if (!drone) return;
        const [sx, sy] = this._toScreen(t, drone.x, drone.y);
        const yaw = drone.yaw || 0;
        // X mirrored, Y-down: world heading (cos yaw, sin yaw) → screen (-cos yaw, sin yaw).
        const sdx = -Math.cos(yaw), sdy = Math.sin(yaw);
        const ang = Math.atan2(sdy, sdx);

        // FOV cone
        const fov = 0.7, reach = 62;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.arc(sx, sy, reach, ang - fov, ang + fov);
        ctx.closePath();
        const grad = ctx.createRadialGradient(sx, sy, 4, sx, sy, reach);
        grad.addColorStop(0, 'rgba(60,255,140,0.28)');
        grad.addColorStop(1, 'rgba(60,255,140,0)');
        ctx.fillStyle = grad;
        ctx.fill();

        // Pulsing range ring
        const pulse = 0.5 + 0.5 * Math.sin(this._time * 3);
        ctx.strokeStyle = `rgba(60,255,140,${0.12 + pulse * 0.12})`;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(sx, sy, 16 + pulse * 5, 0, Math.PI * 2); ctx.stroke();

        // Body triangle
        ctx.save();
        ctx.translate(sx, sy);
        ctx.rotate(ang);
        ctx.beginPath();
        ctx.moveTo(11, 0);
        ctx.lineTo(-7, 7);
        ctx.lineTo(-7, -7);
        ctx.closePath();
        ctx.fillStyle = '#3cff8c';
        ctx.shadowColor = 'rgba(60,255,140,0.9)';
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.strokeStyle = '#eafff2';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
    }

    _updateHud(d) {
        if (this.zoomEl) this.zoomEl.textContent = this._scale.toFixed(1) + '×';
        if (this.coordEl) {
            if (this._mouseWorld) {
                this.coordEl.textContent = `${this._mouseWorld[0].toFixed(1)}, ${this._mouseWorld[1].toFixed(1)}`;
            } else if (d && d.drone) {
                this.coordEl.textContent = `${d.drone.x.toFixed(1)}, ${d.drone.y.toFixed(1)}`;
            }
        }
    }

    // ---- Interaction ----
    _onWheel(e) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const factor = e.deltaY > 0 ? 0.88 : 1.14;
        const newScale = Math.max(this._minScale, Math.min(this._maxScale, this._scale * factor));
        const ratio = newScale / this._scale;
        // Zoom toward cursor
        const ox = mx - (this._cssW / 2 + this._panX);
        const oy = my - (this._cssH / 2 + this._panY);
        this._panX -= ox * (ratio - 1);
        this._panY -= oy * (ratio - 1);
        this._scale = newScale;
    }
    _onDown(e) {
        if (e.button !== 0) return;
        this._dragging = true;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }
    _onMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const t = this._fit();
        if (t.ok) this._mouseWorld = this._toWorld(t, e.clientX - rect.left, e.clientY - rect.top);
        if (!this._dragging) return;
        this._panX += e.clientX - this._lastX;
        this._panY += e.clientY - this._lastY;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
    }
    _onUp() {
        this._dragging = false;
        this._mouseWorld = null;
        if (this.canvas) this.canvas.style.cursor = 'grab';
    }
    _onDblClick() { this.resetView(); }
}

/** Tiny helper: reuse one offscreen canvas for the occupancy bitmap. */
class OffscreenCanvasShim {
    constructor() { this.canvas = null; this.w = 0; this.h = 0; }
    ensure(w, h) {
        if (!this.canvas) this.canvas = document.createElement('canvas');
        if (this.w !== w || this.h !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
            this.w = w; this.h = h;
        }
        return this.canvas;
    }
}
