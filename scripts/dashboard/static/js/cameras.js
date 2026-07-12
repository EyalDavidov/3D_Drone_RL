/**
 * cameras.js — Camera feed renderer
 * Industry-style letterbox: blurred ambient backdrop + sharp contain center.
 */

class CameraFeeds {
    constructor() {
        this._feedMap = {
            rgb_first_person: 'cam-fp',
            rgb_third_1:      'cam-t1',
            rgb_third_2:      'cam-t2',
            rgb_third_3:      'cam-t3',
            depth:            'cam-depth',
            depth_saliency:   'cam-saliency',
            ae_recon:         'cam-ae',
        };

        this._canvases = {};
        this._contexts = {};
        this._images = {};

        for (const [key, canvasId] of Object.entries(this._feedMap)) {
            const canvas = document.getElementById(canvasId);
            if (canvas) {
                this._canvases[key] = canvas;
                this._contexts[key] = canvas.getContext('2d');
                const img = new Image();
                img.onload = () => {
                    this._resizeCanvas(key);
                    this._draw(key);
                };
                this._images[key] = img;
            }
        }

        this._mirrors = [];
        this._mirrorCtx = {};
        this._mirrorImgs = {};

        this._soloCam = null;
        this._wall = document.getElementById('cam-wall');
        this._soloBar = document.getElementById('cam-solo-bar');
        this._bindSoloControls();

        this._onResize = this._onResize.bind(this);
        window.addEventListener('resize', this._onResize);
        if (typeof ResizeObserver !== 'undefined') {
            this._resizeObserver = new ResizeObserver(() => this._onResize());
            const tab = document.getElementById('tab-cameras');
            if (tab) this._resizeObserver.observe(tab);
        }
        requestAnimationFrame(() => {
            this._resetGridView();
            this._onResize();
        });
    }

    _resetGridView() {
        this.setSolo(null);
        document.querySelectorAll('.cam-wall-cell.minimized').forEach(el => {
            el.classList.remove('minimized');
            const body = el.querySelector('.panel-body');
            if (body) body.style.display = '';
        });
    }

    resetToGrid() {
        this._resetGridView();
        this._onResize();
    }

    _bindSoloControls() {
        document.querySelectorAll('.cam-solo-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const cell = btn.closest('[data-cam-id]');
                const id = cell && cell.dataset.camId;
                if (!id) return;
                if (this._soloCam === id) this.setSolo(null);
                else this.setSolo(id);
            });
        });
        const exitBtn = document.getElementById('cam-solo-exit');
        if (exitBtn) exitBtn.addEventListener('click', () => this.setSolo(null));
        document.querySelectorAll('.cam-solo-pick').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = btn.dataset.camId;
                if (id) this.setSolo(id);
            });
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this._soloCam) this.setSolo(null);
        });
    }

    setSolo(camId) {
        const next = camId || null;
        this._soloCam = next;

        if (this._wall) {
            this._wall.classList.toggle('cam-wall-solo', !!next);
        }
        if (this._soloBar) {
            this._soloBar.hidden = !next;
        }

        document.querySelectorAll('.cam-wall-cell[data-cam-id]').forEach(cell => {
            const id = cell.dataset.camId;
            const isActive = next && id === next;
            cell.classList.toggle('cam-solo-active', isActive);
            cell.classList.toggle('cam-solo-hidden', next && id !== next);
        });

        document.querySelectorAll('.cam-solo-btn').forEach(btn => {
            const cell = btn.closest('[data-cam-id]');
            const on = cell && cell.dataset.camId === next;
            btn.classList.toggle('active', on);
            btn.title = on ? 'Back to 4-up (Esc)' : 'Solo view';
        });

        document.querySelectorAll('.cam-solo-pick').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.camId === next);
        });

        requestAnimationFrame(() => this._onResize());
    }

    _onResize() {
        for (const key of Object.keys(this._canvases)) {
            this._resizeCanvas(key);
            this._draw(key);
        }
    }

    _resizeCanvas(key) {
        const canvas = this._canvases[key];
        if (!canvas || !canvas.parentElement) return;
        const rect = canvas.parentElement.getBoundingClientRect();
        if (rect.width < 2 || rect.height < 2) return;
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.floor(rect.width * dpr));
        canvas.height = Math.max(1, Math.floor(rect.height * dpr));
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        const ctx = this._contexts[key];
        if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    update(images) {
        if (!images) return;
        for (const [key] of Object.entries(this._feedMap)) {
            if (images[key] && this._images[key]) {
                if (images[key].startsWith('/recordings') || images[key].startsWith('http')) {
                    this._images[key].src = images[key];
                } else {
                    this._images[key].src = 'data:image/jpeg;base64,' + images[key];
                }
            }
        }
    }

    _containRect(w, h, nw, nh) {
        const scale = Math.min(w / nw, h / nh);
        const dw = nw * scale;
        const dh = nh * scale;
        return { dx: (w - dw) / 2, dy: (h - dh) / 2, dw, dh };
    }

    _drawLetterboxVignette(ctx, w, h, dx, dy, dw, dh) {
        if (dx > 2) {
            const g = ctx.createLinearGradient(0, 0, dx, 0);
            g.addColorStop(0, 'rgba(0,0,0,0.55)');
            g.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = g;
            ctx.fillRect(0, 0, dx, h);
        }
        const right = dx + dw;
        if (right < w - 2) {
            const g = ctx.createLinearGradient(right, 0, w, 0);
            g.addColorStop(0, 'rgba(0,0,0,0)');
            g.addColorStop(1, 'rgba(0,0,0,0.55)');
            ctx.fillStyle = g;
            ctx.fillRect(right, 0, w - right, h);
        }
        if (dy > 2) {
            const g = ctx.createLinearGradient(0, 0, 0, dy);
            g.addColorStop(0, 'rgba(0,0,0,0.45)');
            g.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = g;
            ctx.fillRect(dx, 0, dw, dy);
        }
        const bottom = dy + dh;
        if (bottom < h - 2) {
            const g = ctx.createLinearGradient(0, bottom, 0, h);
            g.addColorStop(0, 'rgba(0,0,0,0)');
            g.addColorStop(1, 'rgba(0,0,0,0.45)');
            ctx.fillStyle = g;
            ctx.fillRect(dx, bottom, dw, h - bottom);
        }
    }

    _draw(key) {
        const ctx = this._contexts[key];
        const canvas = this._canvases[key];
        const img = this._images[key];
        if (!ctx || !canvas || !img || !img.naturalWidth) return;

        const dpr = window.devicePixelRatio || 1;
        const w = canvas.width / dpr;
        const h = canvas.height / dpr;
        const nw = img.naturalWidth;
        const nh = img.naturalHeight;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = '#060a10';
        ctx.fillRect(0, 0, w, h);

        // 1) Ambient blur fill (broadcast / Netflix / FaceTime style)
        ctx.save();
        ctx.filter = 'blur(32px) brightness(0.5) saturate(1.2)';
        const cover = Math.max(w / nw, h / nh);
        const bw = nw * cover;
        const bh = nh * cover;
        ctx.drawImage(img, (w - bw) / 2, (h - bh) / 2, bw, bh);
        ctx.restore();

        // 2) Sharp main feed — contain (full FOV, drone visible)
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        const { dx, dy, dw, dh } = this._containRect(w, h, nw, nh);
        ctx.drawImage(img, dx, dy, dw, dh);

        // 3) Soft vignette on gutter zones
        this._drawLetterboxVignette(ctx, w, h, dx, dy, dw, dh);

        // 4) Video frame bezel (GCS scope border)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
        ctx.lineWidth = 1;
        ctx.strokeRect(dx + 0.5, dy + 0.5, dw - 1, dh - 1);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.strokeRect(dx + 1.5, dy + 1.5, dw - 3, dh - 3);
    }
}
