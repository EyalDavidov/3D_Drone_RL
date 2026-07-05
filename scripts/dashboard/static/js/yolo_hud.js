/**
 * yolo_hud.js — Live mirror of the OpenCV "Brain Nav - YOLO" HUD window.
 * Fit-to-panel rendering with devicePixelRatio for sharp HD display.
 */
class YoloHudView {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('[YoloHudView] canvas not found:', canvasId);
            return;
        }

        this.ctx = this.canvas.getContext('2d');
        this.container = this.canvas.parentElement;
        this._img = new Image();
        this._imgLoaded = false;
        this._dpr = 1;
        this._cssW = 1;
        this._cssH = 1;

        this._img.onload = () => {
            this._imgLoaded = true;
            this._draw();
        };

        this._onResize = this._onResize.bind(this);

        if (window.ResizeObserver && this.container) {
            new ResizeObserver(() => this._onResize()).observe(this.container);
        }
        window.addEventListener('resize', this._onResize);
        this._onResize();
    }

    update(b64) {
        if (!this.canvas || !b64) return;
        const mime = b64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg';
        this._img.src = `data:${mime};base64,` + b64;
    }

    _onResize() {
        if (!this.container) return;
        this._dpr = Math.min(window.devicePixelRatio || 1, 2);
        this._cssW = Math.max(1, this.container.clientWidth);
        this._cssH = Math.max(1, this.container.clientHeight);
        this.canvas.width = Math.round(this._cssW * this._dpr);
        this.canvas.height = Math.round(this._cssH * this._dpr);
        this.canvas.style.width = `${this._cssW}px`;
        this.canvas.style.height = `${this._cssH}px`;
        this._draw();
    }

    _draw() {
        if (!this.ctx || !this.canvas) return;

        const ctx = this.ctx;
        const dpr = this._dpr;
        const w = this._cssW;
        const h = this._cssH;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#141418';
        ctx.fillRect(0, 0, w, h);

        if (!this._imgLoaded || !this._img.naturalWidth) {
            ctx.fillStyle = '#64748b';
            ctx.font = '13px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for YOLO HUD…', w / 2, h / 2);
            return;
        }

        const fit = Math.min(w / this._img.naturalWidth, h / this._img.naturalHeight);
        const drawW = this._img.naturalWidth * fit;
        const drawH = this._img.naturalHeight * fit;
        const ox = (w - drawW) / 2;
        const oy = (h - drawH) / 2;

        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(this._img, ox, oy, drawW, drawH);
    }
}
