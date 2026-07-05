/**
 * slam2d.js — Interactive 2D SLAM map viewer (pan + zoom, HD rendering).
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
        this._img = new Image();
        this._imgLoaded = false;
        this._dpr = 1;
        this._cssW = 1;
        this._cssH = 1;

        this._scale = 1.0;
        this._panX = 0;
        this._panY = 0;
        this._minScale = 0.35;
        this._maxScale = 10.0;

        this._dragging = false;
        this._lastX = 0;
        this._lastY = 0;

        this._img.onload = () => {
            this._imgLoaded = true;
            this._draw();
        };

        this._onWheel = this._onWheel.bind(this);
        this._onDown = this._onDown.bind(this);
        this._onMove = this._onMove.bind(this);
        this._onUp = this._onUp.bind(this);
        this._onDblClick = this._onDblClick.bind(this);
        this._onResize = this._onResize.bind(this);

        this.canvas.addEventListener('wheel', this._onWheel, { passive: false });
        this.canvas.addEventListener('mousedown', this._onDown);
        this.canvas.addEventListener('mousemove', this._onMove);
        this.canvas.addEventListener('mouseup', this._onUp);
        this.canvas.addEventListener('mouseleave', this._onUp);
        this.canvas.addEventListener('dblclick', this._onDblClick);

        if (window.ResizeObserver && this.container) {
            new ResizeObserver(() => this._onResize()).observe(this.container);
        }
        window.addEventListener('resize', this._onResize);
        this.canvas.style.cursor = 'grab';
        this._onResize();
    }

    update(b64) {
        if (!this.canvas || !b64) return;
        const mime = b64.startsWith('iVBOR') ? 'image/png' : 'image/jpeg';
        this._img.src = `data:${mime};base64,` + b64;
    }

    resetView() {
        this._scale = 1.0;
        this._panX = 0;
        this._panY = 0;
        this._draw();
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

    _onWheel(e) {
        e.preventDefault();
        const cx = this._cssW / 2;
        const cy = this._cssH / 2;
        const mx = e.clientX - this.canvas.getBoundingClientRect().left - cx - this._panX;
        const my = e.clientY - this.canvas.getBoundingClientRect().top - cy - this._panY;

        const factor = e.deltaY > 0 ? 0.88 : 1.12;
        const newScale = Math.max(this._minScale, Math.min(this._maxScale, this._scale * factor));
        const ratio = newScale / this._scale;

        this._panX -= mx * (ratio - 1);
        this._panY -= my * (ratio - 1);
        this._scale = newScale;
        this._draw();
    }

    _onDown(e) {
        if (e.button !== 0) return;
        this._dragging = true;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }

    _onMove(e) {
        if (!this._dragging) return;
        this._panX += e.clientX - this._lastX;
        this._panY += e.clientY - this._lastY;
        this._lastX = e.clientX;
        this._lastY = e.clientY;
        this._draw();
    }

    _onUp() {
        this._dragging = false;
        if (this.canvas) this.canvas.style.cursor = 'grab';
    }

    _onDblClick() {
        this.resetView();
    }

    _draw() {
        if (!this.ctx || !this.canvas) return;

        const ctx = this.ctx;
        const dpr = this._dpr;
        const w = this._cssW;
        const h = this._cssH;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.fillStyle = '#050810';
        ctx.fillRect(0, 0, w, h);

        if (!this._imgLoaded || !this._img.naturalWidth) return;

        // Fit map to panel at scale=1, then apply user zoom/pan
        const fit = Math.min(w / this._img.naturalWidth, h / this._img.naturalHeight);
        const baseScale = fit * this._scale;

        ctx.imageSmoothingEnabled = this._scale > 1.05;
        ctx.imageSmoothingQuality = 'high';

        ctx.save();
        ctx.translate(w / 2 + this._panX, h / 2 + this._panY);
        ctx.scale(baseScale, baseScale);
        ctx.drawImage(
            this._img,
            -this._img.naturalWidth / 2,
            -this._img.naturalHeight / 2
        );
        ctx.restore();
    }
}
