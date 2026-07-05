/**
 * cameras.js — Camera feed renderer
 * Renders Base64 PNG images to canvas elements for all 7 camera feeds.
 */

class CameraFeeds {
    constructor() {
        // Primary feed map: data key → canvas element ID
        this._feedMap = {
            rgb_first_person: 'cam-fp',
            rgb_third_1:      'cam-t1',
            rgb_third_2:      'cam-t2',
            rgb_third_3:      'cam-t3',
            depth:            'cam-depth',
            depth_saliency:   'cam-saliency',
            ae_recon:         'cam-ae',
            // slam_map rendered by SlamMap2D (interactive pan/zoom)
        };

        this._canvases = {};
        this._contexts = {};
        this._images   = {};

        for (const [key, canvasId] of Object.entries(this._feedMap)) {
            const canvas = document.getElementById(canvasId);
            if (canvas) {
                this._canvases[key] = canvas;
                this._contexts[key] = canvas.getContext('2d');
                const img = new Image();
                img.onload = () => this._draw(key);
                this._images[key] = img;
            }
        }

        // Extra canvases that mirror a feed (none on Navigation tab)
        this._mirrors = [];
        this._mirrorCtx = {};
        for (const m of this._mirrors) {
            const canvas = document.getElementById(m.canvasId);
            if (canvas) {
                this._mirrorCtx[m.canvasId] = { ctx: canvas.getContext('2d'), canvas };
            }
        }
        this._mirrorImgs = {};
        for (const m of this._mirrors) {
            const img = new Image();
            img.onload = () => {
                const mc = this._mirrorCtx[m.canvasId];
                if (mc) {
                    const { ctx, canvas } = mc;
                    ctx.imageSmoothingEnabled = true;
                    ctx.imageSmoothingQuality = 'high';
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                }
            };
            this._mirrorImgs[m.canvasId] = { img, sourceKey: m.sourceKey };
        }
    }

    update(images) {
        if (!images) return;

        // Primary feeds
        for (const [key] of Object.entries(this._feedMap)) {
            if (images[key] && this._images[key]) {
                // Images from the server can be JPEG or PNG; both work with data URI
                this._images[key].src = 'data:image/jpeg;base64,' + images[key];
            }
        }

        // Mirror feeds
        for (const [canvasId, entry] of Object.entries(this._mirrorImgs)) {
            const src = images[entry.sourceKey];
            if (src) {
                entry.img.src = 'data:image/jpeg;base64,' + src;
            }
        }
    }

    _draw(key) {
        const ctx    = this._contexts[key];
        const canvas = this._canvases[key];
        const img    = this._images[key];
        if (!ctx || !canvas || !img) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
}
