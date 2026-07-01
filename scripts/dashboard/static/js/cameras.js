/**
 * cameras.js — Camera feed renderer
 * Renders Base64 PNG images to canvas elements for all 7 camera feeds.
 */

class CameraFeeds {
    constructor() {
        // Map of feed key → canvas ID
        this._feedMap = {
            rgb_first_person: 'cam-fp',
            rgb_third_1: 'cam-t1',
            rgb_third_2: 'cam-t2',
            rgb_third_3: 'cam-t3',
            depth: 'cam-depth',
            depth_saliency: 'cam-saliency',
            ae_recon: 'cam-ae',
        };

        // Also render third_1 in the Navigation tab's main camera
        this._navMainCanvas = document.getElementById('cam-nav-main');
        this._navMainCtx = this._navMainCanvas ? this._navMainCanvas.getContext('2d') : null;

        this._canvases = {};
        this._contexts = {};
        this._images = {};

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

        // Separate image for nav-main (reuses rgb_third_1 data)
        this._navMainImg = new Image();
        this._navMainImg.onload = () => {
            if (this._navMainCtx && this._navMainCanvas) {
                this._navMainCtx.clearRect(0, 0, this._navMainCanvas.width, this._navMainCanvas.height);
                this._navMainCtx.drawImage(this._navMainImg, 0, 0, this._navMainCanvas.width, this._navMainCanvas.height);
            }
        };
    }

    update(images) {
        if (!images) return;

        for (const [key, canvasId] of Object.entries(this._feedMap)) {
            if (images[key] && this._images[key]) {
                this._images[key].src = 'data:image/png;base64,' + images[key];
            }
        }

        // Also push third_1 to the navigation tab's main camera
        if (images.rgb_third_1 && this._navMainImg) {
            this._navMainImg.src = 'data:image/png;base64,' + images.rgb_third_1;
        }
    }

    _draw(key) {
        const ctx = this._contexts[key];
        const canvas = this._canvases[key];
        const img = this._images[key];
        if (!ctx || !canvas || !img) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
}
