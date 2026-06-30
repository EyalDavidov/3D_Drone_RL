/**
 * cameras.js — Camera feed renderer
 * Renders Base64 PNG images to canvas elements for RGB, Depth, and AE feeds.
 */

class CameraFeeds {
    constructor() {
        this.canvases = {
            rgb: document.getElementById('canvas-rgb'),
            depth: document.getElementById('canvas-depth'),
            ae: document.getElementById('canvas-ae'),
        };

        this.contexts = {};
        for (const [key, canvas] of Object.entries(this.canvases)) {
            this.contexts[key] = canvas.getContext('2d');
        }

        // Reusable Image objects to avoid GC
        this._images = {
            rgb: new Image(),
            depth: new Image(),
            ae: new Image(),
        };

        // Bind draw callbacks
        for (const key of Object.keys(this._images)) {
            this._images[key].onload = () => this._drawImage(key);
        }
    }

    /**
     * Update camera feeds with new image data.
     * @param {Object} images - { rgb, depth, ae_recon } base64 PNG strings
     */
    update(images) {
        if (!images) return;

        if (images.rgb) {
            this._images.rgb.src = 'data:image/png;base64,' + images.rgb;
        }
        if (images.depth) {
            this._images.depth.src = 'data:image/png;base64,' + images.depth;
        }
        if (images.ae_recon) {
            this._images.ae.src = 'data:image/png;base64,' + images.ae_recon;
        }
    }

    /**
     * Draw a loaded image to its canvas, scaling to fill.
     */
    _drawImage(key) {
        const ctx = this.contexts[key];
        const canvas = this.canvases[key];
        const img = this._images[key];

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Apply a slight color tint for depth/ae
        if (key === 'depth') {
            ctx.filter = 'sepia(0.1) saturate(1.5) hue-rotate(180deg)';
        } else if (key === 'ae') {
            ctx.filter = 'sepia(0.1) saturate(1.3) hue-rotate(260deg)';
        } else {
            ctx.filter = 'none';
        }

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        ctx.filter = 'none';
    }
}
