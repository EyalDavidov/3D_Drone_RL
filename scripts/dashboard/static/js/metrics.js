/**
 * metrics.js — Numerical metrics for the 3D map overlay
 */

class MetricsPanel {
    constructor() {
        this.elements = {
            speed: document.getElementById('metric-speed'),
            alt: document.getElementById('metric-alt'),
            dist: document.getElementById('metric-dist'),
            thrust: document.getElementById('metric-thrust'),
        };

        this._current = { speed: 0, alt: 0, dist: 0, thrust: 0 };
        this._alpha = 0.3;
    }

    update(data) {
        if (!data) return;

        const vx = data.lin_vel ? data.lin_vel[0] : 0;
        const vy = data.lin_vel ? data.lin_vel[1] : 0;
        const vz = data.lin_vel ? data.lin_vel[2] : 0;

        const targets = {
            speed: Math.sqrt(vx * vx + vy * vy + vz * vz),
            alt: data.pos ? data.pos[2] : 0,
            dist: data.dist_to_goal || 0,
            thrust: data.llc_outputs ? data.llc_outputs.thrust : 0,
        };

        for (const [key, target] of Object.entries(targets)) {
            this._current[key] += this._alpha * (target - this._current[key]);
            const el = this.elements[key];
            if (el) {
                el.textContent = this._current[key].toFixed(2);
                // Color code distance
                if (key === 'dist') {
                    if (this._current[key] < 0.5) el.style.color = '#34d399';
                    else if (this._current[key] < 2.0) el.style.color = '#fbbf24';
                    else el.style.color = '#f1f5f9';
                }
            }
        }
    }
}
