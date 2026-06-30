/**
 * metrics.js — Numerical metrics panel
 * Displays flight telemetry values with animated transitions.
 */

class MetricsPanel {
    constructor() {
        this.elements = {
            vx: document.getElementById('metric-vx'),
            vy: document.getElementById('metric-vy'),
            vz: document.getElementById('metric-vz'),
            alt: document.getElementById('metric-alt'),
            dist: document.getElementById('metric-dist'),
            thrust: document.getElementById('metric-thrust'),
        };

        // Current displayed values (for smooth animation)
        this._current = {
            vx: 0, vy: 0, vz: 0,
            alt: 0, dist: 0, thrust: 0,
        };

        // Smoothing factor
        this._alpha = 0.3;
    }

    /**
     * Update metrics with new telemetry data.
     * @param {Object} data - Full telemetry payload
     */
    update(data) {
        if (!data) return;

        const targets = {
            vx: data.lin_vel ? data.lin_vel[0] : 0,
            vy: data.lin_vel ? data.lin_vel[1] : 0,
            vz: data.lin_vel ? data.lin_vel[2] : 0,
            alt: data.pos ? data.pos[2] : 0,
            dist: data.dist_to_goal || 0,
            thrust: data.llc_outputs ? data.llc_outputs.thrust : 0,
        };

        for (const [key, target] of Object.entries(targets)) {
            // Smooth transition
            this._current[key] += this._alpha * (target - this._current[key]);
            const val = this._current[key];

            // Format based on metric
            let formatted;
            if (key === 'thrust') {
                formatted = val.toFixed(2);
            } else {
                formatted = val.toFixed(2);
            }

            // Color coding for velocity direction
            const el = this.elements[key];
            if (el) {
                el.textContent = formatted;

                // Highlight large values
                if (key === 'dist') {
                    if (val < 0.5) {
                        el.style.color = '#34d399'; // close to goal
                    } else if (val < 2.0) {
                        el.style.color = '#fbbf24'; // medium
                    } else {
                        el.style.color = '#f1f5f9'; // far
                    }
                }
            }
        }
    }
}
