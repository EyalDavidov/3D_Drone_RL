/**
 * hud.js — Attitude indicator (artificial horizon + yaw compass)
 * Canvas-based rendering of pitch/roll horizon and yaw heading ring.
 */

class AttitudeHUD {
    constructor() {
        this.canvas = document.getElementById('canvas-hud');
        this.ctx = this.canvas.getContext('2d');
        this.size = 220;
        this.cx = this.size / 2;
        this.cy = this.size / 2;
        this.radius = 85;

        // State
        this.roll = 0;
        this.pitch = 0;
        this.yaw = 0;

        // Smoothing
        this._alpha = 0.2;
        this._roll = 0;
        this._pitch = 0;
        this._yaw = 0;

        this._draw();
    }

    /**
     * Update attitude with new telemetry.
     */
    update(data) {
        if (!data) return;
        this.roll = data.roll || 0;
        this.pitch = data.pitch || 0;
        this.yaw = data.yaw || 0;

        // Smooth
        this._roll += this._alpha * (this.roll - this._roll);
        this._pitch += this._alpha * (this.pitch - this._pitch);
        this._yaw += this._alpha * (this.yaw - this._yaw);

        this._draw();
    }

    _draw() {
        const ctx = this.ctx;
        const cx = this.cx;
        const cy = this.cy;
        const r = this.radius;

        ctx.clearRect(0, 0, this.size, this.size);

        // -- Outer ring --
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r + 12, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(99, 130, 190, 0.2)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // -- Yaw compass ticks --
        const headings = ['N', 'E', 'S', 'W'];
        for (let i = 0; i < 36; i++) {
            const angle = (i * 10 - this._yaw * 180 / Math.PI) * Math.PI / 180 - Math.PI / 2;
            const isMajor = i % 9 === 0;
            const innerR = r + (isMajor ? 6 : 10);
            const outerR = r + 14;

            ctx.beginPath();
            ctx.moveTo(cx + innerR * Math.cos(angle), cy + innerR * Math.sin(angle));
            ctx.lineTo(cx + outerR * Math.cos(angle), cy + outerR * Math.sin(angle));
            ctx.strokeStyle = isMajor ? '#60a5fa' : 'rgba(99, 130, 190, 0.3)';
            ctx.lineWidth = isMajor ? 2 : 1;
            ctx.stroke();

            if (isMajor) {
                const labelR = r + 22;
                const hIdx = i / 9;
                ctx.fillStyle = '#60a5fa';
                ctx.font = '10px Inter, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(headings[hIdx], cx + labelR * Math.cos(angle), cy + labelR * Math.sin(angle));
            }
        }
        ctx.restore();

        // -- Artificial horizon --
        ctx.save();
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.clip();

        // Rotate for roll
        ctx.translate(cx, cy);
        ctx.rotate(-this._roll);

        // Pitch offset (pixels per radian)
        const pitchPx = this._pitch * r * 2;

        // Sky
        ctx.fillStyle = '#1e3a5f';
        ctx.fillRect(-r, -r + pitchPx, r * 2, r);

        // Ground
        ctx.fillStyle = '#3b2a1a';
        ctx.fillRect(-r, pitchPx, r * 2, r);

        // Horizon line
        ctx.beginPath();
        ctx.moveTo(-r, pitchPx);
        ctx.lineTo(r, pitchPx);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Pitch ladder lines
        for (let deg = -30; deg <= 30; deg += 10) {
            if (deg === 0) continue;
            const yOff = pitchPx - (deg * Math.PI / 180) * r * 2;
            const halfW = deg < 0 ? 15 : 25;
            ctx.beginPath();
            ctx.moveTo(-halfW, yOff);
            ctx.lineTo(halfW, yOff);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.lineWidth = 1;
            ctx.stroke();

            ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
            ctx.font = '8px Inter, sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText(`${Math.abs(deg)}°`, halfW + 3, yOff + 3);
        }

        ctx.restore();

        // -- Fixed aircraft symbol --
        ctx.save();
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';

        // Center dot
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, Math.PI * 2);
        ctx.fillStyle = '#fbbf24';
        ctx.fill();

        // Wings
        ctx.beginPath();
        ctx.moveTo(cx - 30, cy);
        ctx.lineTo(cx - 12, cy);
        ctx.moveTo(cx + 12, cy);
        ctx.lineTo(cx + 30, cy);
        ctx.moveTo(cx, cy + 12);
        ctx.lineTo(cx, cy + 22);
        ctx.stroke();
        ctx.restore();

        // -- Horizon circle border --
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(99, 130, 190, 0.35)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // -- Roll indicator triangle --
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(-this._roll);
        ctx.beginPath();
        ctx.moveTo(0, -r + 2);
        ctx.lineTo(-5, -r + 10);
        ctx.lineTo(5, -r + 10);
        ctx.closePath();
        ctx.fillStyle = '#fbbf24';
        ctx.fill();
        ctx.restore();

        // -- Fixed roll reference triangle (top) --
        ctx.beginPath();
        ctx.moveTo(cx, cy - r - 1);
        ctx.lineTo(cx - 5, cy - r - 9);
        ctx.lineTo(cx + 5, cy - r - 9);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.fill();

        // -- Yaw readout --
        let yawDeg = (this._yaw * 180 / Math.PI) % 360;
        if (yawDeg < 0) yawDeg += 360;
        ctx.fillStyle = '#60a5fa';
        ctx.font = '500 11px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${yawDeg.toFixed(0)}°`, cx, cy + r + 30);

        // Roll readout
        const rollDeg = this._roll * 180 / Math.PI;
        ctx.fillStyle = '#a78bfa';
        ctx.fillText(`R ${rollDeg.toFixed(1)}°`, cx - 40, cy + r + 30);

        // Pitch readout
        const pitchDeg = this._pitch * 180 / Math.PI;
        ctx.fillStyle = '#34d399';
        ctx.fillText(`P ${pitchDeg.toFixed(1)}°`, cx + 40, cy + r + 30);
    }
}
