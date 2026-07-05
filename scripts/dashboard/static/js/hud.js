/**
 * hud.js — Attitude indicator (artificial horizon + yaw compass)
 * Canvas-based, sized to fit as a small overlay on the 3D map.
 */

class AttitudeHUD {
    constructor() {
        this.canvas = document.getElementById('canvas-hud');
        this.ctx = this.canvas.getContext('2d');
        this.size = 160;
        this.cx = this.size / 2;
        this.cy = this.size / 2;
        this.radius = 60;

        this.roll = 0; this.pitch = 0; this.yaw = 0;
        this._roll = 0; this._pitch = 0; this._yaw = 0;
        this._alpha = 0.2;
        this._draw();
    }

    update(data) {
        if (!data) return;
        this.roll = data.roll || 0;
        this.pitch = data.pitch || 0;
        this.yaw = data.yaw || 0;
        this._roll += this._alpha * (this.roll - this._roll);
        this._pitch += this._alpha * (this.pitch - this._pitch);
        this._yaw += this._alpha * (this.yaw - this._yaw);
        this._draw();
    }

    _draw() {
        const ctx = this.ctx;
        const cx = this.cx, cy = this.cy, r = this.radius;
        ctx.clearRect(0, 0, this.size, this.size);

        // Outer ring
        ctx.beginPath();
        ctx.arc(cx, cy, r + 10, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(99, 130, 190, 0.2)';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Yaw ticks
        const headings = ['N', 'E', 'S', 'W'];
        for (let i = 0; i < 36; i++) {
            const angle = (i * 10 - this._yaw * 180 / Math.PI) * Math.PI / 180 - Math.PI / 2;
            const isMajor = i % 9 === 0;
            const inner = r + (isMajor ? 4 : 8);
            const outer = r + 12;
            ctx.beginPath();
            ctx.moveTo(cx + inner * Math.cos(angle), cy + inner * Math.sin(angle));
            ctx.lineTo(cx + outer * Math.cos(angle), cy + outer * Math.sin(angle));
            ctx.strokeStyle = isMajor ? '#60a5fa' : 'rgba(99, 130, 190, 0.25)';
            ctx.lineWidth = isMajor ? 1.5 : 0.8;
            ctx.stroke();
            if (isMajor) {
                const lr = r + 18;
                ctx.fillStyle = '#60a5fa';
                ctx.font = '8px Inter, sans-serif';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(headings[i / 9], cx + lr * Math.cos(angle), cy + lr * Math.sin(angle));
            }
        }

        // Artificial horizon
        ctx.save();
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.clip();
        ctx.translate(cx, cy); ctx.rotate(-this._roll);
        const pitchPx = this._pitch * r * 2;
        ctx.fillStyle = '#1e3a5f'; ctx.fillRect(-r, -r + pitchPx, r * 2, r);
        ctx.fillStyle = '#3b2a1a'; ctx.fillRect(-r, pitchPx, r * 2, r);
        ctx.beginPath(); ctx.moveTo(-r, pitchPx); ctx.lineTo(r, pitchPx);
        ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1; ctx.stroke();
        ctx.restore();

        // Aircraft symbol
        ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 2; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI * 2); ctx.fillStyle = '#fbbf24'; ctx.fill();
        ctx.beginPath();
        ctx.moveTo(cx - 22, cy); ctx.lineTo(cx - 9, cy);
        ctx.moveTo(cx + 9, cy); ctx.lineTo(cx + 22, cy);
        ctx.moveTo(cx, cy + 9); ctx.lineTo(cx, cy + 16);
        ctx.stroke();

        // Circle border
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(99, 130, 190, 0.3)'; ctx.lineWidth = 1.5; ctx.stroke();

        // Roll triangle
        ctx.save(); ctx.translate(cx, cy); ctx.rotate(-this._roll);
        ctx.beginPath(); ctx.moveTo(0, -r + 2); ctx.lineTo(-4, -r + 8); ctx.lineTo(4, -r + 8); ctx.closePath();
        ctx.fillStyle = '#fbbf24'; ctx.fill(); ctx.restore();

        // Readouts
        let yawDeg = (this._yaw * 180 / Math.PI) % 360; if (yawDeg < 0) yawDeg += 360;
        ctx.font = '500 9px JetBrains Mono, monospace'; ctx.textAlign = 'center';
        ctx.fillStyle = '#60a5fa'; ctx.fillText(`${yawDeg.toFixed(0)}°`, cx, cy + r + 24);
        ctx.fillStyle = '#a78bfa'; ctx.fillText(`R ${(this._roll * 180 / Math.PI).toFixed(1)}°`, cx - 30, cy + r + 24);
        ctx.fillStyle = '#34d399'; ctx.fillText(`P ${(this._pitch * 180 / Math.PI).toFixed(1)}°`, cx + 30, cy + r + 24);
    }
}
