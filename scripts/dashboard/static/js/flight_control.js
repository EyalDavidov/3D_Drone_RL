/**
 * flight_control.js — LLC tracking errors, body-frame velocity, raw LLC I/O
 */

class FlightControlPanel {
    constructor() {
        this.els = {
            yawErr: document.getElementById('fc-yaw-err'),
            zErr: document.getElementById('fc-z-err'),
            xyPosErr: document.getElementById('fc-xy-pos-err'),
            velErrX: document.getElementById('fc-vel-err-x'),
            velErrY: document.getElementById('fc-vel-err-y'),
            velBx: document.getElementById('fc-vel-bx'),
            velBy: document.getElementById('fc-vel-by'),
            velBz: document.getElementById('fc-vel-bz'),
            desVx: document.getElementById('fc-des-vx'),
            desVy: document.getElementById('fc-des-vy'),
            desVz: document.getElementById('fc-des-vz'),
            llAct: document.getElementById('fc-ll-actions'),
            llObsGrid: document.getElementById('fc-ll-obs-grid'),
        };
        this._obsLabels = [
            'des_vx', 'des_vy', 'des_vz', 'yaw_err',
            'lin_vx', 'lin_vy', 'lin_vz',
            'ang_vx', 'ang_vy', 'ang_vz',
            'grav_x', 'grav_y', 'grav_z',
        ];
        this._obsSig = '';
    }

    update(data) {
        const fc = (data && data.flight_control) || {};
        const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d));

        if (this.els.yawErr) {
            const ye = fc.yaw_error_deg;
            this.els.yawErr.textContent = ye != null ? `${fmt(ye, 2)}°` : '—';
            this.els.yawErr.style.color = Math.abs(ye || 0) > 25 ? '#fbbf24' : '#34d399';
        }
        if (this.els.zErr) {
            const ze = fc.z_error_m;
            this.els.zErr.textContent = ze != null ? `${fmt(ze, 3)} m` : '—';
            this.els.zErr.style.color = Math.abs(ze || 0) > 0.15 ? '#fbbf24' : '#34d399';
        }
        if (this.els.xyPosErr) {
            const xy = fc.xy_pos_err_m;
            this.els.xyPosErr.textContent = xy != null ? `${fmt(xy, 3)} m` : '—';
            this.els.xyPosErr.style.color = Math.abs(xy || 0) > 0.5 ? '#fbbf24' : '#34d399';
        }

        const verr = fc.vel_err_b || [0, 0, 0];
        if (this.els.velErrX) {
            this.els.velErrX.textContent = `${fmt(verr[0], 3)} m/s`;
            this.els.velErrX.style.color = Math.abs(verr[0] || 0) > 0.25 ? '#fbbf24' : '#34d399';
        }
        if (this.els.velErrY) {
            this.els.velErrY.textContent = `${fmt(verr[1], 3)} m/s`;
            this.els.velErrY.style.color = Math.abs(verr[1] || 0) > 0.25 ? '#fbbf24' : '#34d399';
        }

        const vb = fc.lin_vel_b || [0, 0, 0];
        if (this.els.velBx) this.els.velBx.textContent = fmt(vb[0], 3);
        if (this.els.velBy) this.els.velBy.textContent = fmt(vb[1], 3);
        if (this.els.velBz) this.els.velBz.textContent = fmt(vb[2], 3);

        const dv = fc.desired_vel_b || [0, 0, 0];
        if (this.els.desVx) this.els.desVx.textContent = fmt(dv[0], 3);
        if (this.els.desVy) this.els.desVy.textContent = fmt(dv[1], 3);
        if (this.els.desVz) this.els.desVz.textContent = fmt(dv[2], 3);

        const la = fc.ll_actions || [];
        if (this.els.llAct) {
            this.els.llAct.textContent = la.length
                ? la.map(v => fmt(v, 3)).join(' · ')
                : '—';
        }

        this._updateObsGrid(fc.ll_obs || []);
    }

    _updateObsGrid(obs) {
        const grid = this.els.llObsGrid;
        if (!grid) return;
        const sig = obs.join(',');
        if (sig === this._obsSig) return;
        this._obsSig = sig;
        grid.innerHTML = '';
        if (!obs.length) {
            grid.innerHTML = '<div class="fc-obs-empty">No LLC obs</div>';
            return;
        }
        obs.forEach((val, i) => {
            const row = document.createElement('div');
            row.className = 'fc-obs-row';
            const lab = document.createElement('span');
            lab.className = 'fc-obs-label';
            lab.textContent = this._obsLabels[i] || `o${i}`;
            const num = document.createElement('span');
            num.className = 'mono fc-obs-val';
            num.textContent = Number(val).toFixed(3);
            row.appendChild(lab);
            row.appendChild(num);
            grid.appendChild(row);
        });
    }
}
