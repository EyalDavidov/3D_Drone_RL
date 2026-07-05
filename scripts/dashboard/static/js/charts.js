/**
 * charts.js — Live rolling charts for PPO actions and LLC outputs
 * Uses Chart.js with dark theme styling and 100-point rolling window.
 */

class LiveCharts {
    constructor() {
        this.maxPoints = 100;

        // PPO data buffers
        this.ppoData = {
            labels: [],
            vx: [], vy: [], vz: [], yaw_rate: [],
        };

        // LLC data buffers
        this.llcData = {
            labels: [],
            thrust: [], mx: [], my: [], mz: [],
        };

        this.tickCount = 0;

        // Chart.js global defaults for dark theme
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = 'rgba(99, 130, 190, 0.1)';
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.font.size = 10;

        this.ppoChart = this._createChart('chart-ppo', {
            datasets: [
                this._dataset('Vx', '#60a5fa'),
                this._dataset('Vy', '#a78bfa'),
                this._dataset('Vz', '#34d399'),
                this._dataset('Yaw Rate', '#fbbf24'),
            ],
            yLabel: 'Normalized (-1 to 1)',
            suggestedMin: -1.2,
            suggestedMax: 1.2,
        });

        this.llcChart = this._createChart('chart-llc', {
            datasets: [
                this._dataset('Thrust', '#f97316'),
                this._dataset('Mx', '#ec4899'),
                this._dataset('My', '#8b5cf6'),
                this._dataset('Mz', '#14b8a6'),
            ],
            yLabel: 'Thrust (N) / Torque (N·m)',
            suggestedMin: -0.02,
            suggestedMax: 0.6,
        });
    }

    _dataset(label, color) {
        return {
            label,
            data: [],
            borderColor: color,
            backgroundColor: color + '15',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            fill: false,
        };
    }

    _createChart(canvasId, config) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: config.datasets,
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 0 },
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            boxWidth: 10,
                            boxHeight: 2,
                            padding: 8,
                            font: { size: 9 },
                        },
                    },
                    tooltip: {
                        backgroundColor: 'rgba(17, 24, 39, 0.9)',
                        borderColor: 'rgba(99, 130, 190, 0.3)',
                        borderWidth: 1,
                        titleFont: { size: 10 },
                        bodyFont: { size: 10, family: "'JetBrains Mono', monospace" },
                        padding: 8,
                    },
                },
                scales: {
                    x: {
                        display: false,
                    },
                    y: {
                        suggestedMin: config.suggestedMin,
                        suggestedMax: config.suggestedMax,
                        grid: {
                            color: 'rgba(99, 130, 190, 0.08)',
                        },
                        ticks: {
                            font: { size: 9, family: "'JetBrains Mono', monospace" },
                            maxTicksLimit: 5,
                        },
                    },
                },
            },
        });
    }

    /**
     * Update charts with new telemetry data.
     */
    update(data) {
        if (!data) return;
        this.tickCount++;

        const label = this.tickCount.toString();

        // PPO
        if (data.ppo_actions) {
            const ppo = data.ppo_actions;
            this._pushData(this.ppoChart, label, [
                ppo.vx, ppo.vy, ppo.vz, ppo.yaw_rate,
            ]);
        }

        // LLC
        if (data.llc_outputs) {
            const llc = data.llc_outputs;
            this._pushData(this.llcChart, label, [
                llc.thrust, llc.moment_x, llc.moment_y, llc.moment_z,
            ]);
        }
    }

    _pushData(chart, label, values) {
        chart.data.labels.push(label);
        values.forEach((v, i) => {
            chart.data.datasets[i].data.push(v);
        });

        // Rolling window
        if (chart.data.labels.length > this.maxPoints) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(ds => ds.data.shift());
        }

        chart.update('none'); // no animation for performance
    }
}
