/**
 * slam3d.js — Live 3D SLAM Map Scene (Three.js)
 *
 * Architectural 3D representation of the SLAM occupancy grid.
 * Walls are rendered as tall boxes (2.5 m), and overlays (drone, frontiers,
 * path, person) float inside the resulting 3D environment.
 *
 * Coordinate mapping:
 *   World X  →  Three.js -X  (NEGATED so left/right matches the Isaac sim)
 *   World Y  →  Three.js Z   (depth)
 *   Altitude →  Three.js Y   (up)
 *
 * World X is negated so the scene's left/right turn direction syncs with the
 * Isaac sim view (previously the drone appeared to turn the wrong way).
 */
class SlamScene3D {
    constructor(containerId) {
        this._container = document.getElementById(containerId);
        if (!this._container) {
            console.warn('[SlamScene3D] container not found:', containerId);
            return;
        }

        const w = this._container.clientWidth  || 640;
        const h = this._container.clientHeight || 500;

        // ---- Scene ----
        this._scene = new THREE.Scene();
        this._scene.background = new THREE.Color(0x050810);
        this._scene.fog = new THREE.Fog(0x050810, 30, 85);

        // ---- Camera — angled isometric view (like nav scene) ----
        this._camera = new THREE.PerspectiveCamera(52, w / h, 0.1, 200);
        this._camera.position.set(18, 16, 10);
        this._camera.lookAt(0, 1.5, -10);

        // ---- Renderer ----
        this._renderer = new THREE.WebGLRenderer({ antialias: true });
        this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this._renderer.setSize(w, h);
        this._renderer.shadowMap.enabled = false;
        this._container.appendChild(this._renderer.domElement);

        // ---- Orbit Controls ----
        if (typeof THREE.OrbitControls !== 'undefined') {
            this._controls = new THREE.OrbitControls(this._camera, this._renderer.domElement);
            this._controls.target.set(0, 1.5, -10);
            this._controls.enableDamping  = true;
            this._controls.dampingFactor  = 0.08;
            this._controls.minDistance    = 4;
            this._controls.maxDistance    = 90;
        } else {
            this._controls = null;
        }

        // ---- Lighting ----
        this._scene.add(new THREE.AmbientLight(0x1a2840, 2.0));
        const sun = new THREE.DirectionalLight(0xffffff, 0.6);
        sun.position.set(12, 30, 8);
        this._scene.add(sun);
        const fill = new THREE.DirectionalLight(0x334466, 0.3);
        fill.position.set(-8, 10, -15);
        this._scene.add(fill);

        // ---- Dark floor (single large plane, matches nav scene) ----
        const floorGeo = new THREE.PlaneGeometry(120, 120);
        const floorMat = new THREE.MeshStandardMaterial({
            color: 0x080c14, roughness: 0.95, metalness: 0.05,
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = -0.01;
        this._scene.add(floor);

        // ---- Cyan grid (matches nav scene aesthetic) ----
        const gridHelper = new THREE.GridHelper(100, 100, 0x0d2035, 0x0d2035);
        gridHelper.position.y = 0.005;
        this._scene.add(gridHelper);

        // ---- Occupancy walls (InstancedMesh, rebuilt when grid changes) ----
        this._wallMesh   = null;
        this._obsMesh    = null;
        this._lastGridVer = -1;
        this._controlsCentered = false;

        // ---- Overlay groups ----
        this._frontierGroup = new THREE.Group();
        this._scene.add(this._frontierGroup);

        this._pathLine   = null;
        this._targetGroup = null;
        this._targetRing  = null;

        this._buildDrone();
        this._buildTarget();
        this._buildPerson();
        this._spawnedTargetMeshes = [];
        this._personConfThreshold = 0.7;

        // ---- Shared geometry for frontier pillars ----
        this._fCylGeo = new THREE.CylinderGeometry(0.08, 0.12, 0.8, 8);
        this._fSphGeo = new THREE.SphereGeometry(0.15, 8, 6);
        this._fMat    = new THREE.MeshPhongMaterial({
            color: 0xff8800, emissive: 0x661e00, transparent: true, opacity: 0.92,
        });

        // ---- Animation ----
        this._t = 0;
        this._animate();

        // ---- Resize ----
        if (window.ResizeObserver) {
            new ResizeObserver(() => this._resize()).observe(this._container);
        }
        window.addEventListener('resize', () => this._resize());
    }

    // ---------------------------------------------------------------
    //  Public API
    // ---------------------------------------------------------------

    /**
     * @param {object} slam3d  — slam_3d telemetry dict from Python
     * @param {Array}  roomBounds — unused (kept for API compatibility)
     */
    update(slam3d, roomBounds) {
        if (!this._renderer) return;

        if (!slam3d) return;

        // Rebuild wall geometry whenever the live map changes anywhere.
        // Uses the server-computed grid version (adler32 over the whole grid)
        // so ANY change triggers a rebuild — the map now builds dynamically.
        if (slam3d.grid && slam3d.H && slam3d.W) {
            const ver = (slam3d.gver !== undefined) ? slam3d.gver : slam3d.grid.length;
            if (ver !== this._lastGridVer) {
                this._lastGridVer = ver;
                this._rebuildWalls(slam3d);
            }
        }

        this._updateDrone(slam3d.drone);
        this._updateFrontiers(slam3d.frontiers, slam3d.active);
        this._updateTarget(slam3d.active);
        this._updatePath(slam3d.path);
        this._personConfThreshold = (typeof slam3d.person_conf_threshold === 'number')
            ? slam3d.person_conf_threshold
            : 0.7;
        this._updateSpawnedTargets(slam3d.spawned_targets, slam3d.persons);
        if (!slam3d.spawned_targets || slam3d.spawned_targets.length === 0) {
            this._updatePerson(slam3d.person);
        } else if (this._personGroup) {
            this._personGroup.visible = false;
        }

        if (slam3d.blueprint && !this._blueprintMesh) {
            this._buildBlueprint(slam3d.blueprint, slam3d.res || 0.4);
        }
    }

    _handleResize() { this._resize(); }

    _buildBlueprint(blueprint, res) {
        if (this._blueprintMesh) {
            this._scene.remove(this._blueprintMesh);
            this._blueprintMesh.geometry.dispose();
            this._blueprintMesh.material.dispose();
            this._blueprintMesh = null;
        }

        const count = blueprint.length;
        if (count === 0) return;

        // Blueprint material: cool semi-transparent cyan wireframe/grid pattern
        const mat = new THREE.MeshBasicMaterial({
            color: 0x00f0ff,
            transparent: true,
            opacity: 0.12,
            wireframe: true,
            depthWrite: false,
        });

        const BLUEPRINT_H = 2.5;
        const geo = new THREE.BoxGeometry(res * 0.95, BLUEPRINT_H, res * 0.95);
        const mesh = new THREE.InstancedMesh(geo, mat, count);
        const dummy = new THREE.Object3D();

        for (let i = 0; i < count; i++) {
            const [wx, wy] = blueprint[i];
            const tx = -wx; // world X → Three -X
            const tz = wy;  // world Y → Three Z
            dummy.position.set(tx, BLUEPRINT_H / 2, tz);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
        }

        mesh.instanceMatrix.needsUpdate = true;
        this._scene.add(mesh);
        this._blueprintMesh = mesh;
    }

    // ---------------------------------------------------------------
    //  Wall rebuild (tall architectural boxes from occupancy grid)
    // ---------------------------------------------------------------

    _rebuildWalls(d) {
        const { grid, H, W, min_x, max_x, min_y, max_y, cell_w, cell_d } = d;

        // Decode the base64 quantised grid
        // (0=unknown 1=free 2=danger 3=dodgeable-obstacle 4=structural-wall)
        const binStr = atob(grid);
        const bytes  = new Uint8Array(binStr.length);
        for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);

        const disposeMesh = (m) => {
            if (!m) return;
            this._scene.remove(m);
            m.geometry.dispose();
            m.material.dispose();
        };
        disposeMesh(this._wallMesh); this._wallMesh = null;
        disposeMesh(this._obsMesh);  this._obsMesh  = null;

        // Build an InstancedMesh for all cells matching `value`.
        const buildLayer = (value, material, height, yBase) => {
            let count = 0;
            for (let i = 0; i < bytes.length; i++) if (bytes[i] === value) count++;
            count = Math.min(count, 60000);
            if (count === 0) return null;
            const geo  = new THREE.BoxGeometry(cell_w, height, cell_d);
            const mesh = new THREE.InstancedMesh(geo, material, count);
            mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            const dummy = new THREE.Object3D();
            let idx = 0;
            for (let r = 0; r < H && idx < count; r++) {
                const rowOff = r * W;
                for (let c = 0; c < W && idx < count; c++) {
                    if (bytes[rowOff + c] === value) {
                        const wx = -(min_x + (c + 0.5) * cell_w);  // world X → Three -X
                        const wz = min_y + (r + 0.5) * cell_d;      // world Y → Three Z
                        dummy.position.set(wx, yBase, wz);
                        dummy.updateMatrix();
                        mesh.setMatrixAt(idx, dummy.matrix);
                        idx++;
                    }
                }
            }
            mesh.count = idx;
            mesh.instanceMatrix.needsUpdate = true;
            mesh.frustumCulled = false;
            this._scene.add(mesh);
            return mesh;
        };

        // Structural walls — tall warm yellow/gold (outer house boundary).
        const WALL_H = 2.5;
        const wallMat = new THREE.MeshStandardMaterial({
            color: 0xffcc33, emissive: 0x443300, emissiveIntensity: 0.35,
            roughness: 0.75, metalness: 0.08,
        });
        this._wallMesh = buildLayer(4, wallMat, WALL_H, WALL_H / 2);

        // Dodgeable obstacles (poles/props) — short teal blocks, clearly NOT walls.
        // The drone is allowed to route through these; the PPO policy weaves around.
        const OBS_H = 1.1;
        const obsMat = new THREE.MeshStandardMaterial({
            color: 0x1f9c8c, emissive: 0x073d36, emissiveIntensity: 0.35,
            roughness: 0.55, metalness: 0.15, transparent: true, opacity: 0.85,
        });
        this._obsMesh = buildLayer(3, obsMat, OBS_H, OBS_H / 2);

        if (!this._wallMesh && !this._obsMesh) return;

        // Re-centre orbit controls once when the first wall batch arrives
        if (this._controls && !this._controlsCentered) {
            const ctrX = -(min_x + max_x) / 2;
            const ctrZ = (min_y + max_y) / 2;
            this._controls.target.set(ctrX, 1.5, ctrZ);
            this._controlsCentered = true;
        }
    }

    // ---------------------------------------------------------------
    //  Overlay updates
    // ---------------------------------------------------------------

    _updateDrone(drone) {
        if (!drone || !this._droneGroup) return;
        const flyH = Math.max(drone.z || 0, 0.4) + 0.6;
        this._droneGroup.position.set(-drone.x, flyH, drone.y);
        // World X → Three -X: heading (cos yaw, sin yaw) maps to Three (-cos yaw, sin yaw).
        // Cone points +Z locally, giving rotation.y = yaw - π/2.
        this._droneGroup.rotation.y = drone.yaw - Math.PI / 2;
    }

    _updateFrontiers(frontiers, active) {
        while (this._frontierGroup.children.length > 0) {
            this._frontierGroup.remove(this._frontierGroup.children[0]);
        }
        if (!frontiers) return;

        for (const f of frontiers) {
            if (active && Math.abs(f[0] - active[0]) < 0.3 && Math.abs(f[1] - active[1]) < 0.3) continue;

            // Glowing pillar at floor level (looks like a waypoint marker)
            const group = new THREE.Group();

            const cyl  = new THREE.Mesh(this._fCylGeo, this._fMat);
            cyl.position.y = 0.4;
            group.add(cyl);

            const top = new THREE.Mesh(this._fSphGeo, this._fMat);
            top.position.y = 0.9;
            group.add(top);

            // Ground ring
            const rGeo = new THREE.RingGeometry(0.18, 0.26, 16);
            const rMat = new THREE.MeshBasicMaterial({
                color: 0xff8800, transparent: true, opacity: 0.4,
                side: THREE.DoubleSide,
            });
            const ring = new THREE.Mesh(rGeo, rMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = 0.01;
            group.add(ring);

            const light = new THREE.PointLight(0xff6600, 0.35, 3.0);
            light.position.y = 0.5;
            group.add(light);

            group.position.set(-f[0], 0, f[1]);
            this._frontierGroup.add(group);
        }
    }

    _updateTarget(active) {
        if (!this._targetGroup) return;
        if (active) {
            this._targetGroup.position.set(-active[0], 0.0, active[1]);
            this._targetGroup.visible = true;
        } else {
            this._targetGroup.visible = false;
        }
    }

    _updatePath(path) {
        if (this._pathLine) {
            this._scene.remove(this._pathLine);
            this._pathLine.geometry.dispose();
            this._pathLine = null;
        }
        if (!path || path.length < 2) return;
        const pts = path.map(([x, y]) => new THREE.Vector3(-x, 0.08, y));
        const geo = new THREE.BufferGeometry().setFromPoints(pts);
        const mat = new THREE.LineBasicMaterial({ color: 0x00e040, linewidth: 2 });
        this._pathLine = new THREE.Line(geo, mat);
        this._scene.add(this._pathLine);
    }

    _updatePerson(person) {
        if (!this._personGroup) return;
        const p = this._normPerson(person);
        if (p) {
            this._personGroup.position.set(-p.x, 0, p.y);
            this._personGroup.visible = true;
            const oldLabel = this._personGroup.children.find(c => c.isSprite);
            if (oldLabel) this._personGroup.remove(oldLabel);
            const label = this._makeConfSprite(p.conf);
            if (label) this._personGroup.add(label);
        } else {
            this._personGroup.visible = false;
        }
    }

    // ---------------------------------------------------------------
    //  Object builders
    // ---------------------------------------------------------------

    _buildDrone() {
        this._droneGroup = new THREE.Group();
        this._droneProps = [];

        const bodyMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b, emissive: 0x0a1628, emissiveIntensity: 0.35,
            roughness: 0.35, metalness: 0.72,
        });
        const accentMat = new THREE.MeshStandardMaterial({
            color: 0x22ff66, emissive: 0x118833, emissiveIntensity: 0.55,
            roughness: 0.4, metalness: 0.3,
        });
        const armMat = new THREE.MeshStandardMaterial({
            color: 0x64748b, roughness: 0.45, metalness: 0.65,
        });
        const motorMat = new THREE.MeshStandardMaterial({
            color: 0x334155, roughness: 0.3, metalness: 0.8,
        });
        const propMat = new THREE.MeshStandardMaterial({
            color: 0x38bdf8, transparent: true, opacity: 0.45, roughness: 0.7,
        });
        const camMat = new THREE.MeshStandardMaterial({
            color: 0x0f172a, roughness: 0.5, metalness: 0.6,
        });

        // Central body plate
        const body = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.05, 0.16), bodyMat);
        body.position.y = 0.04;
        this._droneGroup.add(body);

        const deck = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.015, 0.11), accentMat);
        deck.position.y = 0.075;
        this._droneGroup.add(deck);

        // Camera pod under nose
        const cam = new THREE.Mesh(new THREE.SphereGeometry(0.028, 8, 6), camMat);
        cam.position.set(0, 0.01, 0.06);
        this._droneGroup.add(cam);

        const armLen = 0.34;
        for (let i = 0; i < 4; i++) {
            const angle = Math.PI / 4 + i * (Math.PI / 2);
            const ex = Math.cos(angle) * armLen * 0.5;
            const ez = Math.sin(angle) * armLen * 0.5;

            const arm = new THREE.Mesh(new THREE.BoxGeometry(armLen, 0.03, 0.03), armMat);
            arm.position.set(Math.cos(angle) * armLen * 0.22, 0.04, Math.sin(angle) * armLen * 0.22);
            arm.rotation.y = -angle;
            this._droneGroup.add(arm);

            const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.04, 0.045, 10), motorMat);
            motor.position.set(ex, 0.05, ez);
            this._droneGroup.add(motor);

            const prop = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.13, 0.006, 18), propMat);
            prop.position.set(ex, 0.085, ez);
            prop.userData.spinSign = i % 2 === 0 ? 1 : -1;
            this._droneGroup.add(prop);
            this._droneProps.push(prop);
        }

        // Front heading marker (+Z)
        const noseMat = new THREE.MeshStandardMaterial({
            color: 0xff3355, emissive: 0xaa1122, emissiveIntensity: 0.7,
        });
        const nose = new THREE.Mesh(new THREE.ConeGeometry(0.035, 0.1, 4), noseMat);
        nose.rotation.x = Math.PI / 2;
        nose.position.set(0, 0.05, 0.13);
        this._droneGroup.add(nose);

        const glow = new THREE.PointLight(0x22ff66, 0.55, 4);
        glow.position.y = 0.12;
        this._droneGroup.add(glow);

        this._droneGroup.position.set(0, 2.0, 1.5);
        this._scene.add(this._droneGroup);
    }

    _buildTarget() {
        this._targetGroup = new THREE.Group();

        // Spike pillar (vertical cylinder)
        const pillarMat = new THREE.MeshStandardMaterial({
            color: 0x00c8ff, emissive: 0x003d5c, emissiveIntensity: 0.5,
            transparent: true, opacity: 0.75,
            roughness: 0.3,
        });
        const pillar = new THREE.Mesh(
            new THREE.CylinderGeometry(0.07, 0.12, 2.8, 8), pillarMat
        );
        pillar.position.y = 1.4;
        this._targetGroup.add(pillar);

        // Top sphere
        const topSph = new THREE.Mesh(
            new THREE.SphereGeometry(0.28, 12, 10),
            new THREE.MeshStandardMaterial({
                color: 0x00d4ff, emissive: 0x005566, emissiveIntensity: 0.55,
                transparent: true, opacity: 0.80,
                roughness: 0.25,
            })
        );
        topSph.position.y = 2.95;
        this._targetGroup.add(topSph);

        // Ground rings (spinning)
        for (let ri = 0; ri < 2; ri++) {
            const rGeo = new THREE.RingGeometry(0.35 + ri * 0.22, 0.44 + ri * 0.22, 32);
            const rMat = new THREE.MeshBasicMaterial({
                color: 0x00d4ff, transparent: true, opacity: 0.35 - ri * 0.1,
                side: THREE.DoubleSide,
            });
            const ring = new THREE.Mesh(rGeo, rMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = 0.02;
            ring._rotOffset = ri * Math.PI / 2;
            this._targetGroup.add(ring);
            if (ri === 0) this._targetRing = ring;
        }

        // Glow light
        const tLight = new THREE.PointLight(0x00d4ff, 0.8, 6);
        tLight.position.y = 1.5;
        this._targetGroup.add(tLight);

        this._targetGroup.visible = false;
        this._scene.add(this._targetGroup);
    }

    _buildPerson() {
        this._personGroup = this._buildMiniPersonFigure(0xff00cc, 0x880066);
        this._personGroup.visible = false;
        this._scene.add(this._personGroup);
    }

    _normPerson(p) {
        if (!p) return null;
        if (Array.isArray(p)) return { x: p[0], y: p[1], conf: null, label: null };
        return { x: p.x, y: p.y, conf: p.conf, label: p.label };
    }

    _makeConfSprite(conf, isDetected = true) {
        if (conf == null) return null;
        const pct = Math.round(conf * 100);
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 24;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = isDetected ? 'rgba(0,0,0,0.55)' : 'rgba(24, 7, 24, 0.72)';
        ctx.fillRect(0, 0, 64, 24);
        ctx.font = 'bold 14px monospace';
        ctx.fillStyle = isDetected ? '#e8ffc8' : '#ffc7f1';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${pct}%`, 32, 12);
        const tex = new THREE.CanvasTexture(canvas);
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true });
        const sprite = new THREE.Sprite(mat);
        sprite.scale.set(0.55, 0.2, 1);
        sprite.position.y = 1.05;
        return sprite;
    }

    _isSpawnTargetDetected(tgt, persons) {
        const pts = persons || [];
        let bestAboveThreshold = null;
        let bestAnyConf = null;
        let hasUnscoredHit = false;
        for (const raw of pts) {
            const p = this._normPerson(raw);
            if (!p) continue;
            if (Math.hypot(p.x - tgt[0], p.y - tgt[1]) < 1.5) {
                if (p.conf != null) {
                    bestAnyConf = Math.max(bestAnyConf || 0, p.conf);
                    if (p.conf >= this._personConfThreshold) {
                        bestAboveThreshold = Math.max(bestAboveThreshold || 0, p.conf);
                    }
                } else {
                    hasUnscoredHit = true;
                }
            }
        }
        if (bestAboveThreshold != null) return { detected: true, conf: bestAboveThreshold };
        if (hasUnscoredHit) return { detected: true, conf: bestAnyConf };
        if (bestAnyConf != null) return { detected: false, conf: bestAnyConf };
        return { detected: false, conf: null };
    }

    _updateSpawnedTargets(targets, persons) {
        for (const m of this._spawnedTargetMeshes) this._scene.remove(m);
        this._spawnedTargetMeshes = [];
        if (!targets || targets.length === 0) return;
        for (const tgt of targets) {
            const hit = this._isSpawnTargetDetected(tgt, persons);
            const isDetected = hit.detected;
            const col = isDetected ? 0x22ff66 : 0xff2d95;
            const emCol = isDetected ? 0x0a6630 : 0x880044;
            const g = this._buildMiniPersonFigure(col, emCol);
            g.position.set(-tgt[0], 0, tgt[1]);
            const label = this._makeConfSprite(hit.conf, isDetected);
            if (label) g.add(label);
            this._scene.add(g);
            this._spawnedTargetMeshes.push(g);
        }
    }

    _buildMiniPersonFigure(col, emCol) {
        const g = new THREE.Group();
        const mat = new THREE.MeshStandardMaterial({
            color: col, emissive: emCol, emissiveIntensity: 0.65, roughness: 0.38, metalness: 0.1,
        });

        const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 10, 10), mat);
        head.position.y = 0.68;
        g.add(head);

        const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.11, 0.32, 10), mat);
        torso.position.y = 0.42;
        g.add(torso);

        for (const side of [-1, 1]) {
            const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.032, 0.22, 6), mat);
            arm.position.set(side * 0.14, 0.48, 0);
            arm.rotation.z = side * 0.55;
            g.add(arm);
        }

        for (const side of [-1, 1]) {
            const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.038, 0.26, 6), mat);
            leg.position.set(side * 0.055, 0.13, 0.02);
            leg.rotation.x = 0.08;
            g.add(leg);
        }

        const ringMat = new THREE.MeshBasicMaterial({
            color: col, transparent: true, opacity: 0.32, side: THREE.DoubleSide,
        });
        const ring = new THREE.Mesh(new THREE.RingGeometry(0.14, 0.19, 28), ringMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.01;
        g.add(ring);

        const light = new THREE.PointLight(col, 0.45, 2.5);
        light.position.y = 0.45;
        g.add(light);

        g.userData.isMiniPerson = true;
        return g;
    }

    // ---------------------------------------------------------------
    //  Animation
    // ---------------------------------------------------------------

    _animate() {
        requestAnimationFrame(() => this._animate());
        this._t = Date.now() * 0.001;

        // Pulse frontier pillars
        this._frontierGroup.children.forEach((g, i) => {
            const s = 0.82 + 0.18 * Math.sin(this._t * 2.8 + i * 1.4);
            g.scale.y = s;
            const light = g.children.find(c => c.isPointLight);
            if (light) light.intensity = 0.25 + 0.15 * Math.sin(this._t * 4 + i);
        });

        // Spin target rings
        if (this._targetGroup && this._targetGroup.visible) {
            this._targetGroup.children.forEach(child => {
                if (child.geometry && child.geometry.type === 'RingGeometry') {
                    const off = child._rotOffset || 0;
                    child.rotation.z = this._t * 1.2 + off;
                }
            });
            // Bob beacon
            const beacon = this._targetGroup.children.find(c => c.geometry && c.geometry.type === 'SphereGeometry');
            if (beacon) beacon.position.y = 2.95 + 0.18 * Math.sin(this._t * 2.0);
        }

        // Spin propellers
        if (this._droneProps) {
            for (const prop of this._droneProps) {
                prop.rotation.y += 0.35 * (prop.userData.spinSign || 1);
            }
        }

        // Pulse drone glow
        if (this._droneGroup) {
            const light = this._droneGroup.children.find(c => c.isPointLight);
            if (light) light.intensity = 0.55 + 0.25 * Math.sin(this._t * 5.5);
        }

        // Bob spawned mini-person markers
        this._spawnedTargetMeshes.forEach((g, i) => {
            if (!g.userData.isMiniPerson) return;
            g.position.y = 0.04 * Math.sin(this._t * 2.4 + i * 1.1);
            const light = g.children.find(c => c.isPointLight);
            if (light) light.intensity = 0.35 + 0.12 * Math.sin(this._t * 3 + i);
        });

        // Bob person marker
        if (this._personGroup && this._personGroup.visible) {
            this._personGroup.position.y = 0.04 * Math.sin(this._t * 2.2);
        }

        if (this._controls) this._controls.update();
        this._renderer.render(this._scene, this._camera);
    }

    // ---------------------------------------------------------------
    //  Resize
    // ---------------------------------------------------------------

    _resize() {
        if (!this._container || !this._renderer) return;
        const w = this._container.clientWidth;
        const h = this._container.clientHeight;
        if (w < 1 || h < 1) return;
        this._camera.aspect = w / h;
        this._camera.updateProjectionMatrix();
        this._renderer.setSize(w, h);
    }
}
