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
        this._updatePerson(slam3d.person);
    }

    _handleResize() { this._resize(); }

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
        if (person) {
            this._personGroup.position.set(-person[0], 0, person[1]);
            this._personGroup.visible = true;
        } else {
            this._personGroup.visible = false;
        }
    }

    // ---------------------------------------------------------------
    //  Object builders
    // ---------------------------------------------------------------

    _buildDrone() {
        this._droneGroup = new THREE.Group();

        // Body
        const bMat = new THREE.MeshStandardMaterial({
            color: 0x00e040, emissive: 0x00a020, emissiveIntensity: 0.6,
            roughness: 0.3, metalness: 0.5,
        });
        this._droneGroup.add(new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 8), bMat));

        // Arms
        const armMat = new THREE.MeshStandardMaterial({
            color: 0x90a0b0, roughness: 0.5, metalness: 0.6,
        });
        for (let i = 0; i < 2; i++) {
            const arm = new THREE.Mesh(new THREE.BoxGeometry(0.78, 0.04, 0.04), armMat);
            arm.rotation.y = i * Math.PI / 2;
            this._droneGroup.add(arm);
        }

        // Propeller discs (semi-transparent)
        const propMat = new THREE.MeshStandardMaterial({
            color: 0x60a5fa, transparent: true, opacity: 0.40,
            roughness: 0.8,
        });
        for (const [ox, oz] of [[0.39, 0], [-0.39, 0], [0, 0.39], [0, -0.39]]) {
            const prop = new THREE.Mesh(
                new THREE.CylinderGeometry(0.13, 0.13, 0.02, 10), propMat
            );
            prop.position.set(ox, 0.04, oz);
            this._droneGroup.add(prop);
        }

        // Direction cone (+Z = facing forward in Three.js local space)
        const coneMat = new THREE.MeshStandardMaterial({
            color: 0xef4444, emissive: 0xaa0000, emissiveIntensity: 0.5,
        });
        const cone = new THREE.Mesh(new THREE.ConeGeometry(0.055, 0.18, 4), coneMat);
        cone.rotation.x = Math.PI / 2;
        cone.position.set(0, 0, 0.36);
        this._droneGroup.add(cone);

        // Glow
        const glow = new THREE.PointLight(0x00ff44, 0.7, 5);
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
        this._personGroup = new THREE.Group();

        // Tall pillar
        const pMat = new THREE.MeshStandardMaterial({
            color: 0xff00cc, emissive: 0x880066, emissiveIntensity: 0.6,
            roughness: 0.35,
        });
        const pillar = new THREE.Mesh(
            new THREE.CylinderGeometry(0.06, 0.10, 3.0, 8), pMat
        );
        pillar.position.y = 1.5;
        this._personGroup.add(pillar);

        // Top beacon sphere
        const bGeo = new THREE.SphereGeometry(0.30, 10, 8);
        const bMat = new THREE.MeshStandardMaterial({
            color: 0xff00cc, emissive: 0xcc0099, emissiveIntensity: 0.7,
        });
        const beacon = new THREE.Mesh(bGeo, bMat);
        beacon.position.y = 3.2;
        this._personGroup.add(beacon);

        // Ground ring
        const lGeo = new THREE.RingGeometry(0.4, 0.54, 32);
        const lMat = new THREE.MeshBasicMaterial({
            color: 0xff00cc, transparent: true, opacity: 0.45,
            side: THREE.DoubleSide,
        });
        const ring = new THREE.Mesh(lGeo, lMat);
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = 0.02;
        this._personGroup.add(ring);

        // Glow
        const light = new THREE.PointLight(0xff00ff, 1.2, 8);
        light.position.y = 2.0;
        this._personGroup.add(light);

        this._personGroup.visible = false;
        this._scene.add(this._personGroup);
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

        // Pulse drone glow
        if (this._droneGroup) {
            const light = this._droneGroup.children.find(c => c.isPointLight);
            if (light) light.intensity = 0.55 + 0.25 * Math.sin(this._t * 5.5);
        }

        // Bob person beacon
        if (this._personGroup && this._personGroup.visible) {
            const beacon = this._personGroup.children.find(c => c.geometry && c.geometry.type === 'SphereGeometry');
            if (beacon) beacon.position.y = 3.2 + 0.20 * Math.sin(this._t * 2.2);
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
