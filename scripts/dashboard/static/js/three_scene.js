/**
 * three_scene.js — 3D Navigation Map
 *
 * Coordinate mapping: Sim (X, Y, Z) → Three.js (X, Z, Y)
 *   sim X = lateral     → Three X
 *   sim Y = forward     → Three Z (depth)
 *   sim Z = height      → Three Y (up in Three.js)
 * This makes the level layout HORIZONTAL instead of vertical.
 */

class NavigationScene {
    constructor() {
        this.canvas = document.getElementById('canvas-3d');
        this.container = this.canvas.parentElement;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e17);
        this.scene.fog = new THREE.FogExp2(0x0a0e17, 0.012);

        // Camera — looking at the horizontal layout from above-side
        // Use fallback dimensions if the container isn't visible yet
        const initW = this.container.clientWidth || 800;
        const initH = this.container.clientHeight || 400;
        this.camera = new THREE.PerspectiveCamera(55, initW / initH, 0.1, 200);
        this.camera.position.set(12, 14, 8);
        this._needsInitialResize = true;

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(initW, initH);

        // Controls — orbit target at center of level layout
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.target.set(0, 1, -10);
        this.controls.minDistance = 3;
        this.controls.maxDistance = 60;

        // Lighting
        this.scene.add(new THREE.AmbientLight(0x4466aa, 0.6));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 15);
        this.scene.add(dirLight);
        const ptLight = new THREE.PointLight(0x60a5fa, 0.5, 50);
        ptLight.position.set(0, 5, -10);
        this.scene.add(ptLight);

        this._buildGrid();
        this._buildRooms();
        this._buildDrone();
        this._buildTarget();
        this._buildTrail();
        this._buildPoles();

        this._polesSet = false;

        this._onResize = this._handleResize.bind(this);
        window.addEventListener('resize', this._onResize);

        if (window.ResizeObserver && this.container) {
            this._resizeObserver = new ResizeObserver(() => {
                this._handleResize();
            });
            this._resizeObserver.observe(this.container);
        }

        this._animate();
    }

    // ---- Coordinate helper: sim → Three.js ----
    _s2t(x, y, z) {
        return [x, z, y]; // sim(x,y,z) → three(x, z_sim_as_y, y_sim_as_z)
    }

    _buildGrid() {
        const grid = new THREE.GridHelper(60, 60, 0x1e3a5f, 0x0f1b2d);
        this.scene.add(grid);
    }

    _buildRooms() {
        const roomBounds = [
            [-2, 2, -3, 2, 0, 2],
            [-2, 2, -9, -2, 0, 2],
            [-2, 2, -17, -8, 0, 2],
            [-6, 2, -21, -16, 0, 2],
        ];
        const colors = [0x60a5fa, 0xa78bfa, 0x34d399, 0xfbbf24];
        this._roomMeshes = [];

        roomBounds.forEach((b, i) => {
            const [xMin, xMax, yMin, yMax, zMin, zMax] = b;
            // In Three.js coords: width=X, height=Z_sim(up)=Y_three, depth=Y_sim=Z_three
            const w = xMax - xMin;
            const h = zMax - zMin; // height (up)
            const d = yMax - yMin; // depth (forward)

            const geo = new THREE.BoxGeometry(w, h, d);
            const edges = new THREE.EdgesGeometry(geo);
            const mat = new THREE.LineBasicMaterial({ color: colors[i], transparent: true, opacity: 0.3 });
            const wireframe = new THREE.LineSegments(edges, mat);
            // Position center: three(cx_x, cx_z, cx_y)
            wireframe.position.set(
                (xMin + xMax) / 2,
                (zMin + zMax) / 2,
                (yMin + yMax) / 2
            );
            this.scene.add(wireframe);
            this._roomMeshes.push(wireframe);

            // Semi-transparent floor (in XZ plane at Y=0.01)
            const floorGeo = new THREE.PlaneGeometry(w, d);
            const floorMat = new THREE.MeshBasicMaterial({
                color: colors[i], transparent: true, opacity: 0.05, side: THREE.DoubleSide,
            });
            const floor = new THREE.Mesh(floorGeo, floorMat);
            floor.rotation.x = -Math.PI / 2;
            floor.position.set((xMin + xMax) / 2, 0.01, (yMin + yMax) / 2);
            this.scene.add(floor);

            // Level label
            const label = this._createLabel(`L${i + 1}`, colors[i]);
            const [lx, ly, lz] = this._s2t(xMin + 0.3, yMin + 0.3, zMax + 0.3);
            label.position.set(lx, ly, lz);
            this.scene.add(label);
        });
    }

    _createLabel(text, color) {
        const c = document.createElement('canvas');
        c.width = 64; c.height = 32;
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#' + new THREE.Color(color).getHexString();
        ctx.font = 'bold 20px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 32, 16);
        const tex = new THREE.CanvasTexture(c);
        const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, opacity: 0.7 });
        const sprite = new THREE.Sprite(mat);
        sprite.scale.set(1, 0.5, 1);
        return sprite;
    }

    _buildDrone() {
        this._droneGroup = new THREE.Group();

        const armMat = new THREE.MeshPhongMaterial({ color: 0xe2e8f0 });
        const armGeo = new THREE.BoxGeometry(0.6, 0.05, 0.05);

        const arm1 = new THREE.Mesh(armGeo, armMat);
        this._droneGroup.add(arm1);
        const arm2 = new THREE.Mesh(armGeo, armMat);
        arm2.rotation.y = Math.PI / 2;
        this._droneGroup.add(arm2);

        const bodyGeo = new THREE.SphereGeometry(0.08, 8, 8);
        const bodyMat = new THREE.MeshPhongMaterial({ color: 0x60a5fa, emissive: 0x2563eb, emissiveIntensity: 0.3 });
        this._droneGroup.add(new THREE.Mesh(bodyGeo, bodyMat));

        const motorColors = [0xf87171, 0x34d399, 0xfbbf24, 0xa78bfa];
        const motorPos = [[0.3, 0, 0], [-0.3, 0, 0], [0, 0, 0.3], [0, 0, -0.3]];
        motorPos.forEach((p, i) => {
            const mGeo = new THREE.SphereGeometry(0.04, 6, 6);
            const mMat = new THREE.MeshPhongMaterial({ color: motorColors[i], emissive: motorColors[i], emissiveIntensity: 0.5 });
            const m = new THREE.Mesh(mGeo, mMat);
            m.position.set(...p);
            this._droneGroup.add(m);
        });

        // Direction cone (pointing along +Z in Three.js = +Y in sim = forward)
        const coneGeo = new THREE.ConeGeometry(0.04, 0.12, 4);
        const coneMat = new THREE.MeshPhongMaterial({ color: 0xf87171, emissive: 0xf87171, emissiveIntensity: 0.4 });
        const cone = new THREE.Mesh(coneGeo, coneMat);
        cone.rotation.x = Math.PI / 2;
        cone.position.set(0, 0, 0.35);
        this._droneGroup.add(cone);

        // Start at level 1 spawn: sim(0, 1.5, 1) → three(0, 1, 1.5)
        this._droneGroup.position.set(0, 1, 1.5);
        this.scene.add(this._droneGroup);

        this._droneLight = new THREE.PointLight(0x60a5fa, 0.6, 5);
        this._droneGroup.add(this._droneLight);
    }

    _buildTarget() {
        const geo = new THREE.SphereGeometry(0.2, 16, 16);
        const mat = new THREE.MeshPhongMaterial({
            color: 0x34d399, emissive: 0x34d399, emissiveIntensity: 0.6,
            transparent: true, opacity: 0.7,
        });
        this._targetMesh = new THREE.Mesh(geo, mat);
        this._targetMesh.position.set(0, 1, -2.5); // sim(0, -2.5, 1)
        this.scene.add(this._targetMesh);

        const ringGeo = new THREE.RingGeometry(0.3, 0.35, 32);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
        this._targetRing = new THREE.Mesh(ringGeo, ringMat);
        this._targetRing.position.copy(this._targetMesh.position);
        this.scene.add(this._targetRing);

        this._targetLight = new THREE.PointLight(0x34d399, 0.5, 8);
        this._targetLight.position.copy(this._targetMesh.position);
        this.scene.add(this._targetLight);
    }

    _buildTrail() {
        this._trailMaxPoints = 80; // Shortened tail
        this._trailPoints = [];
        this._lastLevel = undefined;
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this._trailMaxPoints * 3), 3));
        geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this._trailMaxPoints * 3), 3));
        geo.setDrawRange(0, 0);
        const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.8 });
        this._trailLine = new THREE.Line(geo, mat);
        this.scene.add(this._trailLine);
    }

    _buildPoles() {
        this._poleGroup = new THREE.Group();
        this.scene.add(this._poleGroup);
    }

    // ---- Update ----

    update(data) {
        if (!data) return;

        // Drone position: sim(x,y,z) → three(x, z, y)
        if (data.pos) {
            const [tx, ty, tz] = this._s2t(data.pos[0], data.pos[1], data.pos[2]);
            this._droneGroup.position.set(tx, ty, tz);
        }

        // Drone orientation
        if (data.roll !== undefined && data.pitch !== undefined && data.yaw !== undefined) {
            // In Three.js after coord swap:
            //   Three X = sim X (roll axis in sim, but pitch visual in three)
            //   Three Y = sim Z (yaw axis)
            //   Three Z = sim Y
            // yaw rotates around Y_three (= up), pitch around X_three, roll around Z_three
            const euler = new THREE.Euler(data.pitch, data.yaw + Math.PI / 2, -data.roll, 'YXZ');
            this._droneGroup.setRotationFromEuler(euler);
        }

        // Target: sim(x,y,z) → three(x, z, y)
        if (data.goal_pos) {
            const [gx, gy, gz] = this._s2t(data.goal_pos[0], data.goal_pos[1], data.goal_pos[2]);
            this._targetMesh.position.set(gx, gy, gz);
            this._targetRing.position.copy(this._targetMesh.position);
            this._targetLight.position.copy(this._targetMesh.position);
        }

        // Animate target
        const t = Date.now() * 0.003;
        this._targetMesh.scale.setScalar(1 + 0.1 * Math.sin(t));
        this._targetRing.rotation.y = t * 0.5;
        this._targetRing.rotation.x = Math.sin(t * 0.3) * 0.3;

        // Trail
        if (data.pos) {
            const [px, py, pz] = this._s2t(data.pos[0], data.pos[1], data.pos[2]);
            
            // Clear trail on level change or sudden jump (teleport/respawn)
            const isRespawn = (this._lastLevel !== undefined && this._lastLevel !== data.level) ||
                              (this._trailPoints.length > 0 && 
                               Math.hypot(px - this._trailPoints[this._trailPoints.length - 1].x, 
                                          py - this._trailPoints[this._trailPoints.length - 1].y, 
                                          pz - this._trailPoints[this._trailPoints.length - 1].z) > 3.0);
            
            if (isRespawn) {
                this._trailPoints = [];
            }
            this._lastLevel = data.level;

            this._addTrailPoint(px, py, pz);
        }

        // Poles (first time)
        if (data.poles && !this._polesSet) {
            this._updatePoles(data.poles);
            this._polesSet = true;
        }
    }

    _addTrailPoint(x, y, z) {
        this._trailPoints.push({ x, y, z });
        if (this._trailPoints.length > this._trailMaxPoints) this._trailPoints.shift();

        const geo = this._trailLine.geometry;
        const pos = geo.attributes.position;
        const col = geo.attributes.color;
        const n = this._trailPoints.length;

        for (let i = 0; i < n; i++) {
            const p = this._trailPoints[i];
            pos.setXYZ(i, p.x, p.y, p.z);
            const a = i / n;
            col.setXYZ(i, 0.376 * a, 0.647 * a, 0.98 * a);
        }
        pos.needsUpdate = true;
        col.needsUpdate = true;
        geo.setDrawRange(0, n);
    }

    _updatePoles(poles) {
        while (this._poleGroup.children.length > 0) this._poleGroup.remove(this._poleGroup.children[0]);

        const poleGeo = new THREE.CylinderGeometry(0.05, 0.05, 2, 8);
        const poleMat = new THREE.MeshPhongMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.5 });

        poles.forEach(p => {
            const mesh = new THREE.Mesh(poleGeo, poleMat);
            // sim(x,y,z) → three(x, z, y), cylinder is along Y (up in Three.js) by default
            const [tx, ty, tz] = this._s2t(p[0], p[1], p[2]);
            mesh.position.set(tx, ty, tz);
            this._poleGroup.add(mesh);
        });
    }

    _handleResize() {
        if (!this.container) return;
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        if (w === 0 || h === 0) return;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        this._needsInitialResize = false;
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        // Auto-resize when the container first becomes visible
        if (this._needsInitialResize && this.container.clientWidth > 0 && this.container.clientHeight > 0) {
            this._handleResize();
        }
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
