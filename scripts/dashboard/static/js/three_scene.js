/**
 * three_scene.js — 3D Navigation Map
 * Three.js scene with drone model, target sphere, trajectory trail,
 * room wireframes, poles, and orbit controls.
 */

class NavigationScene {
    constructor() {
        this.canvas = document.getElementById('canvas-3d');
        this.container = this.canvas.parentElement;

        // Three.js setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e17);
        this.scene.fog = new THREE.FogExp2(0x0a0e17, 0.015);

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            55,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            200
        );
        this.camera.position.set(8, 8, 15);
        this.camera.lookAt(0, -10, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false,
        });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.target.set(0, -10, 1);
        this.controls.minDistance = 3;
        this.controls.maxDistance = 60;

        // Lighting
        const ambient = new THREE.AmbientLight(0x4466aa, 0.6);
        this.scene.add(ambient);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(10, 20, 15);
        this.scene.add(dirLight);

        const pointLight = new THREE.PointLight(0x60a5fa, 0.5, 50);
        pointLight.position.set(0, -10, 5);
        this.scene.add(pointLight);

        // Build scene objects
        this._buildGrid();
        this._buildRooms();
        this._buildDrone();
        this._buildTarget();
        this._buildTrail();
        this._buildPoles();

        // Room bounds from config — stored for reference
        this._roomBoundsSet = false;
        this._polesSet = false;

        // Resize handler
        this._onResize = this._handleResize.bind(this);
        window.addEventListener('resize', this._onResize);

        // Animation loop
        this._animate();
    }

    // ---- Scene Construction ----

    _buildGrid() {
        const grid = new THREE.GridHelper(50, 50, 0x1e3a5f, 0x0f1b2d);
        grid.rotation.x = 0; // XY plane (Y is forward in our coord system)
        this.scene.add(grid);
    }

    _buildRooms() {
        // Default room wireframes (from env config)
        const roomBounds = [
            [-2, 2, -3, 2, 0, 2],
            [-2, 2, -9, -2, 0, 2],
            [-2, 2, -17, -8, 0, 2],
            [-6, 2, -21, -16, 0, 2],
        ];

        const colors = [0x60a5fa, 0xa78bfa, 0x34d399, 0xfbbf24];
        this._roomMeshes = [];

        roomBounds.forEach((bounds, i) => {
            const [xMin, xMax, yMin, yMax, zMin, zMax] = bounds;
            const w = xMax - xMin;
            const d = yMax - yMin;
            const h = zMax - zMin;

            const geo = new THREE.BoxGeometry(w, d, h);
            const edges = new THREE.EdgesGeometry(geo);
            const mat = new THREE.LineBasicMaterial({
                color: colors[i],
                transparent: true,
                opacity: 0.25,
            });
            const wireframe = new THREE.LineSegments(edges, mat);
            wireframe.position.set(
                xMin + w / 2,
                yMin + d / 2,
                zMin + h / 2
            );
            this.scene.add(wireframe);
            this._roomMeshes.push(wireframe);

            // Semi-transparent floor
            const floorGeo = new THREE.PlaneGeometry(w, d);
            const floorMat = new THREE.MeshBasicMaterial({
                color: colors[i],
                transparent: true,
                opacity: 0.04,
                side: THREE.DoubleSide,
            });
            const floor = new THREE.Mesh(floorGeo, floorMat);
            floor.position.set(xMin + w / 2, yMin + d / 2, 0.01);
            this.scene.add(floor);

            // Level label
            const levelLabel = this._createLabel(`L${i + 1}`, colors[i]);
            levelLabel.position.set(xMin + 0.3, yMin + 0.3, zMax + 0.3);
            this.scene.add(levelLabel);
        });
    }

    _createLabel(text, color) {
        const canvas = document.createElement('canvas');
        canvas.width = 64;
        canvas.height = 32;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#' + new THREE.Color(color).getHexString();
        ctx.font = 'bold 20px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 32, 16);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMat = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            opacity: 0.7,
        });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(1, 0.5, 1);
        return sprite;
    }

    _buildDrone() {
        // Quadcopter shape: cross with 4 colored arms
        this._droneGroup = new THREE.Group();

        const armMat = new THREE.MeshPhongMaterial({ color: 0xe2e8f0 });
        const armGeo = new THREE.BoxGeometry(0.6, 0.05, 0.05);

        // Forward/back arms
        const arm1 = new THREE.Mesh(armGeo, armMat);
        this._droneGroup.add(arm1);

        // Left/right arms
        const arm2 = new THREE.Mesh(armGeo, armMat);
        arm2.rotation.z = Math.PI / 2;
        this._droneGroup.add(arm2);

        // Center body
        const bodyGeo = new THREE.SphereGeometry(0.08, 8, 8);
        const bodyMat = new THREE.MeshPhongMaterial({ color: 0x60a5fa, emissive: 0x2563eb, emissiveIntensity: 0.3 });
        const body = new THREE.Mesh(bodyGeo, bodyMat);
        this._droneGroup.add(body);

        // Motor indicators (4 small spheres)
        const motorColors = [0xf87171, 0x34d399, 0xfbbf24, 0xa78bfa];
        const motorPositions = [
            [0.3, 0, 0], [-0.3, 0, 0],
            [0, 0.3, 0], [0, -0.3, 0],
        ];
        motorPositions.forEach((pos, i) => {
            const mGeo = new THREE.SphereGeometry(0.04, 6, 6);
            const mMat = new THREE.MeshPhongMaterial({
                color: motorColors[i],
                emissive: motorColors[i],
                emissiveIntensity: 0.5,
            });
            const motor = new THREE.Mesh(mGeo, mMat);
            motor.position.set(...pos);
            this._droneGroup.add(motor);
        });

        // Direction indicator (small cone pointing forward / -Y)
        const coneGeo = new THREE.ConeGeometry(0.04, 0.12, 4);
        const coneMat = new THREE.MeshPhongMaterial({ color: 0xf87171, emissive: 0xf87171, emissiveIntensity: 0.4 });
        const cone = new THREE.Mesh(coneGeo, coneMat);
        cone.rotation.x = -Math.PI / 2;
        cone.position.set(0, -0.35, 0);
        this._droneGroup.add(cone);

        this._droneGroup.position.set(0, 1.5, 1);
        this.scene.add(this._droneGroup);

        // Glow point light on the drone
        this._droneLight = new THREE.PointLight(0x60a5fa, 0.6, 5);
        this._droneGroup.add(this._droneLight);
    }

    _buildTarget() {
        // Glowing target sphere
        const geo = new THREE.SphereGeometry(0.2, 16, 16);
        const mat = new THREE.MeshPhongMaterial({
            color: 0x34d399,
            emissive: 0x34d399,
            emissiveIntensity: 0.6,
            transparent: true,
            opacity: 0.7,
        });
        this._targetMesh = new THREE.Mesh(geo, mat);
        this._targetMesh.position.set(0, -2.5, 1);
        this.scene.add(this._targetMesh);

        // Outer ring
        const ringGeo = new THREE.RingGeometry(0.3, 0.35, 32);
        const ringMat = new THREE.MeshBasicMaterial({
            color: 0x34d399,
            transparent: true,
            opacity: 0.3,
            side: THREE.DoubleSide,
        });
        this._targetRing = new THREE.Mesh(ringGeo, ringMat);
        this._targetRing.position.copy(this._targetMesh.position);
        this.scene.add(this._targetRing);

        // Target glow
        this._targetLight = new THREE.PointLight(0x34d399, 0.5, 8);
        this._targetLight.position.copy(this._targetMesh.position);
        this.scene.add(this._targetLight);
    }

    _buildTrail() {
        this._trailMaxPoints = 300;
        this._trailPoints = [];

        const geo = new THREE.BufferGeometry();
        const positions = new Float32Array(this._trailMaxPoints * 3);
        const colors = new Float32Array(this._trailMaxPoints * 3);
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geo.setDrawRange(0, 0);

        const mat = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
        });
        this._trailLine = new THREE.Line(geo, mat);
        this.scene.add(this._trailLine);
    }

    _buildPoles() {
        this._poleMeshes = [];
        this._poleGroup = new THREE.Group();
        this.scene.add(this._poleGroup);
    }

    // ---- Update Methods ----

    update(data) {
        if (!data) return;

        // Update drone position
        if (data.pos) {
            this._droneGroup.position.set(data.pos[0], data.pos[1], data.pos[2]);
        }

        // Update drone orientation
        if (data.roll !== undefined && data.pitch !== undefined && data.yaw !== undefined) {
            // Convert euler angles to Three.js rotation
            // In our coord system: X=right, Y=forward, Z=up
            this._droneGroup.rotation.set(data.pitch, 0, data.roll);
            this._droneGroup.rotation.order = 'ZXY';
            // Apply yaw around Z axis
            this._droneGroup.rotation.z = -data.roll; // roll around forward axis
            this._droneGroup.rotation.x = -data.pitch; // pitch
            this._droneGroup.rotation.y = 0;

            // Build rotation from yaw (around Z), pitch (around X), roll (around Y)
            const euler = new THREE.Euler(data.pitch, 0, data.yaw + Math.PI / 2, 'ZYX');
            this._droneGroup.setRotationFromEuler(euler);
        }

        // Update target
        if (data.goal_pos) {
            this._targetMesh.position.set(data.goal_pos[0], data.goal_pos[1], data.goal_pos[2]);
            this._targetRing.position.copy(this._targetMesh.position);
            this._targetLight.position.copy(this._targetMesh.position);
        }

        // Animate target
        const t = Date.now() * 0.003;
        this._targetMesh.scale.setScalar(1 + 0.1 * Math.sin(t));
        this._targetRing.rotation.z = t * 0.5;
        this._targetRing.rotation.x = Math.sin(t * 0.3) * 0.3;

        // Update trail
        if (data.pos) {
            this._addTrailPoint(data.pos[0], data.pos[1], data.pos[2]);
        }

        // Update poles (first time only)
        if (data.poles && !this._polesSet) {
            this._updatePoles(data.poles);
            this._polesSet = true;
        }
    }

    _addTrailPoint(x, y, z) {
        this._trailPoints.push({ x, y, z });
        if (this._trailPoints.length > this._trailMaxPoints) {
            this._trailPoints.shift();
        }

        const geo = this._trailLine.geometry;
        const pos = geo.attributes.position;
        const col = geo.attributes.color;
        const n = this._trailPoints.length;

        for (let i = 0; i < n; i++) {
            const p = this._trailPoints[i];
            pos.setXYZ(i, p.x, p.y, p.z);

            // Fade from transparent (old) to bright (new)
            const alpha = i / n;
            col.setXYZ(i, 0.376 * alpha, 0.647 * alpha, 0.98 * alpha); // accent-primary fade
        }

        pos.needsUpdate = true;
        col.needsUpdate = true;
        geo.setDrawRange(0, n);
    }

    _updatePoles(poles) {
        // Remove old poles
        while (this._poleGroup.children.length > 0) {
            this._poleGroup.remove(this._poleGroup.children[0]);
        }

        const poleGeo = new THREE.CylinderGeometry(0.05, 0.05, 2, 8);
        const poleMat = new THREE.MeshPhongMaterial({
            color: 0x94a3b8,
            transparent: true,
            opacity: 0.5,
        });

        poles.forEach(pole => {
            const mesh = new THREE.Mesh(poleGeo, poleMat);
            // Cylinder is created along Y-axis, we need it along Z
            mesh.rotation.x = Math.PI / 2;
            mesh.position.set(pole[0], pole[1], pole[2]);
            this._poleGroup.add(mesh);
        });
    }

    _handleResize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
