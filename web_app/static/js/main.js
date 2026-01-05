// 3D Scene Setup
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

// Add some cool objects
const geometry = new THREE.TorusKnotGeometry(10, 3, 100, 16);
const material = new THREE.MeshNormalMaterial({ wireframe: true });
const torusKnot = new THREE.Mesh(geometry, material);
scene.add(torusKnot);

// Particles
const particlesGeometry = new THREE.BufferGeometry();
const particlesCount = 5000;
const posArray = new Float32Array(particlesCount * 3);

for (let i = 0; i < particlesCount * 3; i++) {
    posArray[i] = (Math.random() - 0.5) * 50;
}

particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
const particlesMaterial = new THREE.PointsMaterial({
    size: 0.02,
    color: 0x00f2ff,
});
const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
scene.add(particlesMesh);

camera.position.z = 30;

// Mouse interaction
let mouseX = 0;
let mouseY = 0;

document.addEventListener('mousemove', (event) => {
    mouseX = event.clientX - window.innerWidth / 2;
    mouseY = event.clientY - window.innerHeight / 2;
});

// Animation Loop
const animate = () => {
    requestAnimationFrame(animate);

    torusKnot.rotation.x += 0.01;
    torusKnot.rotation.y += 0.01;

    // Parallax effect
    particlesMesh.rotation.y = mouseY * 0.0001;
    particlesMesh.rotation.x = mouseX * 0.0001;

    renderer.render(scene, camera);
};

animate();

// Resizing
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------------------------------------------------------
// UI Logic
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultContainer = document.getElementById('result-container');
const resultImage = document.getElementById('result-image');
const resetBtn = document.getElementById('reset-btn');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border)';
    const files = e.dataTransfer.files;
    if (files.length) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    // Show loading state

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                return;
            }
            resultImage.src = 'data:image/jpeg;base64,' + data.image;
            dropZone.style.display = 'none';
            resultContainer.style.display = 'block';
        })
        .catch(err => {
            console.error(err);
            alert('Error analyzing image');
        });
}

resetBtn.addEventListener('click', () => {
    resultContainer.style.display = 'none';
    dropZone.style.display = 'block';
    fileInput.value = '';
});
