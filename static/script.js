document.addEventListener('DOMContentLoaded', () => {
    // Particles.js initialization
    particlesJS('particles-js', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: '#2b9348' },
            shape: { type: 'circle' },
            opacity: { value: 0.5, random: true },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: '#40916c', opacity: 0.4, width: 1 },
            move: { enable: true, speed: 2, direction: 'none', random: true, out_mode: 'out' }
        },
        interactivity: {
            detect_on: 'canvas',
            events: { onhover: { enable: true, mode: 'repulse' }, onclick: { enable: true, mode: 'push' }, resize: true },
            modes: { repulse: { distance: 100, duration: 0.4 }, push: { particles_nb: 4 } }
        },
        retina_detect: true
    });

    // Ripple effect for buttons
    document.querySelectorAll('.ripple').forEach(btn => {
        btn.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-circle');
            const x = e.clientX - this.getBoundingClientRect().left;
            const y = e.clientY - this.getBoundingClientRect().top;
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            this.appendChild(ripple);
            setTimeout(() => ripple.remove(), 700);
        });
    });

    // Glitch effect data-text attribute
    document.querySelectorAll('.glitch').forEach(el => {
        el.setAttribute('data-text', el.textContent);
    });

    // Smooth scroll for nav links (handled by Flask routing)
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const href = link.getAttribute('href');
            window.location.href = href;
        });
    });

    // Floating plant follows cursor in hero section
    const hero = document.querySelector('.hero');
    const plant = document.querySelector('.floating-plant');
    if (hero && plant) {
        hero.addEventListener('mousemove', (e) => {
            const rect = hero.getBoundingClientRect();
            const x = e.clientX - rect.left - 20;
            const y = e.clientY - rect.top - 20;
            plant.style.transform = `translate(${x}px, ${y}px)`;
        });
        hero.addEventListener('mouseleave', () => {
            plant.style.transform = 'translate(0, 0)';
        });
    }

    // Drag-and-drop and image preview for predict page
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('file-upload');
    const previewImg = document.getElementById('previewImg');
    const imagePreview = document.getElementById('imagePreview');

    if (dropZone && fileInput && previewImg && imagePreview) {
        // Drag events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                previewImage(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                previewImage(fileInput.files[0]);
            }
        });

        // Preview image function
        function previewImage(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    }
});

// Ripple circle style
const style = document.createElement('style');
style.textContent = `
    .ripple-circle {
        position: absolute;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 50%;
        width: 0;
        height: 0;
        pointer-events: none;
        animation: rippleEffect 0.7s linear;
        transform: translate(-50%, -50%);
    }
`;
document.head.appendChild(style);