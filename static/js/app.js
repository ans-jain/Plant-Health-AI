document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');
    const dropzonePrompt = document.getElementById('dropzonePrompt');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const btnRemove = document.getElementById('btnRemove');
    const btnDiagnose = document.getElementById('btnDiagnose');
    const btnText = btnDiagnose.querySelector('.btn-text');
    const spinner = btnDiagnose.querySelector('.spinner-small');
    
    const emptyState = document.getElementById('emptyState');
    const diagnosisContent = document.getElementById('diagnosisContent');
    const statusBanner = document.getElementById('statusBanner');
    const statusIcon = document.getElementById('statusIcon');
    const healthStatus = document.getElementById('healthStatus');
    const specificDiagnosis = document.getElementById('specificDiagnosis');
    const confidenceScore = document.getElementById('confidenceScore');
    const originalView = document.getElementById('originalView');
    const gradcamView = document.getElementById('gradcamView');
    const pathologyCause = document.getElementById('pathologyCause');
    const treatmentList = document.getElementById('treatmentList');
    const organicCare = document.getElementById('organicCare');

    let currentFile = null;

    // File selection handler
    function handleFile(file) {
        if (!file || !file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG or PNG).');
            return;
        }
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropzonePrompt.style.display = 'none';
            previewContainer.style.display = 'block';
            btnDiagnose.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag & Drop handlers
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
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // Remove file
    btnRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        dropzonePrompt.style.display = 'block';
        previewContainer.style.display = 'none';
        btnDiagnose.disabled = true;
    });

    // Run Diagnosis
    btnDiagnose.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI Loading state
        btnDiagnose.disabled = true;
        btnText.textContent = 'Analyzing Leaf Sample...';
        spinner.style.display = 'block';

        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/api/classify', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.status === 'success') {
                renderDiagnosis(data);
            } else {
                alert('Diagnostic Error: ' + (data.message || 'Unknown error occurred.'));
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Server communication error: ' + error.message);
        } finally {
            btnDiagnose.disabled = false;
            btnText.textContent = 'Diagnose Leaf Health';
            spinner.style.display = 'none';
        }
    });

    function renderDiagnosis(data) {
        emptyState.style.display = 'none';
        diagnosisContent.style.display = 'block';

        const isHealthy = data.health_status === 'Healthy';
        
        // Status banner styling
        if (isHealthy) {
            statusBanner.className = 'status-banner';
            statusIcon.textContent = '✓';
            healthStatus.textContent = 'Healthy Leaf';
            healthStatus.style.color = '#34d399';
        } else {
            statusBanner.className = 'status-banner diseased';
            statusIcon.textContent = '⚠';
            healthStatus.textContent = 'Disease Detected';
            healthStatus.style.color = '#f87171';
        }

        specificDiagnosis.textContent = data.predicted_class.replace(/_/g, ' ');
        confidenceScore.textContent = (data.confidence * 100).toFixed(1) + '%';

        // Images
        originalView.src = data.original_image_base64;
        gradcamView.src = data.gradcam_base64;

        // Pathology & Advice
        const details = data.details || {};
        pathologyCause.textContent = details.cause || 'No specific disease details available.';
        
        treatmentList.innerHTML = '';
        const treatments = details.treatments || [];
        if (treatments.length === 0) {
            const li = document.createElement('li');
            li.textContent = 'No treatment necessary. Plant is in prime health.';
            treatmentList.appendChild(li);
        } else {
            treatments.forEach(t => {
                const li = document.createElement('li');
                li.textContent = t;
                treatmentList.appendChild(li);
            });
        }

        organicCare.textContent = details.organic_care || 'Maintain regular inspection and proper watering.';
    }
});
