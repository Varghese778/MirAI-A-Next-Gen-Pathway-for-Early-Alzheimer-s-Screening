/**
 * MirAI Stage-2 Module
 * Genetic risk assessment form handling
 */

let assessmentId = null;

document.addEventListener('DOMContentLoaded', async () => {
    // Check authentication - validate with server
    if (!window.API.session.isValid()) {
        window.location.href = 'index.html';
        return;
    }

    // Validate session with server
    try {
        const valid = await window.API.auth.validate();
        if (!valid) {
            alert('Your session has expired. Please log in again.');
            window.API.session.clear();
            window.location.href = 'index.html';
            return;
        }
    } catch (e) {
        console.error('Session validation error:', e);
        window.API.session.clear();
        window.location.href = 'index.html';
        return;
    }

    // Get assessment ID from session
    assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (!assessmentId) {
        alert('No assessment found. Please complete Stage-1 first.');
        window.location.href = 'dashboard.html';
        return;
    }

    // Load user info
    loadUserInfo();

    // Form submission handler
    document.getElementById('stage2Form').addEventListener('submit', handleSubmit);

    // Radio button selection styling
    document.querySelectorAll('.radio-option').forEach(option => {
        option.addEventListener('click', () => {
            document.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
            option.classList.add('selected');
            option.querySelector('input').checked = true;
        });
    });
});

function loadUserInfo() {
    const username = window.API.session.username || 'User';
    document.getElementById('userAvatar').textContent = username.charAt(0).toUpperCase();
}

async function handleSubmit(e) {
    e.preventDefault();

    const btn = document.getElementById('submitBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        // Get form values
        const apoeValue = document.querySelector('input[name="apoe_e4_count"]:checked');
        if (!apoeValue) {
            alert('Please select your APOE ε4 genotype.');
            btn.classList.remove('loading');
            btn.disabled = false;
            return;
        }

        const apoe_e4_count = parseInt(apoeValue.value);
        const prsInput = document.getElementById('prs_input').value;
        const polygenic_risk_score = prsInput ? parseFloat(prsInput) : null;

        // Validate PRS range
        if (polygenic_risk_score !== null && (polygenic_risk_score < 0 || polygenic_risk_score > 1)) {
            alert('Polygenic Risk Score must be between 0 and 1.');
            btn.classList.remove('loading');
            btn.disabled = false;
            return;
        }

        // Submit to API
        const result = await window.API.stage2.submit(assessmentId, {
            apoe_e4_count,
            polygenic_risk_score,
            consent_given: true
        });

        console.log('Stage-2 result:', result);

        // Store result and navigate to combined results
        sessionStorage.setItem('mirai_stage2_complete', 'true');
        window.location.href = 'combined-results.html';

    } catch (error) {
        console.error('Stage-2 submission error:', error);
        alert('Failed to submit genetic data: ' + error.message);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}
