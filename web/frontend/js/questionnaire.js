/**
 * MirAI Questionnaire Module
 * Multi-step questionnaire wizard
 */

let questions = [];
let currentIndex = 0;
let answers = {};
let assessmentId = null;

document.addEventListener('DOMContentLoaded', async () => {
    // Check authentication and assessment
    if (!window.API.session.isValid()) {
        window.location.href = 'index.html';
        return;
    }

    assessmentId = sessionStorage.getItem('mirai_assessment_id');
    if (!assessmentId) {
        window.location.href = 'dashboard.html';
        return;
    }

    // Load questions
    await loadQuestions();

    // Button handlers
    document.getElementById('prevBtn').addEventListener('click', prevQuestion);
    document.getElementById('nextBtn').addEventListener('click', nextQuestion);

    // Prevent Enter key from submitting/advancing
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
        }
    });
});

async function loadQuestions() {
    try {
        const result = await window.API.questionnaire.getQuestions();

        // Flatten sections into individual questions
        questions = [];
        if (result.sections) {
            result.sections.forEach(section => {
                section.questions.forEach(q => {
                    questions.push({
                        ...q,
                        section: section.section
                    });
                });
            });
        }

        if (questions.length === 0) {
            alert('Failed to load questions');
            window.location.href = 'dashboard.html';
            return;
        }

        renderQuestion();
    } catch (error) {
        console.error('Failed to load questions:', error);
        alert('Failed to load questions. Please try again.');
        window.location.href = 'dashboard.html';
    }
}

function renderQuestion() {
    const question = questions[currentIndex];
    const container = document.getElementById('answerContainer');

    // Update section title
    document.getElementById('sectionTitle').textContent = question.section;

    // Update question text
    document.getElementById('questionText').textContent = question.text;

    // Update progress
    updateProgress();

    // Get previously stored answer for this question
    const savedAnswer = answers[question.key];

    // Render answer options based on question type
    let html = '';

    if (question.type === 'number') {
        const currentValue = savedAnswer !== undefined ? savedAnswer : '';
        html = `
            <div class="number-input">
                <input type="number" id="numberInput" 
                       value="${currentValue}"
                       min="${question.min || 0}" 
                       max="${question.max || 999}"
                       placeholder="Enter value">
                <span class="number-unit">${question.unit || ''}</span>
            </div>
        `;
    } else {
        // Multiple choice
        html = '<div class="answer-options">';
        question.options.forEach((option, idx) => {
            // Compare as numbers since we store answers as numbers
            const optionVal = parseFloat(option.value);
            const isSelected = savedAnswer !== undefined && savedAnswer === optionVal;
            html += `
                <label class="answer-option ${isSelected ? 'selected' : ''}" data-value="${option.value}">
                    <input type="radio" name="answer" value="${option.value}" ${isSelected ? 'checked' : ''}>
                    <span class="answer-radio"></span>
                    <span class="answer-label">${option.label}</span>
                </label>
            `;
        });
        html += '</div>';
    }

    container.innerHTML = html;

    // Add event listeners
    if (question.type === 'number') {
        const input = document.getElementById('numberInput');
        const minVal = question.min !== undefined ? question.min : 1;
        const maxVal = question.max !== undefined ? question.max : 100;

        // Store value on input (no clamping during typing to allow editing)
        input.addEventListener('input', (e) => {
            let val = e.target.value;
            if (val !== '') {
                answers[question.key] = parseFloat(val);
            } else {
                answers[question.key] = null;
            }
        });

        // Validate and clamp only on blur (when leaving the field)
        input.addEventListener('blur', (e) => {
            let val = e.target.value;
            if (val !== '') {
                let numVal = parseFloat(val);
                // Only clamp if outside valid range
                if (numVal < minVal) {
                    numVal = minVal;
                    e.target.value = numVal;
                }
                if (numVal > maxVal) {
                    numVal = maxVal;
                    e.target.value = numVal;
                }
                answers[question.key] = numVal;
            }
        });

        // Prevent Enter from advancing
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
            }
        });
        input.focus();
    } else {
        container.querySelectorAll('.answer-option').forEach(option => {
            option.addEventListener('click', () => {
                // Remove selection from all
                container.querySelectorAll('.answer-option').forEach(o => o.classList.remove('selected'));
                // Select clicked
                option.classList.add('selected');
                option.querySelector('input').checked = true;
                // Store answer
                answers[question.key] = parseFloat(option.dataset.value);
            });
        });
    }

    // Update button states
    document.getElementById('prevBtn').disabled = currentIndex === 0;

    const nextBtn = document.getElementById('nextBtn');
    if (currentIndex === questions.length - 1) {
        nextBtn.querySelector('span').textContent = 'Submit';
    } else {
        nextBtn.querySelector('span').textContent = 'Next';
    }
}

function updateProgress() {
    const progress = ((currentIndex + 1) / questions.length) * 100;
    document.getElementById('progressFill').style.width = `${progress}%`;
    document.getElementById('progressText').textContent = `Question ${currentIndex + 1} of ${questions.length}`;
    document.getElementById('progressPercent').textContent = `${Math.round(progress)}%`;
}

function prevQuestion() {
    if (currentIndex > 0) {
        currentIndex--;
        renderQuestion();
    }
}

async function nextQuestion() {
    const question = questions[currentIndex];

    // Validate current answer
    if (answers[question.key] === undefined || answers[question.key] === null) {
        alert('Please answer the question before proceeding.');
        return;
    }

    if (currentIndex < questions.length - 1) {
        currentIndex++;
        renderQuestion();
    } else {
        // Submit assessment
        await submitAssessment();
    }
}

async function submitAssessment() {
    const nextBtn = document.getElementById('nextBtn');
    nextBtn.classList.add('loading');
    nextBtn.disabled = true;

    try {
        await window.API.questionnaire.submitAnswers(assessmentId, answers);
        window.location.href = 'results.html';
    } catch (error) {
        console.error('Submit error:', error);
        alert('Failed to submit assessment. Please try again.');
        nextBtn.classList.remove('loading');
        nextBtn.disabled = false;
    }
}

// ========================================
// DEMO PANEL FUNCTIONS (For Testing)
// ========================================

function toggleDemoPanel() {
    const content = document.getElementById('demoContent');
    content.classList.toggle('show');
}

// Demo data presets for different risk levels
const DEMO_PRESETS = {
    high: {
        age: 78,
        sex: 1,
        education: 0,
        employment: 1,
        family_history_dementia: 1,
        family_history_relationship: 0,
        judgment_problems: 1,
        conversation_difficulty: 1,
        learning_difficulty: 1,
        appointment_memory: 1,
        daily_activities_difficulty: 1,
        adl_assistance: 1,
        navigation_difficulty: 1,
        diabetes: 1,
        hypertension: 1,
        physical_activity: 0,
        sleep_hours: 4,
        depression: 1,
        head_injury: 1
    },
    moderate: {
        age: 65,
        sex: 1,
        education: 1,
        employment: 1,
        family_history_dementia: 1,
        family_history_relationship: 1,
        judgment_problems: 0,
        conversation_difficulty: 1,
        learning_difficulty: 0,
        appointment_memory: 1,
        daily_activities_difficulty: 0,
        adl_assistance: 0,
        navigation_difficulty: 0,
        diabetes: 1,
        hypertension: 0,
        physical_activity: 1,
        sleep_hours: 6,
        depression: 0,
        head_injury: 0
    },
    low: {
        age: 55,
        sex: 0,
        education: 2,
        employment: 0,
        family_history_dementia: 0,
        judgment_problems: 0,
        conversation_difficulty: 0,
        learning_difficulty: 0,
        appointment_memory: 0,
        daily_activities_difficulty: 0,
        adl_assistance: 0,
        navigation_difficulty: 0,
        diabetes: 0,
        hypertension: 0,
        physical_activity: 2,
        sleep_hours: 8,
        depression: 0,
        head_injury: 0
    }
};

function fillDemoData(riskLevel) {
    const preset = DEMO_PRESETS[riskLevel];
    if (!preset) return;

    // Fill answers with preset data
    answers = { ...preset };

    // Jump to last question
    currentIndex = questions.length - 1;
    renderQuestion();

    // Close demo panel
    document.getElementById('demoContent').classList.remove('show');

    // Show confirmation
    alert(`✅ Filled ${riskLevel.toUpperCase()} risk data. Click Submit to complete.`);
}
