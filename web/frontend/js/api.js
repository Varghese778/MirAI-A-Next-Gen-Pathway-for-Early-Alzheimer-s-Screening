/**
 * MirAI API Client
 * Handles all API communication with the backend
 */

// Auto-detect environment: use localhost for dev, render URL for production
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000/api'
    : 'https://mirai-backend.onrender.com/api';

// Session management
const session = {
    token: localStorage.getItem('mirai_token'),
    userId: localStorage.getItem('mirai_user_id'),
    username: localStorage.getItem('mirai_username'),

    save(data) {
        this.token = data.token;
        this.userId = data.user_id;
        this.username = data.username;
        localStorage.setItem('mirai_token', data.token);
        localStorage.setItem('mirai_user_id', data.user_id);
        localStorage.setItem('mirai_username', data.username);
    },

    clear() {
        this.token = null;
        this.userId = null;
        this.username = null;
        localStorage.removeItem('mirai_token');
        localStorage.removeItem('mirai_user_id');
        localStorage.removeItem('mirai_username');
    },

    isValid() {
        return !!this.token;
    }
};

// API request helper
async function apiRequest(endpoint, options = {}) {
    const url = `${API_BASE}${endpoint}`;

    const config = {
        headers: {
            'Content-Type': 'application/json',
            ...options.headers
        },
        ...options
    };

    try {
        const response = await fetch(url, config);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Request failed');
        }

        return data;
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Auth API
const authAPI = {
    async register(username, password) {
        const data = await apiRequest('/auth/register', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
        if (data.success) {
            session.save(data);
        }
        return data;
    },

    async login(username, password) {
        const data = await apiRequest('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
        if (data.success) {
            session.save(data);
        }
        return data;
    },

    async logout() {
        if (session.token) {
            try {
                await apiRequest(`/auth/logout?token=${session.token}`, {
                    method: 'POST'
                });
            } catch (e) {
                // Ignore logout errors
            }
        }
        session.clear();
    },

    async validate() {
        if (!session.token) return false;
        try {
            const data = await apiRequest(`/auth/validate?token=${session.token}`);
            return data.valid;
        } catch (e) {
            return false;
        }
    }
};

// Questionnaire API
const questionnaireAPI = {
    async getQuestions() {
        return await apiRequest('/questionnaire/questions');
    },

    async startAssessment() {
        return await apiRequest('/questionnaire/start', {
            method: 'POST',
            body: JSON.stringify({ token: session.token })
        });
    },

    async submitAnswers(assessmentId, answers) {
        return await apiRequest('/questionnaire/submit', {
            method: 'POST',
            body: JSON.stringify({
                token: session.token,
                assessment_id: assessmentId,
                answers: answers
            })
        });
    },

    async getResult(assessmentId) {
        return await apiRequest(`/questionnaire/result/${assessmentId}?token=${session.token}`);
    },

    async getHistory() {
        return await apiRequest(`/questionnaire/history?token=${session.token}`);
    }
};

// Stage-2 API
const stage2API = {
    async getStatus() {
        return await apiRequest('/stage2/status');
    },

    async submit(assessmentId, geneticData) {
        return await apiRequest('/stage2/submit', {
            method: 'POST',
            body: JSON.stringify({
                token: session.token,
                assessment_id: parseInt(assessmentId),
                apoe_e4_count: geneticData.apoe_e4_count,
                polygenic_risk_score: geneticData.polygenic_risk_score,
                consent_given: geneticData.consent_given
            })
        });
    },

    async getResult(assessmentId) {
        return await apiRequest(`/stage2/result/${assessmentId}?token=${session.token}`);
    },

    async getCombined(assessmentId) {
        return await apiRequest(`/stage2/combined/${assessmentId}?token=${session.token}`);
    }
};

// Stage-3 API
const stage3API = {
    async getStatus() {
        return await apiRequest('/stage3/status');
    },

    async submit(assessmentId, biomarkerData) {
        return await apiRequest('/stage3/submit', {
            method: 'POST',
            body: JSON.stringify({
                token: session.token,
                assessment_id: parseInt(assessmentId),
                ptau217: biomarkerData.ptau217,
                ptau217_abeta42_ratio: biomarkerData.ptau217_abeta42_ratio,
                nfl: biomarkerData.nfl,
                consent_given: biomarkerData.consent_given
            })
        });
    },

    async getResult(assessmentId) {
        return await apiRequest(`/stage3/result/${assessmentId}?token=${session.token}`);
    },

    async getFinal(assessmentId) {
        return await apiRequest(`/stage3/final/${assessmentId}?token=${session.token}`);
    }
};

// Export for use in other modules
window.API = {
    session,
    auth: authAPI,
    questionnaire: questionnaireAPI,
    stage2: stage2API,
    stage3: stage3API
};
