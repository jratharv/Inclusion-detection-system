// Inclusion Detection System JavaScript

class InclusionDetectionApp {
    constructor() {
        this.socket = io();
        this.signalData = [];
        this.confidenceData = [];
        this.timestamps = [];
        this.maxDataPoints = 100;
        
        this.initializeEventListeners();
        this.initializeCharts();
        this.connectWebSocket();
        this.updateStatus();
    }
    
    initializeEventListeners() {
        // Button event listeners
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModel());
        document.getElementById('startBtn').addEventListener('click', () => this.startDetection());
        document.getElementById('stopBtn').addEventListener('click', () => this.stopDetection());
        document.getElementById('clearLogBtn').addEventListener('click', () => this.clearLog());
    }
    
    initializeCharts() {
        // Initialize signal chart
        const signalLayout = {
            title: 'Sensor Signal Value',
            xaxis: { title: 'Time Steps' },
            yaxis: { title: 'Signal Value' },
            margin: { t: 30, r: 30, b: 50, l: 50 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        
        Plotly.newPlot('signalChart', [{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            name: 'Signal',
            line: { color: '#74b9ff' }
        }], signalLayout, { responsive: true });
        
        // Initialize confidence chart
        const confidenceLayout = {
            title: 'Detection Confidence',
            xaxis: { title: 'Time Steps' },
            yaxis: { title: 'Confidence', range: [0, 1] },
            margin: { t: 30, r: 30, b: 50, l: 50 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            shapes: [{
                type: 'line',
                x0: 0, x1: 1, xref: 'paper',
                y0: 0.5, y1: 0.5,
                line: { color: 'red', width: 2, dash: 'dash' }
            }]
        };
        
        Plotly.newPlot('confidenceChart', [{
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            name: 'Confidence',
            line: { color: '#00b894' }
        }], confidenceLayout, { responsive: true });
    }
    
    connectWebSocket() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.addLogEntry('Connected to Inclusion Detection System', 'info');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.addLogEntry('Disconnected from server', 'warning');
        });
        
        this.socket.on('sensor_data', (data) => {
            this.updateRealtimeData(data);
        });
        
        this.socket.on('training_status', (data) => {
            this.updateTrainingProgress(data);
        });
    }
    
    async trainModel() {
        try {
            const response = await fetch('/api/train', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'Training started') {
                document.getElementById('trainBtn').disabled = true;
                document.getElementById('trainingProgress').style.display = 'block';
                this.addLogEntry('Model training started...', 'info');
            } else {
                this.addLogEntry(data.status, 'warning');
            }
        } catch (error) {
            console.error('Training error:', error);
            this.addLogEntry('Training failed: ' + error.message, 'error');
        }
    }
    
    async startDetection() {
        try {
            const response = await fetch('/api/start', { method: 'POST' });
            const data = await response.json();
            
            if (data.status === 'Detection started') {
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                this.addLogEntry('Real-time detection started', 'success');
                this.updateStatus();
            } else {
                this.addLogEntry(data.status, 'warning');
            }
        } catch (error) {
            console.error('Start detection error:', error);
            this.addLogEntry('Failed to start detection: ' + error.message, 'error');
        }
    }
    
    async stopDetection() {
        try {
            const response = await fetch('/api/stop', { method: 'POST' });
            const data = await response.json();
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            this.addLogEntry('Detection stopped', 'info');
            this.updateStatus();
        } catch (error) {
            console.error('Stop detection error:', error);
            this.addLogEntry('Failed to stop detection: ' + error.message, 'error');
        }
    }
    
    updateTrainingProgress(data) {
        const progressBar = document.querySelector('.progress-bar');
        const statusText = document.getElementById('trainingStatus');
        
        progressBar.style.width = data.progress + '%';
        progressBar.textContent = data.progress + '%';
        statusText.textContent = data.status;
        
        if (data.progress === 100) {
            document.getElementById('trainBtn').disabled = false;
            document.getElementById('startBtn').disabled = false;
            document.getElementById('trainingProgress').style.display = 'none';
            
            this.addLogEntry(`Training completed! Accuracy: ${(data.accuracy * 100).toFixed(2)}%`, 'success');
            this.updateStatus();
        }
    }
    
    updateRealtimeData(data) {
        // Update data arrays
        this.signalData.push(data.signal_value);
        this.confidenceData.push(data.confidence);
        this.timestamps.push(new Date(data.timestamp));
        
        // Keep only latest data points
        if (this.signalData.length > this.maxDataPoints) {
            this.signalData = this.signalData.slice(-this.maxDataPoints);
            this.confidenceData = this.confidenceData.slice(-this.maxDataPoints);
            this.timestamps = this.timestamps.slice(-this.maxDataPoints);
        }
        
        // Update charts
        const timeIndices = Array.from({length: this.signalData.length}, (_, i) => i);
        
        Plotly.restyle('signalChart', {
            x: [timeIndices],
            y: [this.signalData]
        });
        
        Plotly.restyle('confidenceChart', {
            x: [timeIndices],
            y: [this.confidenceData]
        });
        
        // Update status display
        document.getElementById('currentReading').textContent = data.status;
        document.getElementById('currentReading').className = `badge ${data.status === 'Normal' ? 'bg-success' : 'bg-danger'}`;
        
        document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
        
        // Show alert if inclusion detected
        if (data.alert) {
            this.showInclusionAlert(data);
        }
        
        // Add to log
        const logType = data.status === 'Normal' ? 'normal' : 'alert';
        this.addLogEntry(`${data.status} - Confidence: ${(data.confidence * 100).toFixed(1)}%`, logType);
    }
    
    showInclusionAlert(data) {
        const alertPanel = document.getElementById('alertPanel');
        const alertMessage = document.getElementById('alertMessage');
        
        alertMessage.textContent = `Detected at ${new Date(data.timestamp).toLocaleTimeString()} with ${(data.confidence * 100).toFixed(1)}% confidence`;
        alertPanel.style.display = 'block';
        
        // Auto-hide alert after 10 seconds
        setTimeout(() => {
            alertPanel.style.display = 'none';
        }, 10000);
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();
            
            // Update model status
            const modelStatusEl = document.getElementById('modelStatus');
            modelStatusEl.textContent = status.model_trained ? 'Trained' : 'Not Trained';
            modelStatusEl.className = `badge ${status.model_trained ? 'bg-success' : 'bg-secondary'}`;
            
            // Update detection status
            const detectionStatusEl = document.getElementById('detectionStatus');
            detectionStatusEl.textContent = status.is_running ? 'Running' : 'Stopped';
            detectionStatusEl.className = `badge ${status.is_running ? 'bg-success' : 'bg-secondary'}`;
            
            // Update button states
            document.getElementById('startBtn').disabled = !status.model_trained || status.is_running;
            document.getElementById('stopBtn').disabled = !status.is_running;
            
        } catch (error) {
            console.error('Status update error:', error);
        }
    }
    
    addLogEntry(message, type) {
        const logContainer = document.getElementById('detectionLog');
        const timestamp = new Date().toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.innerHTML = `
            <span class="timestamp">${timestamp}</span>
            <span class="message">${message}</span>
        `;
        
        logContainer.appendChild(logEntry);
        logContainer.scrollTop = logContainer.scrollHeight;
        
        // Keep only latest 100 log entries
        const entries = logContainer.children;
        if (entries.length > 100) {
            logContainer.removeChild(entries[0]);
        }
    }
    
    clearLog() {
        document.getElementById('detectionLog').innerHTML = '';
        this.addLogEntry('Log cleared', 'info');
    }
}

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new InclusionDetectionApp();
});