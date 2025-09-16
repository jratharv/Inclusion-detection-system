from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import threading
import time
from datetime import datetime
from collections import deque
import json

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'inclusion_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class InclusionDetectionSystem:
    def __init__(self):
        self.model = None
        self.sequence_length = 50
        self.feature_dim = 3
        self.is_running = False
        self.data_buffer = deque(maxlen=self.sequence_length)
        self.predictions = deque(maxlen=100)
        self.timestamps = deque(maxlen=100)
        self.signal_data = deque(maxlen=100)
        self.inclusion_flags = deque(maxlen=100)
        self.current_status = "Normal"
        self.current_confidence = 0.0
        self.model_trained = False
        
    def generate_synthetic_training_data(self, num_samples=5000):
        """Generate synthetic sensor data with inclusion patterns"""
        X = []
        y = []
        
        for i in range(num_samples):
            sequence = np.zeros((self.sequence_length, self.feature_dim))
            has_inclusion = np.random.random() < 0.3
            
            for t in range(self.sequence_length):
                sequence[t, 0] = np.sin(t * 0.1) + np.random.normal(0, 0.1)
                sequence[t, 1] = np.cos(t * 0.15) + np.random.normal(0, 0.1)
                sequence[t, 2] = np.random.normal(0, 0.1)
                
                if has_inclusion and t > self.sequence_length - 15:
                    inclusion_intensity = 1.5 + np.random.random()
                    sequence[t, 0] += inclusion_intensity * np.sin(t * 0.5)
                    sequence[t, 1] += inclusion_intensity * np.random.normal(0, 0.3)
                    sequence[t, 2] += inclusion_intensity * np.random.normal(0, 0.5)
            
            X.append(sequence)
            y.append(1 if has_inclusion else 0)
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, X, y=None):
        """Normalize the data for LSTM input"""
        X_normalized = np.tanh(X)
        if y is not None:
            return X_normalized, y
        return X_normalized
    
    def build_lstm_model(self):
        """Build and compile the LSTM model"""
        model = keras.Sequential([
            keras.Input(shape=(self.sequence_length, self.feature_dim)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, epochs=15):
        """Train the LSTM model with synthetic data"""
        print("Training LSTM model...")
        socketio.emit('training_status', {'status': 'Training started...', 'progress': 0})
        
        X_train, y_train = self.generate_synthetic_training_data(4000)
        X_val, y_val = self.generate_synthetic_training_data(1000)
        
        X_train_norm, y_train = self.preprocess_data(X_train, y_train)
        X_val_norm, y_val = self.preprocess_data(X_val, y_val)
        
        if self.model is None:
            self.build_lstm_model()
        
        # Custom callback to emit progress
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, socketio_instance):
                self.socketio = socketio_instance
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int((epoch + 1) / epochs * 100)
                self.socketio.emit('training_status', {
                    'status': f'Epoch {epoch + 1}/{epochs} completed',
                    'progress': progress,
                    'accuracy': float(logs.get('val_accuracy', 0))
                })
        
        history = self.model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=0,
            callbacks=[ProgressCallback(socketio)]
        )
        
        self.model_trained = True
        final_accuracy = history.history['val_accuracy'][-1]
        socketio.emit('training_status', {
            'status': 'Training completed!',
            'progress': 100,
            'accuracy': float(final_accuracy)
        })
        
        print(f"Training completed! Final validation accuracy: {final_accuracy:.4f}")
        return history
    
    def generate_realtime_data(self):
        """Generate real-time sensor data stream"""
        t = 0
        while self.is_running:
            has_inclusion = np.random.random() < 0.05
            
            data_point = np.zeros(self.feature_dim)
            data_point[0] = np.sin(t * 0.1) + np.random.normal(0, 0.1)
            data_point[1] = np.cos(t * 0.15) + np.random.normal(0, 0.1)
            data_point[2] = np.random.normal(0, 0.1)
            
            if has_inclusion:
                intensity = 2.0 + np.random.random()
                data_point[0] += intensity * np.sin(t * 0.5)
                data_point[1] += intensity * np.random.normal(0, 0.3)
                data_point[2] += intensity * np.random.normal(0, 0.5)
            
            self.data_buffer.append(data_point)
            self.signal_data.append(data_point[0])
            
            t += 1
            time.sleep(0.2)  # 5Hz data rate for web interface
    
    def predict_inclusion(self, sequence):
        """Predict inclusion for a given sequence"""
        if len(sequence) < self.sequence_length or not self.model_trained:
            return 0.0, "Insufficient data"
        
        input_sequence = np.array(sequence[-self.sequence_length:]).reshape(1, self.sequence_length, self.feature_dim)
        input_sequence_norm = self.preprocess_data(input_sequence)
        
        prediction = self.model.predict(input_sequence_norm, verbose=0)[0][0]
        
        if prediction > 0.5:
            return prediction, "Inclusion Detected"
        else:
            return prediction, "Normal"
    
    def real_time_detection_loop(self):
        """Main real-time detection loop"""
        while self.is_running:
            if len(self.data_buffer) >= self.sequence_length and self.model_trained:
                confidence, status = self.predict_inclusion(list(self.data_buffer))
                
                self.current_confidence = confidence
                self.current_status = status
                
                self.predictions.append(confidence)
                self.timestamps.append(datetime.now().isoformat())
                self.inclusion_flags.append(1 if status == "Inclusion Detected" else 0)
                
                # Emit real-time data to connected clients
                socketio.emit('sensor_data', {
                    'timestamp': datetime.now().isoformat(),
                    'signal_value': float(list(self.signal_data)[-1]) if self.signal_data else 0,
                    'confidence': float(confidence),
                    'status': status,
                    'alert': status == "Inclusion Detected"
                })
            
            time.sleep(0.5)
    
    def start_system(self):
        """Start the real-time detection system"""
        if not self.is_running:
            self.is_running = True
            
            # Start data generation thread
            self.data_thread = threading.Thread(target=self.generate_realtime_data)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.real_time_detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
    
    def stop_system(self):
        """Stop the real-time detection system"""
        self.is_running = False

# Global detector instance
detector = InclusionDetectionSystem()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'is_running': detector.is_running,
        'model_trained': detector.model_trained,
        'current_status': detector.current_status,
        'current_confidence': detector.current_confidence,
        'buffer_size': len(detector.data_buffer)
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the LSTM model"""
    if not detector.model_trained:
        thread = threading.Thread(target=detector.train_model)
        thread.daemon = True
        thread.start()
        return jsonify({'status': 'Training started'})
    else:
        return jsonify({'status': 'Model already trained'})

@app.route('/api/start', methods=['POST'])
def start_detection():
    """Start real-time detection"""
    if detector.model_trained:
        detector.start_system()
        return jsonify({'status': 'Detection started'})
    else:
        return jsonify({'status': 'Model not trained yet'})

@app.route('/api/stop', methods=['POST'])
def stop_detection():
    """Stop real-time detection"""
    detector.stop_system()
    return jsonify({'status': 'Detection stopped'})

@app.route('/api/data')
def get_data():
    """Get recent sensor data for visualization"""
    return jsonify({
        'signal_data': list(detector.signal_data),
        'predictions': list(detector.predictions),
        'timestamps': list(detector.timestamps),
        'inclusion_flags': list(detector.inclusion_flags)
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'status': 'Connected to Inclusion Detection System'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

if __name__ == '__main__':
    print("Starting Flask-based Inclusion Detection System")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)