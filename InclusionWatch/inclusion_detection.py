import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import threading
from datetime import datetime
from collections import deque

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class InclusionDetectionSystem:
    def __init__(self):
        self.model = None
        self.sequence_length = 50  # Number of time steps to look back
        self.feature_dim = 3  # Number of sensor features (e.g., temperature, pressure, vibration)
        self.is_running = False
        self.data_buffer = deque(maxlen=self.sequence_length)
        self.predictions = []
        self.timestamps = []
        self.signal_data = deque(maxlen=200)  # For visualization
        self.inclusion_flags = deque(maxlen=200)  # For visualization
        
    def generate_synthetic_training_data(self, num_samples=5000):
        """Generate synthetic sensor data with inclusion patterns"""
        print("Generating synthetic training data...")
        
        X = []
        y = []
        
        for i in range(num_samples):
            # Create base signal
            sequence = np.zeros((self.sequence_length, self.feature_dim))
            
            # Decide if this sequence contains an inclusion (30% probability)
            has_inclusion = np.random.random() < 0.3
            
            for t in range(self.sequence_length):
                # Base normal signal with some noise
                sequence[t, 0] = np.sin(t * 0.1) + np.random.normal(0, 0.1)  # Temperature-like
                sequence[t, 1] = np.cos(t * 0.15) + np.random.normal(0, 0.1)  # Pressure-like
                sequence[t, 2] = np.random.normal(0, 0.1)  # Vibration-like
                
                if has_inclusion and t > self.sequence_length - 15:
                    # Add inclusion pattern in last part of sequence
                    inclusion_intensity = 1.5 + np.random.random()
                    sequence[t, 0] += inclusion_intensity * np.sin(t * 0.5)
                    sequence[t, 1] += inclusion_intensity * np.random.normal(0, 0.3)
                    sequence[t, 2] += inclusion_intensity * np.random.normal(0, 0.5)
            
            X.append(sequence)
            y.append(1 if has_inclusion else 0)
        
        return np.array(X), np.array(y)
    
    def preprocess_data(self, X, y=None):
        """Normalize the data for LSTM input"""
        # Normalize features to [-1, 1] range
        X_normalized = np.tanh(X)
        
        if y is not None:
            return X_normalized, y
        return X_normalized
    
    def build_lstm_model(self):
        """Build and compile the LSTM model"""
        print("Building LSTM model...")
        
        model = keras.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.feature_dim)),
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
    
    def train_model(self, epochs=20):
        """Train the LSTM model with synthetic data"""
        print("Training LSTM model for inclusion detection...")
        
        # Generate training data
        X_train, y_train = self.generate_synthetic_training_data(4000)
        X_val, y_val = self.generate_synthetic_training_data(1000)
        
        # Preprocess data
        X_train_norm, y_train = self.preprocess_data(X_train, y_train)
        X_val_norm, y_val = self.preprocess_data(X_val, y_val)
        
        # Build model if not exists
        if self.model is None:
            self.build_lstm_model()
        
        print(f"Training data shape: {X_train_norm.shape}")
        print(f"Validation data shape: {X_val_norm.shape}")
        
        # Train the model
        history = self.model.fit(
            X_train_norm, y_train,
            validation_data=(X_val_norm, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        print("\nTraining completed!")
        val_accuracy = history.history['val_accuracy'][-1]
        print(f"Final validation accuracy: {val_accuracy:.4f}")
        
        return history
    
    def generate_realtime_data(self):
        """Generate real-time sensor data stream"""
        t = 0
        while self.is_running:
            # Generate new data point
            data_point = np.zeros(self.feature_dim)
            
            # Simulate inclusion with 5% probability
            has_inclusion = np.random.random() < 0.05
            
            # Base signal
            data_point[0] = np.sin(t * 0.1) + np.random.normal(0, 0.1)
            data_point[1] = np.cos(t * 0.15) + np.random.normal(0, 0.1)
            data_point[2] = np.random.normal(0, 0.1)
            
            if has_inclusion:
                # Add inclusion pattern
                intensity = 2.0 + np.random.random()
                data_point[0] += intensity * np.sin(t * 0.5)
                data_point[1] += intensity * np.random.normal(0, 0.3)
                data_point[2] += intensity * np.random.normal(0, 0.5)
            
            self.data_buffer.append(data_point)
            self.signal_data.append(data_point[0])  # Store first feature for visualization
            
            t += 1
            time.sleep(0.1)  # 10Hz data rate
    
    def predict_inclusion(self, sequence):
        """Predict inclusion for a given sequence"""
        if len(sequence) < self.sequence_length:
            return 0.0, "Insufficient data"
        
        # Prepare sequence for prediction
        input_sequence = np.array(sequence[-self.sequence_length:]).reshape(1, self.sequence_length, self.feature_dim)
        input_sequence_norm = self.preprocess_data(input_sequence)
        
        # Make prediction
        prediction = self.model.predict(input_sequence_norm, verbose=0)[0][0]
        
        if prediction > 0.5:
            return prediction, "Inclusion Detected"
        else:
            return prediction, "Normal"
    
    def real_time_detection_loop(self):
        """Main real-time detection loop"""
        print("\nStarting real-time inclusion detection...")
        print("=" * 60)
        
        while self.is_running:
            if len(self.data_buffer) >= self.sequence_length:
                # Make prediction
                confidence, status = self.predict_inclusion(list(self.data_buffer))
                
                # Store for visualization
                self.predictions.append(confidence)
                self.timestamps.append(datetime.now())
                self.inclusion_flags.append(1 if status == "Inclusion Detected" else 0)
                
                # Print live results
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Status: {status:15} | Confidence: {confidence:.4f}")
                
                # Alert if inclusion detected
                if status == "Inclusion Detected":
                    print(f"🚨 ALERT: Inclusion detected with {confidence:.1%} confidence!")
            
            time.sleep(0.5)  # Check every 500ms
    
    def setup_visualization(self):
        """Setup matplotlib visualization"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Signal plot
        self.ax1.set_title('Real-Time Sensor Signal (Temperature-like)')
        self.ax1.set_ylabel('Signal Value')
        self.ax1.grid(True)
        self.line1, = self.ax1.plot([], [], 'b-', label='Signal')
        self.inclusion_markers, = self.ax1.plot([], [], 'ro', markersize=8, label='Inclusion Detected')
        self.ax1.legend()
        
        # Prediction confidence plot
        self.ax2.set_title('Inclusion Detection Confidence')
        self.ax2.set_xlabel('Time Steps')
        self.ax2.set_ylabel('Confidence')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True)
        self.ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
        self.line2, = self.ax2.plot([], [], 'g-', label='Confidence')
        self.ax2.legend()
        
        plt.tight_layout()
    
    def animate_plots(self, frame):
        """Animation function for real-time plots"""
        if len(self.signal_data) > 0:
            # Update signal plot
            x_data = list(range(len(self.signal_data)))
            self.line1.set_data(x_data, list(self.signal_data))
            
            # Mark inclusion detections
            inclusion_x = [i for i, flag in enumerate(self.inclusion_flags) if flag == 1]
            inclusion_y = [list(self.signal_data)[i] for i in inclusion_x if i < len(self.signal_data)]
            self.inclusion_markers.set_data(inclusion_x, inclusion_y)
            
            self.ax1.relim()
            self.ax1.autoscale_view()
        
        if len(self.predictions) > 0:
            # Update confidence plot
            x_data = list(range(len(self.predictions)))
            self.line2.set_data(x_data, self.predictions)
            
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        return self.line1, self.line2, self.inclusion_markers
    
    def start_real_time_system(self):
        """Start the complete real-time system"""
        print("Starting Real-Time Inclusion Detection System")
        print("=" * 50)
        
        self.is_running = True
        
        # Start data generation thread
        data_thread = threading.Thread(target=self.generate_realtime_data)
        data_thread.daemon = True
        data_thread.start()
        
        # Start detection thread
        detection_thread = threading.Thread(target=self.real_time_detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Setup and start visualization
        self.setup_visualization()
        ani = animation.FuncAnimation(self.fig, self.animate_plots, interval=100, blit=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            self.stop_system()
    
    def stop_system(self):
        """Stop the real-time system"""
        print("\nStopping Real-Time Inclusion Detection System...")
        self.is_running = False


def main():
    """Main function to run the inclusion detection system"""
    print("Real-Time Inclusion Detection System using LSTM")
    print("=" * 50)
    
    # Initialize the system
    detector = InclusionDetectionSystem()
    
    # Train the model
    print("\n1. Training Phase")
    detector.train_model(epochs=15)
    
    print("\n2. Real-Time Detection Phase")
    print("Starting real-time detection in 3 seconds...")
    print("Press Ctrl+C to stop the system")
    
    time.sleep(3)
    
    # Start real-time detection
    try:
        detector.start_real_time_system()
    except KeyboardInterrupt:
        detector.stop_system()
        print("\nSystem stopped by user.")


if __name__ == "__main__":
    main()