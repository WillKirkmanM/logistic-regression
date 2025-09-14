// Define the Logistic Regression model structure
pub struct LogisticRegression {
    // Weights for each feature
    weights: Vec<f64>,
    // Bias term (intercept)
    bias: f64,
}

impl LogisticRegression {
    // Constructor to initialise a new model
    pub fn new(num_features: usize) -> Self {
        LogisticRegression {
            // Initialise weights to zeros. A better strategy might be random initialisation.
            weights: vec![0.0; num_features],
            bias: 0.0,
        }
    }

    // The Sigmoid function, which squashes the output to a probability between 0 and 1.
    // Formula: σ(z) = 1 / (1 + e^(-z))
    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    // Predicts the probability for a single sample.
    // The linear combination is z = w · x + b
    pub fn predict_proba(&self, features: &[f64]) -> f64 {
        let z = features
            .iter()
            .zip(&self.weights)
            .fold(0.0, |acc, (x, w)| acc + x * w) + self.bias;
        self.sigmoid(z)
    }

    // Predicts the class (0 or 1) based on a threshold (typically 0.5)
    pub fn predict(&self, features: &[f64]) -> u8 {
        if self.predict_proba(features) >= 0.5 {
            1
        } else {
            0
        }
    }

    // Train the model using batch gradient descent.
    pub fn train(
        &mut self,
        features: &[Vec<f64>], // Input features (X)
        labels: &[u8],         // True labels (y)
        learning_rate: f64,    // How much to update weights at each step
        epochs: usize,         // Number of passes through the entire dataset
    ) {
        let num_samples = features.len();
        let num_features = self.weights.len();

        for epoch in 0..epochs {
            let mut dw = vec![0.0; num_features]; // Gradient for weights
            let mut db = 0.0;                     // Gradient for bias

            // --- Calculate gradients ---
            for i in 0..num_samples {
                let y_pred = self.predict_proba(&features[i]);
                let error = y_pred - labels[i] as f64;

                // Accumulate gradients for each feature's weight
                for j in 0..num_features {
                    dw[j] += error * features[i][j];
                }
                // Accumulate gradient for bias
                db += error;
            }

            // --- Update weights and bias ---
            // Average the gradients over all samples
            for j in 0..num_features {
                self.weights[j] -= learning_rate * (dw[j] / num_samples as f64);
            }
            self.bias -= learning_rate * (db / num_samples as f64);
            
            // Optional: Print cost every N epochs to monitor training
            if epoch % 100 == 0 {
                let cost = self.cost(features, labels);
                println!("Epoch {}: Cost = {}", epoch, cost);
            }
        }
    }
    
    // Calculates the Log Loss (Binary Cross-Entropy) for the entire dataset.
    // J(w,b) = -1/m * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]
    fn cost(&self, features: &[Vec<f64>], labels: &[u8]) -> f64 {
        let num_samples = features.len();
        let mut total_cost = 0.0;
        let epsilon = 1e-15; // Small value to avoid log(0)

        for i in 0..num_samples {
            let y_pred = self.predict_proba(&features[i]).max(epsilon).min(1.0 - epsilon);
            let y_true = labels[i] as f64;
            
            total_cost += y_true * y_pred.ln() + (1.0 - y_true) * (1.0 - y_pred).ln();
        }

        -total_cost / num_samples as f64
    }
}

fn main() {
    // Sample dataset: [hours_studied, hours_slept] -> passed_exam (1) or not (0)
    let features = vec![
        vec![2.0, 4.0], // Fails
        vec![4.0, 6.0], // Passes
        vec![5.0, 5.0], // Passes
        vec![1.0, 3.0], // Fails
        vec![8.0, 9.0], // Passes
        vec![9.0, 8.0], // Passes
        vec![3.0, 2.0], // Fails
    ];
    let labels = vec![0, 1, 1, 0, 1, 1, 0];

    // Initialise the model. The number of features is 2.
    let mut model = LogisticRegression::new(2);

    // Train the model
    println!("--- Starting Training ---");
    model.train(&features, &labels, 0.1, 1000);
    println!("--- Training Finished ---\n");

    // Print the learned parameters
    println!("Learned weights: {:?}", model.weights);
    println!("Learned bias: {:.4}\n", model.bias);

    // Make predictions on new data
    let test_student_1 = vec![3.0, 5.0]; // Should be borderline, maybe pass
    let test_student_2 = vec![1.0, 1.0]; // Should fail

    let prob_1 = model.predict_proba(&test_student_1);
    let prediction_1 = model.predict(&test_student_1);

    let prob_2 = model.predict_proba(&test_student_2);
    let prediction_2 = model.predict(&test_student_2);
    
    println!(
        "Student 1 (Features: {:?}) -> Pass Probability: {:.4}, Prediction: {}",
        test_student_1, prob_1, prediction_1
    );
    println!(
        "Student 2 (Features: {:?}) -> Pass Probability: {:.4}, Prediction: {}",
        test_student_2, prob_2, prediction_2
    );
}