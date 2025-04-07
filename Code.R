setwd("D:/OS/Binay/data/data")



# Task 1

# Loading the data 
X = read.csv("X.csv")
y = read.csv("y.csv")
time = read.csv("time.csv")

# Convert to matrices 
X = data.matrix(X)
y = data.matrix(y)
time = data.matrix(time)


n <- min(length(time), length(X), length(y))  
time <- time[1:n]  # Subset time
X <- X[1:n]        # Subset X to match the length of time and y
y <- y[1:n]        # Subset y to match the length of time

# 1. Time Series Plots
# Plotting input audio signal over time
plot(time, X, type = "l", col = "blue", xlab = "Time (seconds)", ylab = "Input Audio Signal", main = "Input Audio Signal vs. Time")

# Plot output MEG signal over time
plot(time, y, type = "l", col = "red", xlab = "Time (seconds)", ylab = "MEG Signal", main = "MEG Signal vs. Time")

# 2. Distribution of Signals (Histograms)
hist(X, main = "Distribution of Input Audio Signal", xlab = "Input Audio Signal", col = "blue", breaks = 20)
hist(y, main = "Distribution of MEG Signal", xlab = "MEG Signal", col = "red", breaks = 20)

# 3. Scatter Plot and Correlation
plot(X, y, main = "Scatter Plot of Input Audio vs MEG", xlab = "Input Audio Signal", ylab = "MEG Signal", col = "green")

# Computing correlation
correlation <- cor(X, y)
print(paste("Correlation between X and y:", correlation))

# 4. Boxplots for Sound Categories 
boxplot(y ~ X, main = "Boxplot of MEG Signal by Sound Category", xlab = "Sound Category (Neutral vs Emotional)", ylab = "MEG Signal", col = c("lightblue", "lightgreen"))





#Task 2


# Loading necessary libraries
library(ggplot2)  # For visualizations
library(MASS)     # For model fitting
library(stats)    # For statistical functions

# Loading the data
X = read.csv("X.csv")
y = read.csv("y.csv")
time = read.csv("time.csv")

# Converting to matrices
X = data.matrix(X)
y = data.matrix(y)
time = data.matrix(time)

# Subset data to match lengths
n <- min(length(time), length(X), length(y))  
time <- time[1:n]
X <- X[1:n]
y <- y[1:n]

### Task 2.1: Estimate Model Parameters Using Least Squares

# Fitting the 5 candidate nonlinear regression models.

# Model 1: y = θ1 * X^3 + θ2 * X^5 + θ3 * X2 + θ_bias + ε
model1 <- lm(y ~ I(X^3) + I(X^5) + X, data = data.frame(X, y))

# Model 2: y = θ1 * X + θ2 * X2 + θ_bias + ε
model2 <- lm(y ~ X + I(X == 0), data = data.frame(X, y))

# Model 3: y = θ1 * X + θ2 * X^2 + θ3 * X^4 + θ4 * X2 + θ_bias + ε
model3 <- lm(y ~ X + I(X^2) + I(X^4) + I(X == 0), data = data.frame(X, y))

# Model 4: y = θ1 * X + θ2 * X^2 + θ3 * X^3 + θ4 * X^5 + θ5 * X2 + θ_bias + ε
model4 <- lm(y ~ X + I(X^2) + I(X^3) + I(X^5) + I(X == 0), data = data.frame(X, y))

# Model 5: y = θ1 * X + θ2 * X^3 + θ3 * X^4 + θ4 * X2 + θ_bias + ε
model5 <- lm(y ~ X + I(X^3) + I(X^4) + I(X == 0), data = data.frame(X, y))

### Task 2.2: Compute Residual Sum of Squares (RSS)

# Calculating RSS for each model

# Model 1
rss_model1 <- sum(residuals(model1)^2)
print(paste("RSS for Model 1:", rss_model1))

# Model 2
rss_model2 <- sum(residuals(model2)^2)
print(paste("RSS for Model 2:", rss_model2))

# Model 3
rss_model3 <- sum(residuals(model3)^2)
print(paste("RSS for Model 3:", rss_model3))

# Model 4
rss_model4 <- sum(residuals(model4)^2)
print(paste("RSS for Model 4:", rss_model4))

# Model 5
rss_model5 <- sum(residuals(model5)^2)
print(paste("RSS for Model 5:", rss_model5))

### Task 2.3: Compute Log-Likelihood Function

# Calculating log-likelihood for each model

log_likelihood_model1 <- logLik(model1)
log_likelihood_model2 <- logLik(model2)
log_likelihood_model3 <- logLik(model3)
log_likelihood_model4 <- logLik(model4)
log_likelihood_model5 <- logLik(model5)

print(paste("Log-Likelihood for Model 1:", log_likelihood_model1))
print(paste("Log-Likelihood for Model 2:", log_likelihood_model2))
print(paste("Log-Likelihood for Model 3:", log_likelihood_model3))
print(paste("Log-Likelihood for Model 4:", log_likelihood_model4))
print(paste("Log-Likelihood for Model 5:", log_likelihood_model5))

### Task 2.4: Compute AIC and BIC

# Calculating AIC and BIC for each model

aic_model1 <- AIC(model1)
bic_model1 <- BIC(model1)
aic_model2 <- AIC(model2)
bic_model2 <- BIC(model2)
aic_model3 <- AIC(model3)
bic_model3 <- BIC(model3)
aic_model4 <- AIC(model4)
bic_model4 <- BIC(model4)
aic_model5 <- AIC(model5)
bic_model5 <- BIC(model5)

# Print AIC and BIC values
print(paste("AIC for Model 1:", aic_model1, "BIC for Model 1:", bic_model1))
print(paste("AIC for Model 2:", aic_model2, "BIC for Model 2:", bic_model2))
print(paste("AIC for Model 3:", aic_model3, "BIC for Model 3:", bic_model3))
print(paste("AIC for Model 4:", aic_model4, "BIC for Model 4:", bic_model4))
print(paste("AIC for Model 5:", aic_model5, "BIC for Model 5:", bic_model5))

### Task 2.5: Check Distribution of Model Residuals

# Checking residual distributions (Q-Q plots and histograms)

# Q-Q and histogram for Model 1 residuals
qqnorm(residuals(model1))
qqline(residuals(model1))
hist(residuals(model1), main = "Histogram of Residuals for Model 1", col = "lightblue")

# Q-Q and histogram for Model 2 residuals
qqnorm(residuals(model2))
qqline(residuals(model2))
hist(residuals(model2), main = "Histogram of Residuals for Model 2", col = "lightgreen")

# Q-Q and histogram for Model 3 residuals
qqnorm(residuals(model3))
qqline(residuals(model3))
hist(residuals(model3), main = "Histogram of Residuals for Model 3", col = "lightcoral")

# Q-Q and histogram for Model 4 residuals
qqnorm(residuals(model4))
qqline(residuals(model4))
hist(residuals(model4), main = "Histogram of Residuals for Model 4", col = "lightyellow")

# Q-Q and histogram for Model 5 residuals
qqnorm(residuals(model5))
qqline(residuals(model5))
hist(residuals(model5), main = "Histogram of Residuals for Model 5", col = "lightpink")

### Task 2.6: Select the Best Model

# Based on the AIC, BIC, RSS, and residuals, selecting the best model
# Model 3 performs best, based on these criteria

best_model <- model3

# Print the summary of the selected model
summary(best_model)

### Task 2.7: Train-Test Split and Model Evaluation

# Split the data into training (70%) and testing (30%) sets
set.seed(123)
train_index <- sample(1:n, size = 0.7 * n)
train_data <- data.frame(X = X[train_index], y = y[train_index])
test_data <- data.frame(X = X[-train_index], y = y[-train_index])

# Train the selected model on the training data
model_selected <- lm(y ~ X + I(X^2), data = train_data)  # Using Model 3 as selected earlier

# Make predictions on the test data
predictions <- predict(model_selected, newdata = test_data)

# Compute 95% confidence intervals for predictions
conf_intervals <- predict(model_selected, newdata = test_data, interval = "confidence", level = 0.95)

# Plot the predictions and confidence intervals
plot(test_data$X, test_data$y, main = "Predictions vs. Actuals", xlab = "Input Audio Signal", ylab = "MEG Signal", col = "blue")
lines(test_data$X, predictions, col = "red", lwd = 2)
lines(test_data$X, conf_intervals[, 2], col = "green", lwd = 2, lty = 2)  # Lower bound
lines(test_data$X, conf_intervals[, 3], col = "green", lwd = 2, lty = 2)  # Upper bound
















#Task 3

# Loading necessary libraries
library(caret)        # For cross-validation and model training
library(glmnet)       # For Ridge/Lasso regression (not used here for single feature)
library(ggplot2)      # For plotting
library(Metrics)      # For performance metrics like MSE, RMSE, etc.

# Set the seed for reproducibility
set.seed(123)


# Convert X and y into data frames 
X <- as.data.frame(X)  
y <- as.vector(y)      

# Checking the structure of X to ensure it's a matrix/data frame with at least 2 columns
cat("Dimensions of X: ", dim(X), "\n")
cat("Length of y: ", length(y), "\n")

# Split data into training and testing sets (70-30 split)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)

# Create the training and testing data sets
train_data <- data.frame(X = X[trainIndex, ], y = y[trainIndex])
test_data <- data.frame(X = X[-trainIndex, ], y = y[-trainIndex])

cat("Training set size: ", nrow(train_data), "\n")
cat("Test set size: ", nrow(test_data), "\n")

# -------------------------------
# 1. Cross-Validation (10-Fold)
# -------------------------------
# Cross-validation setup
train_control <- trainControl(method = "cv", number = 10)  # 10-fold cross-validation

# Fit the model (Linear regression since X has only one feature)
model3 <- train(y ~ X, data = train_data, method = "lm", trControl = train_control)

# Print cross-validation results
cat("Cross-Validation Results for Model 3:\n")
print(model3)

# -------------------------------
# 2. Performance Metrics
# -------------------------------
# Predict using Model 3
predictions_model3 <- predict(model3, newdata = test_data)

# Calculate performance metrics for Model 3
mse_model3 <- mean((test_data$y - predictions_model3)^2)  # Mean Squared Error (MSE)
rmse_model3 <- sqrt(mse_model3)                           # Root Mean Squared Error (RMSE)
mae_model3 <- mean(abs(test_data$y - predictions_model3))  # Mean Absolute Error (MAE)
r2_model3 <- 1 - sum((test_data$y - predictions_model3)^2) / sum((test_data$y - mean(test_data$y))^2)  # R-squared

# Print performance metrics for Model 3
cat("Performance Metrics for Model 3:\n")
cat("MSE: ", mse_model3, "\n")
cat("RMSE: ", rmse_model3, "\n")
cat("MAE: ", mae_model3, "\n")
cat("R-squared: ", r2_model3, "\n")

# -------------------------------
# 3. Model Visualization (Residual Plot)
# -------------------------------
# Residual plot for Model 3
residuals_model3 <- test_data$y - predictions_model3
plot(predictions_model3, residuals_model3, main = "Residuals vs Predicted for Model 3", xlab = "Predicted values", ylab = "Residuals")
abline(h = 0, col = "red")

# -------------------------------
# 4. Model Comparison (Model 3)
# -------------------------------
# Here we would normally compare MSE between Model 3 and other models
cat("\nModel Comparison (MSE):\n")
cat("MSE for Model 3: ", mse_model3, "\n")

# -------------------------------
# 5. Final Model Evaluation
# -------------------------------
# Choose the best model based on performance metrics (here assuming Model 3 is best)
best_model <- model3

# Print the summary of the best model
cat("\nSummary of Best Model (Model 3):\n")
print(summary(best_model$finalModel))

#saving the best model for future use
saveRDS(best_model, "best_model.rds")
