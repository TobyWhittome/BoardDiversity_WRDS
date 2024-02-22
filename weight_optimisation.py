from keras.models import Sequential
from keras.layers import Dense

# Going to need a way to optimise these weights.
#Options:

"""
1. A neural network
  a. Sequential model.
    I need multi input, single output
2. Optimization algorithms
3. Trial and error

"""

weights = [0.2, 0.7, 0.6, 0.8, 0.2, 0.3, 0.5, 0.432]
# Define the model
model = Sequential()
#Outputs a 64 dimensional thing
model.add(Dense(64, input_dim=len(weights), activation='relu'))
#Outputs a 32 dimensional thing
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

print(model.summary())

# Compile the model
# optimizer optimses the loss function. This is the learning
#loss function is your grade
#This is essentially a regression. Which is why we use MSE
#Video uses rmsprop as optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
#Make it the correct dimension. Split it up
# batch size = how many samples we train in tandem
# epochs = how many times we go through the entire dataset.
# validation is important for testing
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
#

# Evaluate the model
mse = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)