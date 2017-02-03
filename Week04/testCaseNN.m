Theta1_t = reshape(sin(0 : 0.5 : 5.9), 4, 3);
Theta2_t = reshape(sin(0 : 0.3 : 5.9), 4, 5);
X_t = reshape(sin(1:16), 8, 2);
p_t = predict(Theta1_t, Theta2_t, X_t);