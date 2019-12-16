function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
matrixY = zeros(m,num_labels);
for i = 1:m
    matrixY(i,y(i))=1;
end

X = [ones(m,1),X];

z2 = X*Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2,1),1),a2];
z3 = a2 *Theta2';
a3 = sigmoid(z3);

sumvec = zeros(m,1);
for t = 1:m
    sumvec(t)=log(a3(t,:))*matrixY(t,:)'+log(1-a3(t,:))*(1-matrixY(t,:))';
end

regularTheta1= Theta1(:,2:size(Theta1,2)).^2;
regularTheta2= Theta2(:,2:size(Theta2,2)).^2;

J = -1/m*sum(sumvec)+lambda*(sum(sum(regularTheta1))+sum(sum(regularTheta2)))/(2*m);

DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));
for p = 1:m
    a1 = X(p,:)';
    z2 = Theta1*a1;
    a2 = sigmoid(z2);
    a2 = [1;a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3); %对已知参数进行正向传递
    
    delta3 = a3 - matrixY(p,:)';
    temp = Theta2'*delta3;  
    delta2 = temp(2:end).*sigmoidGradient(z2); %sigmoidGradient是激励函数的导函数
    
    DELTA1 = DELTA1 + delta2*a1';
    DELTA2 = DELTA2 + delta3*a2'; %通过链式法则求得各个参数对结果的误差影响
    
end    

    Theta1_grad = DELTA1/m + lambda*[zeros(size(Theta1,1),1),Theta1(:,(2:end))]/m;
    Theta2_grad = DELTA2/m + lambda*[zeros(size(Theta2,1),1),Theta2(:,(2:end))]/m;%对各个参数进行更新
    





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
