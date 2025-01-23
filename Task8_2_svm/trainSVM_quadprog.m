function model = trainSVM_quadprog(XTrain, YTrain, numClasses, C)
    numSamples = size(XTrain, 1);
    model = cell(numClasses, 1);

    for class = 1:numClasses
        % Binary classification: set current class to 1, others to -1
        binaryY = -ones(numSamples, 1);
        binaryY(YTrain == class) = 1;

        % Compute H matrix
        H = (binaryY * binaryY') .* (XTrain * XTrain');
        H = H + eye(numSamples) * (1/C); % Apply penalty parameter C
        
        f = -ones(numSamples, 1);
        
        % Define constraints
        A = -eye(numSamples);
        a = zeros(numSamples, 1);
        B = binaryY';
        b = 0;

        % Solve for alpha using quadratic programming
        options = optimoptions('quadprog', 'Display', 'off');
        alpha = quadprog(H, f, A, a, B, b, [], [], [], options);

        % Compute weight vector w and bias b
        w = XTrain' * (alpha .* binaryY);
        supportVectors = find(alpha > 1e-4);
        b = mean(binaryY(supportVectors) - XTrain(supportVectors, :) * w);

        % Save the model for the current class
        model{class}.w = w;
        model{class}.b = b;
    end
end
