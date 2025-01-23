function model = trainSVM(XTrain, YTrain, numClasses, C)
    numSamples = size(XTrain, 1);
    model = cell(numClasses, 1);  % Store the model for each class

    for class = 1:numClasses
        % Binary classification: set current class to 1, others to -1
        binaryY = -ones(numSamples, 1);
        binaryY(YTrain == class) = 1;

        % Compute H matrix
        K = XTrain * XTrain';  % Compute Gram matrix
        H = (binaryY * binaryY') .* K;

        % Initialize alpha
        alpha = zeros(numSamples, 1);
        maxIter = 1000;  % Set maximum number of iterations
        for iter = 1:maxIter
            alphaPrev = alpha;  % Save previous alpha
            for i = 1:numSamples
                % Compute error
                f_i = sum(alpha .* binaryY .* K(:, i));
                % Compute gradient
                if binaryY(i) * f_i < 1
                    alpha(i) = alpha(i) + C * (1 - binaryY(i) * f_i);
                end
                % Limit the range of alpha
                alpha(i) = min(max(alpha(i), 0), C);
            end
            % Check for convergence
            if norm(alpha - alphaPrev, 2) < 1e-4
                break;
            end
        end

        % Compute weight vector w
        w = XTrain' * (alpha .* binaryY);
        
        % Compute bias b
        supportVectors = find(alpha > 1e-4);
        model{class}.b = mean(binaryY(supportVectors) - XTrain(supportVectors, :) * w);
        
        % Save the model for the current class
        model{class}.w = w;
    end
end
