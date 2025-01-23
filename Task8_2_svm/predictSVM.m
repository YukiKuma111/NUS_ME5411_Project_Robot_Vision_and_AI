function YPred = predictSVM(model, XTest)
    numClasses = length(model);
    numSamples = size(XTest, 1);
    scores = zeros(numSamples, numClasses);

    for class = 1:numClasses
        w = model{class}.w;
        b = model{class}.b;
        scores(:, class) = XTest * w + b;
    end

    % choose the highest score as ypred
    [~, YPred] = max(scores, [], 2);
end
