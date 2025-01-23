function y = sigm(x)
    % compute sigmoid value
    y = 1 ./ (1 + exp(-x));
end