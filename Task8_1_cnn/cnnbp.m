function net = cnnbp(net, y)
    % Backpropagation for Convolutional Neural Network
    %
    % Inputs:
    % - net: The neural network structure
    % - y: The expected output (ground truth)
    %
    % Outputs:
    % - net: The updated neural network structure with computed gradients

    n = numel(net.layers); % Number of layers in the network

    % Error computation
    net.e = net.o - y; % Difference between output and ground truth
    % Loss calculation (Mean Squared Error)
    net.L = 1 / 2 * sum(net.e(:) .^ 2) / size(net.e, 2);

    % Delta for the output layer
    net.od = net.e .* (net.o .* (1 - net.o)); % Derivative of sigmoid activation

    % Delta for feature vector
    net.fvd = (net.ffW' * net.od);

    % Delta for the final convolutional layer
    if strcmp(net.layers{n}.type, 'c')
        net.fvd = net.fvd .* (net.fv .* (1 - net.fv)); % Sigmoid derivative for feature vector
    end

    % Reshape feature vector delta to match output map dimensions
    sa = size(net.layers{n}.a{1});
    fvnum = sa(1) * sa(2);
    for j = 1 : numel(net.layers{n}.a)
        net.layers{n}.d{j} = reshape(net.fvd(((j - 1) * fvnum + 1) : j * fvnum, :), sa(1), sa(2), sa(3));
    end

    % Backpropagate delta through the network
    for l = (n - 1) : -1 : 1
        if strcmp(net.layers{l}.type, 'c') % For convolutional layers
            for j = 1 : numel(net.layers{l}.a)
                net.layers{l}.d{j} = net.layers{l}.a{j} .* (1 - net.layers{l}.a{j}) .* ...
                    (expand(net.layers{l + 1}.d{j}, [net.layers{l + 1}.scale, net.layers{l + 1}.scale, 1]) / net.layers{l + 1}.scale ^ 2);
            end
        elseif strcmp(net.layers{l}.type, 's') % For subsampling (pooling) layers
            for i = 1 : numel(net.layers{l}.a)
                z = zeros(size(net.layers{l}.a{1}));
                for j = 1 : numel(net.layers{l + 1}.a)
                    z = z + convn(net.layers{l + 1}.d{j}, rot180(net.layers{l + 1}.k{i}{j}), 'full');
                end
                net.layers{l}.d{i} = z;
            end
        end
    end

    % Compute gradients
    for l = 2 : n
        if strcmp(net.layers{l}.type, 'c') % For convolutional layers
            for j = 1 : numel(net.layers{l}.a)
                for i = 1 : numel(net.layers{l - 1}.a)
                    net.layers{l}.dk{i}{j} = convn(flipall(net.layers{l - 1}.a{i}), net.layers{l}.d{j}, 'valid') / size(net.layers{l}.d{j}, 3);
                end
                net.layers{l}.db{j} = sum(net.layers{l}.d{j}(:)) / size(net.layers{l}.d{j}, 3);
            end
        end
    end

    % Gradients for fully connected layer weights and biases
    net.dffW = net.od * (net.fv)' / size(net.od, 2);
    net.dffb = mean(net.od, 2);

    % Function to rotate a matrix 180 degrees
    function X = rot180(X)
        X = flip(flip(X, 1), 2);
    end
end
